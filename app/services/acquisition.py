"""Movie acquisition pipeline: TMDB resolve -> Prowlarr search -> TorBox cached retrieve.

Rate limits (TorBox): 300 req/min general API; createtorrent is limited to
60/hour for uncached torrents, which is irrelevant here because this module
only ever adds torrents that are already cached on TorBox
(``add_only_if_cached="true"``). Prowlarr and TMDB have their own limits;
a simple client-side throttle plus one retry with backoff on HTTP 429 is
applied to all TorBox calls.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from loguru import logger

from app.utils import utils

TMDB_ATTRIBUTION = (
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TORBOX_BASE_URL = "https://api.torbox.app/v1/api"

# Prowlarr movie categories
MOVIE_CATEGORIES = [2000, 2030, 2040, 2045, 2050, 2070, 2080, 2090]

# TorBox client-side throttle: stay well under 300 req/min.
_MIN_REQUEST_INTERVAL = 0.25  # seconds between TorBox calls
_MAX_429_RETRIES = 1
_BACKOFF_ON_429 = 5.0  # seconds

_VIDEO_EXTENSIONS = (".mkv", ".mp4")
_MIN_SIZE_GIB = 1.5
_MAX_SIZE_GIB = 15.0

_BTIH_RE = re.compile(r"btih:([a-fA-F0-9]{40})")
_CAM_TS_RE = re.compile(
    r"\b(cam|hdcam|hdts|ts|telesync|telecine|scr|dvdscr|r5)\b", re.IGNORECASE
)
_RESOLUTION_REJECT_RE = re.compile(r"\b(2160p|720p|480p|4k|uhd|sd)\b", re.IGNORECASE)
_TIER1_RE = re.compile(r"\b(bluray|blu-ray|remux|brrip|bdrip)\b", re.IGNORECASE)
_TIER2_RE = re.compile(r"\b(web-?dl)\b", re.IGNORECASE)


class AcquisitionError(Exception):
    """Raised for user-actionable acquisition failures."""


@dataclass
class AcquireProfile:
    tmdb_api_key: Optional[str] = None  # v3 API key (?api_key=...)
    tmdb_read_access_token: Optional[str] = None  # v4 token (Bearer); preferred when set
    prowlarr_url: Optional[str] = None  # e.g. http://localhost:9696
    prowlarr_api_key: Optional[str] = None
    torbox_api_key: Optional[str] = None
    prefer_quality: str = "1080p-bluray"
    max_candidates: int = 10  # how many ranked releases to try on TorBox
    year: Optional[int] = None  # disambiguator
    mock: bool = False  # mock-acquire mode
    mock_video_path: Optional[str] = None  # CC test video for mock mode
    download_dir: Optional[str] = None  # default utils.video_dir()
    request_timeout: float = 60.0


@dataclass
class AcquiredMedia:
    file_path: str
    tmdb_id: Optional[int]
    imdb_id: Optional[str]
    metadata: dict  # keys: title, year, overview, runtime (minutes), genres (list[str]), poster_path
    release_title: Optional[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_last_torbox_request_ts = 0.0


def _throttle_torbox() -> None:
    """Simple client-side throttle for TorBox calls."""
    global _last_torbox_request_ts
    elapsed = time.monotonic() - _last_torbox_request_ts
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_torbox_request_ts = time.monotonic()


def _torbox_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _redact_url(url: str) -> str:
    """Remove common secret-bearing query params from a URL for logging."""
    return re.sub(
        r"(api_key|token|auth|key)=([^&\s]+)",
        lambda m: f"{m.group(1)}=<redacted>",
        url,
        flags=re.IGNORECASE,
    )


def _torbox_request(
    method: str,
    path: str,
    api_key: str,
    timeout: float,
    **kwargs: Any,
) -> Any:
    """Call the TorBox API, unwrap the envelope, throttle, retry once on 429.

    Returns the ``data`` field of the envelope. Raises AcquisitionError on
    success=false or transport failures. Never logs request URLs containing
    secrets.
    """
    url = f"{TORBOX_BASE_URL}{path}"
    attempts = _MAX_429_RETRIES + 1
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        _throttle_torbox()
        try:
            response = requests.request(
                method,
                url,
                headers=_torbox_headers(api_key),
                timeout=timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise AcquisitionError(
                f"TorBox request failed for {path}: {type(exc).__name__}"
            ) from exc
        if response.status_code == 429:
            last_exc = AcquisitionError(f"TorBox rate-limited on {path} (HTTP 429)")
            if attempt < attempts - 1:
                logger.warning(
                    f"TorBox 429 on {path}; backing off {_BACKOFF_ON_429}s and retrying"
                )
                time.sleep(_BACKOFF_ON_429)
                continue
            break
        if response.status_code != 200:
            raise AcquisitionError(
                f"TorBox {path} returned HTTP {response.status_code}"
            )
        try:
            envelope = response.json()
        except ValueError as exc:
            raise AcquisitionError(f"TorBox {path} returned invalid JSON") from exc
        if not isinstance(envelope, dict):
            raise AcquisitionError(f"TorBox {path} returned an unexpected payload")
        if not envelope.get("success"):
            raise AcquisitionError(
                f"TorBox {path} failed: {envelope.get('error') or envelope.get('detail') or 'unknown error'}"
            )
        return envelope.get("data")
    raise last_exc or AcquisitionError(f"TorBox {path} failed after retries")


def _sanitize_filename(name: str) -> str:
    """Make a release title safe as a Windows/Unix filename."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).strip().strip(".")
    return cleaned or "acquired_video"


def _extract_hash(item: dict[str, Any]) -> Optional[str]:
    """Extract a usable btih info hash from a Prowlarr result item."""
    info_hash = item.get("infoHash")
    if isinstance(info_hash, str) and re.fullmatch(r"[a-fA-F0-9]{40}", info_hash.strip()):
        return info_hash.strip().lower()
    magnet = item.get("magnetUrl") or ""
    match = _BTIH_RE.search(magnet)
    if match:
        return match.group(1).lower()
    return None


# ---------------------------------------------------------------------------
# Release parsing / ranking
# ---------------------------------------------------------------------------


def _parse_release(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Parse one Prowlarr item into a rankable release dict, or None to skip."""
    title = (item.get("title") or "").strip()
    if not title:
        return None
    lowered = title.lower()

    # HARD REJECT: wrong resolution (we only want 1080p) and CAM/TS sources.
    if "1080p" not in lowered:
        return None
    if _RESOLUTION_REJECT_RE.search(title):
        return None
    if _CAM_TS_RE.search(title):
        return None

    info_hash = _extract_hash(item)
    if not info_hash:
        return None

    if _TIER1_RE.search(title):
        tier = 1
    elif _TIER2_RE.search(title):
        tier = 2
    else:
        tier = 3  # WEBRip / HDTV / other

    seeders = item.get("seeders")
    seeders = seeders if isinstance(seeders, int) and seeders > 0 else 0
    size = item.get("size") or 0
    size_gib = size / (1024 ** 3) if isinstance(size, (int, float)) else 0.0
    size_ok = _MIN_SIZE_GIB <= size_gib <= _MAX_SIZE_GIB

    return {
        "title": title,
        "info_hash": info_hash,
        "magnet_url": item.get("magnetUrl"),
        "download_url": item.get("downloadUrl"),
        "seeders": seeders,
        "size": size,
        "tier": tier,
        "size_ok": size_ok,
    }


def rank_releases(releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter and rank raw Prowlarr items (unit-testable, no network).

    Sort order: quality tier asc, then seeders desc (None treated as 0).
    Items whose size falls outside [1.5, 15] GiB are flagged via
    ``size_ok=False`` but not rejected.
    """
    parsed = []
    for item in releases:
        release = _parse_release(item)
        if release is None:
            continue
        parsed.append(release)
    parsed.sort(key=lambda r: (r["tier"], -r["seeders"]))
    for release in parsed:
        if not release["size_ok"]:
            logger.debug(
                f"Size sanity: '{release['title']}' is "
                f"{release['size'] / (1024 ** 3):.2f} GiB, outside "
                f"[{_MIN_SIZE_GIB}, {_MAX_SIZE_GIB}] GiB preferred range"
            )
    return parsed


# ---------------------------------------------------------------------------
# TMDB
# ---------------------------------------------------------------------------


def _tmdb_auth(profile: AcquireProfile) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(headers, extra_params)`` for TMDB auth.

    The v4 Read Access Token (Bearer header) is preferred when present;
    otherwise fall back to the v3 API key as the ``api_key`` query param.
    """
    token = (profile.tmdb_read_access_token or "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}, {}
    api_key = (profile.tmdb_api_key or "").strip()
    if api_key:
        return {}, {"api_key": api_key}
    raise AcquisitionError(
        "TMDB credentials are required for acquisition: set "
        "TMDB_READ_ACCESS_TOKEN (v4) or TMDB_API_KEY (v3)"
    )


def search_tmdb(
    movie_name: str,
    profile: AcquireProfile,
) -> tuple[int, Optional[str], dict[str, Any]]:
    """Resolve a movie name via TMDB.

    Returns ``(tmdb_id, imdb_id, metadata)`` where metadata has keys
    title, year, overview, runtime, genres, poster_path.
    Raises AcquisitionError on failure.
    """
    headers, auth_params = _tmdb_auth(profile)

    params: dict[str, Any] = {
        **auth_params,
        "query": movie_name,
        "include_adult": "false",
    }
    if profile.year:
        params["year"] = profile.year

    try:
        response = requests.get(
            f"{TMDB_BASE_URL}/search/movie",
            params=params,
            headers=headers,
            timeout=profile.request_timeout,
        )
    except requests.RequestException as exc:
        raise AcquisitionError(f"TMDB search request failed: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise AcquisitionError(
            f"TMDB search returned HTTP {response.status_code}; check the TMDB credentials"
        )
    results = response.json().get("results") or []
    if not results:
        raise AcquisitionError(f"TMDB found no movie matching '{movie_name}'")

    best = max(results, key=lambda r: r.get("popularity") or 0.0)
    alternatives = [r for r in results if r is not best][:5]
    for alt in alternatives:
        release_date = alt.get("release_date") or ""
        logger.info(
            f"TMDB alternative: '{alt.get('title')}' "
            f"({release_date[:4] or '????'}, popularity "
            f"{(alt.get('popularity') or 0.0):.2f})"
        )

    tmdb_id = best.get("id")
    if not tmdb_id:
        raise AcquisitionError("TMDB search result missing movie id")

    detail_resp = requests.get(
        f"{TMDB_BASE_URL}/movie/{tmdb_id}",
        params=auth_params,
        headers=headers,
        timeout=profile.request_timeout,
    )
    if detail_resp.status_code != 200:
        raise AcquisitionError(
            f"TMDB movie detail returned HTTP {detail_resp.status_code}"
        )
    detail = detail_resp.json()

    ids_resp = requests.get(
        f"{TMDB_BASE_URL}/movie/{tmdb_id}/external_ids",
        params=auth_params,
        headers=headers,
        timeout=profile.request_timeout,
    )
    if ids_resp.status_code != 200:
        raise AcquisitionError(
            f"TMDB external_ids returned HTTP {ids_resp.status_code}"
        )
    imdb_id = ids_resp.json().get("imdb_id")

    release_date = detail.get("release_date") or ""
    year = int(release_date[:4]) if release_date[:4].isdigit() else None
    metadata = {
        "title": detail.get("title") or movie_name,
        "year": year,
        "overview": detail.get("overview") or "",
        "runtime": detail.get("runtime"),
        "genres": [g.get("name") for g in detail.get("genres") or [] if g.get("name")],
        "poster_path": detail.get("poster_path"),
    }
    logger.info(
        f"TMDB resolved '{movie_name}' -> {metadata['title']} "
        f"({metadata['year']}), tmdb_id={tmdb_id}, imdb_id={imdb_id}"
    )
    return tmdb_id, imdb_id, metadata


# ---------------------------------------------------------------------------
# Prowlarr
# ---------------------------------------------------------------------------


def search_prowlarr(
    title: str,
    year: Optional[int],
    profile: AcquireProfile,
) -> list[dict[str, Any]]:
    """Search Prowlarr for movie releases; returns raw camelCase items."""
    base_url = (profile.prowlarr_url or "").strip().rstrip("/")
    if not base_url:
        raise AcquisitionError("Prowlarr URL is required for acquisition")
    api_key = (profile.prowlarr_api_key or "").strip()
    if not api_key:
        raise AcquisitionError("Prowlarr API key is required for acquisition")

    query = f"{title} {year}" if year else title
    params: list[tuple[str, Any]] = [
        ("query", query),
        ("type", "movie"),
    ]
    params.extend(("categories", c) for c in MOVIE_CATEGORIES)

    try:
        response = requests.get(
            f"{base_url}/api/v1/search",
            headers={"X-Api-Key": api_key},
            params=params,
            timeout=profile.request_timeout,
        )
    except requests.RequestException as exc:
        raise AcquisitionError(f"Prowlarr search request failed: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise AcquisitionError(
            f"Prowlarr search returned HTTP {response.status_code}; check URL and API key"
        )
    try:
        items = response.json()
    except ValueError as exc:
        raise AcquisitionError("Prowlarr search returned invalid JSON") from exc
    if not isinstance(items, list):
        raise AcquisitionError("Prowlarr search returned an unexpected payload")

    usable = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if (item.get("protocol") or "").lower() != "torrent":
            continue
        if not (item.get("magnetUrl") or item.get("infoHash") or item.get("downloadUrl")):
            continue
        usable.append(item)
    logger.info(
        f"Prowlarr returned {len(items)} results, {len(usable)} usable torrents for '{query}'"
    )
    return usable


# ---------------------------------------------------------------------------
# TorBox
# ---------------------------------------------------------------------------


def _check_cached(info_hash: str, profile: AcquireProfile) -> bool:
    data = _torbox_request(
        "GET",
        "/torrents/checkcached",
        profile.torbox_api_key or "",
        profile.request_timeout,
        params={"hash": info_hash, "format": "list", "list_files": "true"},
    )
    if not isinstance(data, list):
        return False
    return any(
        isinstance(entry, dict) and (entry.get("hash") or "").lower() == info_hash
        for entry in data
    )


def _create_torrent(release: dict[str, Any], profile: AcquireProfile) -> Optional[dict[str, Any]]:
    magnet = release.get("magnet_url") or f"magnet:?xt=urn:btih:{release['info_hash']}"
    try:
        data = _torbox_request(
            "POST",
            "/torrents/createtorrent",
            profile.torbox_api_key or "",
            profile.request_timeout,
            data={
                "magnet": magnet,
                "seed": "3",  # don't seed
                "allow_zip": "true",
                "add_only_if_cached": "true",
            },
        )
    except AcquisitionError as exc:
        logger.debug(f"createtorrent rejected '{release['title']}': {exc}")
        return None
    if not isinstance(data, dict) or not data.get("torrent_id"):
        return None
    return data


def _wait_until_ready(torrent_id: Any, profile: AcquireProfile) -> Optional[dict[str, Any]]:
    """Poll mylist every 5s up to 3 min until the torrent is ready."""
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        data = _torbox_request(
            "GET",
            "/torrents/mylist",
            profile.torbox_api_key or "",
            profile.request_timeout,
            params={"id": torrent_id, "bypass_cache": "true"},
        )
        item: Optional[dict[str, Any]] = None
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and str(entry.get("id")) == str(torrent_id):
                    item = entry
                    break
        elif isinstance(data, dict):
            item = data
        if item and (
            item.get("download_finished")
            or item.get("download_present")
            or item.get("cached")
        ):
            return item
        time.sleep(5.0)
    return None


def _pick_video_file(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    files = item.get("files") or []
    candidates = []
    for f in files:
        if not isinstance(f, dict):
            continue
        name = (f.get("name") or f.get("short_name") or "").strip()
        lowered = name.lower()
        if not lowered.endswith(_VIDEO_EXTENSIONS):
            continue
        if "sample" in lowered:
            continue
        candidates.append(f)
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.get("size") or 0)


def _request_download_link(torrent_id: Any, file_id: Any, profile: AcquireProfile) -> str:
    data = _torbox_request(
        "GET",
        "/torrents/requestdl",
        profile.torbox_api_key or "",
        profile.request_timeout,
        params={
            "token": profile.torbox_api_key or "",  # REQUIRED as query param here
            "torrent_id": torrent_id,
            "file_id": file_id,
        },
    )
    if not isinstance(data, str) or not data.startswith("http"):
        raise AcquisitionError("TorBox requestdl did not return a CDN URL")
    # NEVER log this URL or the token.
    logger.info("TorBox download link acquired")
    return data


def _download_file(url: str, dest_path: str, timeout: float) -> None:
    """Stream-download a URL to dest_path; remove partial file on failure."""
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise AcquisitionError(
                    f"CDN download returned HTTP {response.status_code}"
                )
            with open(dest_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
    except Exception:
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except OSError:
            logger.warning(f"Could not remove partial file: {dest_path}")
        raise


def _ffprobe_duration_seconds(file_path: str) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _verify_duration(file_path: str, runtime_minutes: Optional[int]) -> None:
    """Warn (never fail) if probed duration deviates >15% from TMDB runtime."""
    if not runtime_minutes:
        return
    if not shutil.which("ffprobe"):
        logger.debug("ffprobe not on PATH; skipping duration sanity check")
        return
    duration = _ffprobe_duration_seconds(file_path)
    if duration is None:
        logger.debug("ffprobe duration probe failed; skipping sanity check")
        return
    expected = runtime_minutes * 60.0
    if abs(duration - expected) / expected > 0.15:
        logger.warning(
            f"Downloaded duration {duration / 60:.1f} min deviates >15% from "
            f"TMDB runtime {runtime_minutes} min; verify the release"
        )


def torbox_retrieve(
    ranked_releases: list[dict[str, Any]],
    profile: AcquireProfile,
) -> tuple[str, str]:
    """Try ranked releases against TorBox (cached-only) and download the first hit.

    Returns ``(file_path, release_title)``. Raises AcquisitionError if no
    candidate yields a file.
    """
    api_key = (profile.torbox_api_key or "").strip()
    if not api_key:
        raise AcquisitionError("TorBox API key is required for acquisition")
    profile = AcquireProfile(**{**profile.__dict__, "torbox_api_key": api_key})

    download_dir = profile.download_dir or utils.video_dir()
    os.makedirs(download_dir, exist_ok=True)

    candidates = ranked_releases[: max(1, profile.max_candidates)]
    tried = 0
    for release in candidates:
        tried += 1
        info_hash = release["info_hash"]

        if not _check_cached(info_hash, profile):
            logger.debug(f"Not cached on TorBox, skipping: {release['title']}")
            continue

        created = _create_torrent(release, profile)
        if not created:
            logger.debug(f"createtorrent failed, skipping: {release['title']}")
            continue
        torrent_id = created.get("torrent_id")

        item = _wait_until_ready(torrent_id, profile)
        if not item:
            logger.debug(f"Torrent never became ready, skipping: {release['title']}")
            continue

        video_file = _pick_video_file(item)
        if not video_file:
            logger.debug(f"No suitable .mkv/.mp4 file, skipping: {release['title']}")
            continue

        cdn_url = _request_download_link(torrent_id, video_file.get("id"), profile)

        original_name = video_file.get("name") or video_file.get("short_name") or ""
        _, ext = os.path.splitext(original_name)
        ext = ext.lower() if ext.lower() in _VIDEO_EXTENSIONS else ".mkv"
        filename = _sanitize_filename(release["title"]) + ext
        dest_path = os.path.join(download_dir, filename)

        logger.info(f"Downloading '{release['title']}' to {dest_path}")
        _download_file(cdn_url, dest_path, max(profile.request_timeout, 300.0))
        logger.info(f"Download complete: {dest_path}")
        return dest_path, release["title"]

    raise AcquisitionError(
        f"No TorBox candidate yielded a file after trying {tried} release(s)"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def acquire(movie_name: str, profile: AcquireProfile) -> AcquiredMedia:
    """Acquire a movie file: mock, or TMDB -> Prowlarr -> TorBox cached download."""
    if profile.mock:
        mock_path = profile.mock_video_path or ""
        if not mock_path or not os.path.isfile(mock_path):
            raise AcquisitionError(
                f"Mock acquisition requires an existing mock_video_path, got: {mock_path!r}"
            )
        logger.info(f"Mock acquisition: using {mock_path} for '{movie_name}'")
        return AcquiredMedia(
            file_path=mock_path,
            tmdb_id=None,
            imdb_id=None,
            metadata={
                "title": movie_name,
                "year": None,
                "overview": "Mock acquisition for testing.",
                "runtime": None,
                "genres": [],
                "poster_path": None,
            },
            release_title="MOCK",
        )

    tmdb_id, imdb_id, metadata = search_tmdb(movie_name, profile)

    raw_items = search_prowlarr(metadata["title"], metadata["year"], profile)
    ranked = rank_releases(raw_items)
    if not ranked:
        raise AcquisitionError(
            f"No acceptable 1080p releases found for '{metadata['title']}'"
        )

    file_path, release_title = torbox_retrieve(ranked, profile)
    _verify_duration(file_path, metadata.get("runtime"))

    return AcquiredMedia(
        file_path=file_path,
        tmdb_id=tmdb_id,
        imdb_id=imdb_id,
        metadata=metadata,
        release_title=release_title,
    )
