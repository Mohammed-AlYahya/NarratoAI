"""Recap pipeline orchestrator.

Stages (each idempotent, state persisted to ``storage/recap_jobs/<slug>.json``):

1. ACQUIRE   — get the movie file via ``app.services.acquisition``.
2. SUBTITLES — ``--subtitle`` file, embedded-stream extraction via ffmpeg,
               or NarratoAI's fun_asr transcription backends.
3. SCRIPT    — LLM plot analysis + narration copy + script matching via
               ``SubtitleAnalyzerAdapter`` (prompt_category="film_tv_narration").
4. SPLIT     — TTS every segment once to measure real durations, then greedy
               pack consecutive segments into parts of at most
               ``max_part_seconds``. Regenerate shorter copy (up to 3 attempts)
               when the story does not fit the part budget.
5. RENDER    — one ``task.start_subclip_unified`` call per part (never two
               concurrently in one process) -> 1080x1920 vertical parts with
               burned subtitles and BGM.
6. UPLOAD    — YouTube playlist via ``app.services.uploader``.

Heavy imports (``app.*``, acquisition, uploader) are done lazily inside
functions so that ``python -m recap --help`` and the unit tests work without
those modules or credentials on disk.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .config import RecapSettings, load_settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_ACQUIRE = "acquire"
STAGE_SUBTITLES = "subtitles"
STAGE_SCRIPT = "script"
STAGE_SPLIT = "split"
STAGE_RENDER = "render"
STAGE_UPLOAD = "upload"
STAGE_DONE = "done"

STAGE_ORDER = [
    STAGE_ACQUIRE,
    STAGE_SUBTITLES,
    STAGE_SCRIPT,
    STAGE_SPLIT,
    STAGE_RENDER,
    STAGE_UPLOAD,
]

#: ~2.5 spoken English words per second is typical recap pacing.
WORDS_PER_SECOND = 2.5
#: Hard cap on narration-compression attempts before giving up.
MAX_COMPRESS_ATTEMPTS = 3
#: Safety factor applied when scaling the narration word count down.
COMPRESS_FACTOR = 0.9
MIN_WORD_COUNT = 200

_providers_registered = False


# ---------------------------------------------------------------------------
# Small pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    """Slugify a movie name for use in state/output file names.

    >>> slugify("Upgrade")
    'upgrade'
    >>> slugify("The Matrix: Reloaded")
    'the-matrix-reloaded'
    """
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_name).strip("-").lower()
    return slug or "movie"


def split_into_parts(
    items_with_durations: list[dict],
    max_parts: int,
    max_seconds: float,
) -> Optional[list[list[dict]]]:
    """Greedily pack consecutive items into parts of at most ``max_seconds``.

    Args:
        items_with_durations: script items, each with a ``duration`` (seconds).
        max_parts: maximum number of parts allowed.
        max_seconds: maximum total duration of a single part.

    Returns:
        A list of parts (each a list of the original item dicts), or ``None``
        when the items cannot fit within the budget. Items are never
        reordered; a single item longer than ``max_seconds`` means no fit.
    """
    if not items_with_durations:
        return None
    parts: list[list[dict]] = []
    current: list[dict] = []
    current_seconds = 0.0
    for item in items_with_durations:
        duration = float(item.get("duration") or 0.0)
        if duration > max_seconds:
            return None
        if current and current_seconds + duration > max_seconds:
            parts.append(current)
            if len(parts) > max_parts:
                return None
            current = []
            current_seconds = 0.0
        current.append(item)
        current_seconds += duration
    if current:
        parts.append(current)
    if not parts or len(parts) > max_parts:
        return None
    return parts


# ---------------------------------------------------------------------------
# Job-state persistence
# ---------------------------------------------------------------------------


def default_jobs_dir() -> Path:
    """storage/recap_jobs (created on demand). Lazy-imports NarratoAI utils."""
    from app.utils import utils

    return Path(utils.storage_dir("recap_jobs", create=True))


def state_path(slug: str, jobs_dir: Optional[Path] = None) -> Path:
    directory = Path(jobs_dir) if jobs_dir else default_jobs_dir()
    return directory / f"{slug}.json"


def new_state(movie_name: str) -> dict:
    return {
        "movie_name": movie_name,
        "stage": "new",
        "acquired": None,
        "subtitle_path": None,
        "script_path": None,
        "word_count": None,
        "parts": [],
        "playlist_id": None,
        "playlist_url": None,
        "error": None,
    }


def load_state(slug: str, jobs_dir: Optional[Path] = None) -> Optional[dict]:
    path = state_path(slug, jobs_dir)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict, jobs_dir: Optional[Path] = None) -> Path:
    slug = slugify(state["movie_name"])
    path = state_path(slug, jobs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


# ---------------------------------------------------------------------------
# ffmpeg / fonts
# ---------------------------------------------------------------------------


def _find_ffmpeg() -> Optional[str]:
    """Resolve an ffmpeg binary: NarratoAI config first, then PATH."""
    try:
        from app.config import config as app_config

        configured = (app_config.app.get("ffmpeg_path") or "").strip()
        if configured and os.path.isfile(configured):
            return configured
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _pick_font_name() -> str:
    """Pick a subtitle font that works headless.

    ``utils.init_resources()`` materialises at least one font into
    ``resource/fonts`` on Windows (copies simhei.ttf) or downloads Source Han
    Sans elsewhere; ``generate_video._resolve_font_path`` then finds it via
    the relative name we return. Falls back to the "SimHei" family name.
    """
    from app.utils import utils

    try:
        utils.init_resources()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning(f"Font resource init failed, using family fallback: {exc}")
    try:
        for filename in sorted(os.listdir(utils.font_dir())):
            if filename.lower().endswith((".ttf", ".ttc", ".otf")):
                return filename
    except Exception:
        pass
    return "SimHei"


# ---------------------------------------------------------------------------
# Stage 1: ACQUIRE
# ---------------------------------------------------------------------------


def _stage_acquire(
    state: dict,
    settings: RecapSettings,
    *,
    year: Optional[int],
    mock: bool,
    mock_video_path: Optional[str],
) -> None:
    acquired = state.get("acquired")
    if acquired and acquired.get("file_path") and os.path.isfile(acquired["file_path"]):
        logger.info(f"[acquire] already acquired: {acquired['file_path']}")
        return

    from app.services.acquisition import AcquireProfile, acquire
    from app.utils import utils

    profile = AcquireProfile(
        tmdb_api_key=settings.tmdb_api_key,
        prowlarr_url=settings.prowlarr_url,
        prowlarr_api_key=settings.prowlarr_api_key,
        torbox_api_key=settings.torbox_api_key,
        prefer_quality=settings.prefer_quality,
        max_candidates=settings.top_n_candidates,
        year=year,
        mock=mock,
        mock_video_path=mock_video_path,
        download_dir=utils.video_dir(),
        request_timeout=60.0,
    )
    logger.info(f"[acquire] acquiring {state['movie_name']!r} (mock={mock})")
    media = acquire(state["movie_name"], profile)
    state["acquired"] = (
        dataclasses.asdict(media) if dataclasses.is_dataclass(media) else dict(media)
    )
    if not os.path.isfile(state["acquired"].get("file_path", "")):
        raise RuntimeError(
            f"acquire() returned a file_path that does not exist: "
            f"{state['acquired'].get('file_path')!r}"
        )
    logger.info(f"[acquire] acquired -> {state['acquired']['file_path']}")


# ---------------------------------------------------------------------------
# Stage 2: SUBTITLES
# ---------------------------------------------------------------------------


def _extract_embedded_subtitles(video_path: str, out_srt: Path) -> bool:
    """Try to extract the first embedded subtitle stream via ffmpeg."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        logger.warning(
            "[subtitles] no ffmpeg available (config.app.ffmpeg_path unset and "
            "ffmpeg not on PATH); skipping embedded subtitle extraction"
        )
        return False
    cmd = [ffmpeg, "-y", "-i", video_path, "-map", "0:s:0", str(out_srt)]
    logger.debug(f"[subtitles] embedded extraction: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=False
        )
    except Exception as exc:
        logger.warning(f"[subtitles] ffmpeg extraction failed to run: {exc}")
        return False
    ok = (
        result.returncode == 0
        and out_srt.is_file()
        and out_srt.stat().st_size > 0
    )
    if not ok:
        logger.info(f"[subtitles] no usable embedded subtitle stream in {video_path}")
        out_srt.unlink(missing_ok=True)
    return ok


def _fun_asr_config() -> dict:
    """Read the [fun_asr] section via app.config, with a tomllib fallback."""
    try:
        from app.config import config as app_config

        fun_cfg = getattr(app_config, "fun_asr", None)
        if isinstance(fun_cfg, dict) and fun_cfg:
            return fun_cfg
        fun_cfg = app_config._cfg.get("fun_asr", {})  # noqa: SLF001
        if isinstance(fun_cfg, dict):
            return fun_cfg
    except Exception:
        pass
    # Fallback: parse config.toml directly (app.config may be unimportable).
    try:
        import tomllib

        from app.config import config as app_config  # for config_file path

        with open(app_config.config_file, "rb") as f:
            data = tomllib.load(f)
        section = data.get("fun_asr", {})
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _transcribe_with_fun_asr(video_path: str, out_srt: Path) -> Optional[Path]:
    """Transcribe via a configured NarratoAI fun_asr backend, else None."""
    fun_cfg = _fun_asr_config()

    backend = (fun_cfg.get("backend") or "").strip().lower()
    enabled = bool(fun_cfg.get("auto_transcribe_enabled"))
    has_target = bool(
        fun_cfg.get("api_key")
        or fun_cfg.get("api_url")
        or fun_cfg.get("firered_api_url")
    )
    if backend not in {"local", "firered", "bailian"} or not (enabled or has_target):
        return None

    from app.services import fun_asr_subtitle

    logger.info(f"[subtitles] transcribing with fun_asr backend={backend!r}")
    if backend == "local":
        created = fun_asr_subtitle.create_with_local_fun_asr(
            local_file=video_path,
            subtitle_file=str(out_srt),
            api_url=fun_cfg.get("api_url", "") or fun_asr_subtitle.LOCAL_FUN_ASR_API_URL,
            hotword=fun_cfg.get("hotword", "") or "",
            enable_spk=fun_cfg.get("enable_spk"),
        )
    elif backend == "firered":
        created = fun_asr_subtitle.create_with_local_firered_asr(
            local_file=video_path,
            subtitle_file=str(out_srt),
            api_url=fun_cfg.get("firered_api_url", "")
            or fun_asr_subtitle.LOCAL_FIRERED_ASR_API_URL,
        )
    else:  # bailian
        created = fun_asr_subtitle.create_with_fun_asr(
            local_file=video_path,
            subtitle_file=str(out_srt),
            api_key=fun_cfg.get("api_key", "") or "",
        )
    if created and os.path.isfile(created) and os.path.getsize(created) > 0:
        return Path(created)
    return None


def _stage_subtitles(
    state: dict,
    settings: RecapSettings,
    *,
    subtitle_arg: Optional[str],
    jobs_dir: Path,
) -> None:
    existing = state.get("subtitle_path")
    if existing and os.path.isfile(existing) and os.path.getsize(existing) > 0:
        logger.info(f"[subtitles] already have subtitles: {existing}")
        return

    slug = slugify(state["movie_name"])
    out_srt = jobs_dir / f"{slug}.srt"

    # (a) explicit CLI subtitle file
    if subtitle_arg:
        src = Path(subtitle_arg)
        if not src.is_file():
            raise FileNotFoundError(f"--subtitle file not found: {subtitle_arg}")
        shutil.copyfile(src, out_srt)
        state["subtitle_path"] = str(out_srt)
        logger.info(f"[subtitles] using --subtitle {src}")
        return

    video_path = state["acquired"]["file_path"]

    # (b) embedded subtitle stream
    if _extract_embedded_subtitles(video_path, out_srt):
        state["subtitle_path"] = str(out_srt)
        logger.info(f"[subtitles] extracted embedded subtitles -> {out_srt}")
        return

    # (c) fun_asr backends
    try:
        created = _transcribe_with_fun_asr(video_path, out_srt)
    except Exception as exc:
        logger.warning(f"[subtitles] fun_asr transcription failed: {exc}")
        created = None
    if created:
        state["subtitle_path"] = str(created)
        logger.info(f"[subtitles] transcribed via fun_asr -> {created}")
        return

    # (d) give up with a clear message
    raise RuntimeError(
        "No subtitles available for the acquired video. Pass --subtitle <file.srt>, "
        "use a release with an embedded subtitle stream, or enable a fun_asr "
        "backend in config.toml ([fun_asr] auto_transcribe_enabled = true)."
    )


def _read_subtitle_content(state: dict) -> str:
    from app.services.subtitle_text import read_subtitle_text

    decoded = read_subtitle_text(state["subtitle_path"])
    content = decoded.text if hasattr(decoded, "text") else str(decoded)
    if not content.strip():
        raise RuntimeError(f"Subtitle file is empty: {state['subtitle_path']}")
    return content


# ---------------------------------------------------------------------------
# Stage 3: SCRIPT (LLM)
# ---------------------------------------------------------------------------


def _register_providers_once() -> None:
    global _providers_registered
    if _providers_registered:
        return
    from app.services.llm.providers import register_all_providers

    register_all_providers()
    _providers_registered = True


def _llm_credentials() -> tuple[str, str, str, str]:
    """(provider, api_key, model, base_url) from NarratoAI config.toml."""
    from app.config import config as app_config

    provider = (app_config.app.get("text_llm_provider") or "").strip()
    if not provider:
        raise RuntimeError(
            "config.toml is missing app.text_llm_provider; configure the "
            "text_llm_* keys first (see README-RECAP.md)."
        )
    api_key = app_config.app.get(f"text_{provider}_api_key") or ""
    model = app_config.app.get(f"text_{provider}_model_name") or ""
    base_url = app_config.app.get(f"text_{provider}_base_url") or ""
    if not api_key or not model:
        raise RuntimeError(
            f"config.toml is missing text_{provider}_api_key / "
            f"text_{provider}_model_name; configure them first."
        )
    return provider, api_key, model, base_url


def _make_analyzer():
    """Create a SubtitleAnalyzerAdapter from NarratoAI text LLM config."""
    _register_providers_once()
    from app.services.llm.migration_adapter import SubtitleAnalyzerAdapter

    _provider, api_key, model, base_url = _llm_credentials()
    return SubtitleAnalyzerAdapter(
        api_key,
        model,
        base_url,
        provider=None,
        prompt_category="film_tv_narration",
    )


def _tmdb_facts_block(state: dict) -> str:
    """Additive TMDB metadata hook prepended to the plot analysis."""
    acquired = state.get("acquired") or {}
    metadata = acquired.get("metadata") or {}
    title = metadata.get("title") or metadata.get("name") or state["movie_name"]
    year = ""
    release = metadata.get("release_date") or ""
    if release:
        year = str(release)[:4]
    genres = _genre_names(metadata.get("genres"))
    overview = metadata.get("overview") or ""
    lines = [f"Title: {title}"]
    if year:
        lines.append(f"Year: {year}")
    if genres:
        lines.append(f"Genres: {', '.join(genres)}")
    if overview:
        lines.append(f"Overview: {overview}")
    if len(lines) <= 1:
        return ""
    return "Movie facts (TMDB): " + "\n".join(lines) + "\n\n"


def _genre_names(genres: Any) -> list[str]:
    names: list[str] = []
    if isinstance(genres, (list, tuple)):
        for genre in genres:
            if isinstance(genre, dict):
                name = genre.get("name")
            else:
                name = str(genre)
            if name:
                names.append(str(name))
    return names


def _drama_genre(state: dict) -> str:
    metadata = (state.get("acquired") or {}).get("metadata") or {}
    names = _genre_names(metadata.get("genres"))
    return names[0] if names else "Drama"


def _require_success(result: dict, what: str) -> dict:
    if not isinstance(result, dict) or result.get("status") != "success":
        message = result.get("message") if isinstance(result, dict) else repr(result)
        raise RuntimeError(f"LLM {what} failed: {message}")
    return result


def _parse_script_items(narration_script: Any) -> list[dict]:
    """Parse the LLM narration-script JSON into an items list."""
    if isinstance(narration_script, (dict, list)):
        payload = narration_script
    else:
        payload = json.loads(str(narration_script))
    if isinstance(payload, dict):
        items = payload.get("items")
        if items is None:
            # Some models return the list under another single key.
            list_values = [v for v in payload.values() if isinstance(v, list)]
            items = list_values[0] if len(list_values) == 1 else None
    else:
        items = payload
    if not isinstance(items, list) or not items:
        raise RuntimeError("LLM narration script JSON has no non-empty 'items' list")
    cleaned: list[dict] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"narration script item #{idx} is not an object")
        normalised = dict(item)
        normalised["_id"] = int(normalised.get("_id") or idx)
        normalised["OST"] = int(normalised.get("OST", 0) or 0)
        for key in ("timestamp", "picture", "narration"):
            normalised.setdefault(key, "")
        if not str(normalised["narration"]).strip():
            raise RuntimeError(f"narration script item #{idx} has empty narration")
        cleaned.append(normalised)
    return cleaned


def _generate_script_items(
    analyzer,
    state: dict,
    settings: RecapSettings,
    subtitle_content: str,
    word_count: int,
) -> list[dict]:
    """Run plot-analysis -> narration-copy -> script-matching once."""
    short_name = state["movie_name"]
    drama_genre = _drama_genre(state)

    analysis_result = _require_success(
        analyzer.analyze_subtitle(subtitle_content), "plot analysis"
    )
    plot_analysis = analysis_result.get("analysis") or ""
    if not isinstance(plot_analysis, str):
        plot_analysis = json.dumps(plot_analysis, ensure_ascii=False)
    plot_analysis = _tmdb_facts_block(state) + plot_analysis

    copy_result = _require_success(
        analyzer.generate_narration_copy(
            short_name=short_name,
            plot_analysis=plot_analysis,
            subtitle_content=subtitle_content,
            temperature=0.7,
            narration_language=settings.narration_language,
            drama_genre=drama_genre,
            narration_word_count=int(word_count),
        ),
        "narration copy",
    )
    narration_copy = copy_result.get("narration_copy") or ""
    if not narration_copy.strip():
        raise RuntimeError("LLM returned an empty narration copy")

    match_result = _require_success(
        analyzer.match_narration_copy_to_script(
            short_name=short_name,
            plot_analysis=plot_analysis,
            subtitle_content=subtitle_content,
            narration_copy=narration_copy,
            temperature=0.3,
            narration_language=settings.narration_language,
            drama_genre=drama_genre,
            original_sound_ratio=0,
        ),
        "script matching",
    )
    return _parse_script_items(match_result.get("narration_script"))


def _stage_script(
    state: dict,
    settings: RecapSettings,
    *,
    jobs_dir: Path,
) -> None:
    if state.get("script_path") and os.path.isfile(state["script_path"]):
        logger.info(f"[script] already generated: {state['script_path']}")
        return

    from app.utils import utils

    subtitle_content = _read_subtitle_content(state)
    analyzer = _make_analyzer()
    word_count = int(settings.max_parts * settings.max_part_seconds * WORDS_PER_SECOND)
    logger.info(f"[script] generating narration script (~{word_count} words target)")
    items = _generate_script_items(
        analyzer, state, settings, subtitle_content, word_count
    )

    slug = slugify(state["movie_name"])
    script_path = Path(utils.script_dir()) / f"recap_{slug}.json"
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)
    state["script_path"] = str(script_path)
    state["word_count"] = sum(
        len(str(item.get("narration", "")).split()) for item in items
    )
    logger.info(f"[script] wrote {len(items)} items -> {script_path}")


def _load_script_items(state: dict) -> list[dict]:
    with open(state["script_path"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"script file has no items: {state['script_path']}")
    return items


# ---------------------------------------------------------------------------
# Stage 4: MEASURE & SPLIT
# ---------------------------------------------------------------------------


def _estimate_item_duration(item: dict) -> float:
    """Duration fallback for items TTS skipped (OST==1) or failed on."""
    timestamp = str(item.get("timestamp") or "")
    match = re.match(
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
        timestamp,
    )
    if match:
        def _to_seconds(ts: str) -> float:
            hh, mm, rest = ts.replace(",", ".").split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(rest)

        try:
            delta = _to_seconds(match.group(2)) - _to_seconds(match.group(1))
            if delta > 0:
                return delta
        except Exception:
            pass
    words = len(str(item.get("narration", "")).split())
    return max(1.0, words * 0.35)


def _measure_durations(state: dict, settings: RecapSettings, slug: str) -> list[dict]:
    """TTS every item once; per-segment audio duration is the source of truth."""
    from app.services import voice

    items = _load_script_items(state)
    measure_task_id = f"recap-{slug}-measure"
    logger.info(f"[split] measuring {len(items)} segments via TTS ({measure_task_id})")
    tts_results = voice.tts_multiple(
        task_id=measure_task_id,
        list_script=items,
        voice_name=settings.resolve_tts_voice_name(),
        voice_rate=1.0,
        voice_pitch=1.0,
        tts_engine=settings.tts_engine,
    )
    durations_by_id = {r["_id"]: float(r["duration"]) for r in tts_results}
    measured: list[dict] = []
    for item in items:
        merged = dict(item)
        duration = durations_by_id.get(item["_id"])
        if duration is None or duration <= 0:
            duration = _estimate_item_duration(item)
            logger.warning(
                f"[split] no TTS duration for _id={item['_id']}, estimated {duration:.1f}s"
            )
        merged["duration"] = duration
        measured.append(merged)
    return measured


def _stage_split(
    state: dict,
    settings: RecapSettings,
    *,
    jobs_dir: Path,
) -> None:
    if state.get("parts"):
        logger.info(f"[split] already split into {len(state['parts'])} part(s)")
        return

    slug = slugify(state["movie_name"])
    subtitle_content = _read_subtitle_content(state)
    analyzer = None  # created lazily only if regeneration is needed

    word_count = state.get("word_count") or int(
        settings.max_parts * settings.max_part_seconds * WORDS_PER_SECOND
    )
    budget_seconds = settings.max_parts * settings.max_part_seconds

    parts: Optional[list[list[dict]]] = None
    for attempt in range(1, MAX_COMPRESS_ATTEMPTS + 1):
        measured = _measure_durations(state, settings, slug)
        actual_seconds = sum(item["duration"] for item in measured)
        parts = split_into_parts(measured, settings.max_parts, settings.max_part_seconds)
        if parts is not None:
            logger.info(
                f"[split] fits: {len(parts)} part(s), {actual_seconds:.1f}s total "
                f"(attempt {attempt})"
            )
            break
        logger.warning(
            f"[split] {actual_seconds:.1f}s of narration does not fit "
            f"{settings.max_parts} x {settings.max_part_seconds}s "
            f"(attempt {attempt}/{MAX_COMPRESS_ATTEMPTS})"
        )
        if attempt == MAX_COMPRESS_ATTEMPTS:
            break
        # Regenerate shorter narration copy; never ship a dangling ending.
        ratio = budget_seconds / max(actual_seconds, 1.0)
        word_count = max(MIN_WORD_COUNT, int(word_count * ratio * COMPRESS_FACTOR))
        logger.info(f"[split] regenerating narration copy (~{word_count} words)")
        if analyzer is None:
            analyzer = _make_analyzer()
        items = _generate_script_items(
            analyzer, state, settings, subtitle_content, word_count
        )
        from app.utils import utils

        with open(state["script_path"], "w", encoding="utf-8") as f:
            json.dump({"items": items}, f, ensure_ascii=False, indent=2)
        state["word_count"] = sum(
            len(str(item.get("narration", "")).split()) for item in items
        )

    if parts is None:
        raise RuntimeError(
            f"The recap narration cannot be compressed into "
            f"{settings.max_parts} part(s) of {settings.max_part_seconds}s after "
            f"{MAX_COMPRESS_ATTEMPTS} attempts. Raise --max-parts / "
            f"--max-part-seconds or shorten the source. (Hard rule: the story "
            f"must conclude within the part budget.)"
        )

    state["parts"] = [
        {
            "n": index + 1,
            "indices": list(range(offset, offset + len(part))),
            "duration": round(sum(item["duration"] for item in part), 2),
            "script_path": None,
            "task_id": f"recap-{slug}-p{index + 1}",
            "video_path": None,
            "video_id": None,
            "status": "pending",
        }
        for index, (offset, part) in enumerate(_with_offsets(parts))
    ]
    logger.info(f"[split] planned {len(state['parts'])} part(s)")


def _with_offsets(parts: list[list[dict]]):
    offset = 0
    for part in parts:
        yield offset, part
        offset += len(part)


# ---------------------------------------------------------------------------
# Stage 5: RENDER
# ---------------------------------------------------------------------------


def _build_clip_params(
    state: dict,
    settings: RecapSettings,
    part: dict,
    part_script_path: Path,
):
    from app.models.schema import VideoAspect, VideoClipParams
    from app.utils import utils

    song_dir = utils.song_dir()
    has_songs = False
    try:
        has_songs = any(
            name.lower().endswith((".mp3", ".flac", ".wav"))
            for name in os.listdir(song_dir)
        )
    except Exception:
        has_songs = False

    return VideoClipParams(
        video_clip_json_path=str(part_script_path),
        video_origin_path=state["acquired"]["file_path"],
        original_subtitle_path=state.get("subtitle_path") or "",
        video_aspect=VideoAspect.portrait,
        voice_name=settings.resolve_tts_voice_name(),
        voice_rate=1.0,
        voice_pitch=1.0,
        tts_engine=settings.tts_engine,
        tts_volume=1.0,          # narration loud
        original_volume=0.1,     # movie audio ducked hard
        bgm_type="random" if has_songs else "",
        bgm_file="",
        bgm_volume=0.25 if has_songs else 0.0,
        subtitle_enabled=True,
        subtitle_position="bottom",
        font_name=_pick_font_name(),
        font_size=36,
        text_fore_color="white",
        stroke_color="black",
        stroke_width=1.5,
        n_threads=4,
    )


def _stage_render(
    state: dict,
    settings: RecapSettings,
    *,
    jobs_dir: Path,
) -> None:
    from app.services import task
    from app.utils import utils

    slug = slugify(state["movie_name"])
    all_items = _load_script_items(state)

    for part in state["parts"]:
        video_path = part.get("video_path")
        if part.get("status") == "rendered" and video_path and os.path.isfile(video_path):
            logger.info(f"[render] part {part['n']} already rendered: {video_path}")
            continue

        # Part script JSON: plain list (task.start_subclip_unified expects a
        # bare list), _id renumbered from 1, timestamps kept (they reference
        # the source movie).
        part_items = [dict(all_items[i]) for i in part["indices"]]
        for new_id, item in enumerate(part_items, start=1):
            item["_id"] = new_id
        part_script_path = Path(utils.script_dir()) / f"recap_{slug}_p{part['n']}.json"
        with open(part_script_path, "w", encoding="utf-8") as f:
            json.dump(part_items, f, ensure_ascii=False, indent=2)
        part["script_path"] = str(part_script_path)

        params = _build_clip_params(state, settings, part, part_script_path)
        logger.info(
            f"[render] rendering part {part['n']}/{len(state['parts'])} "
            f"({len(part_items)} segments, task {part['task_id']})"
        )
        # Never run two of these concurrently in one process.
        task.start_subclip_unified(part["task_id"], params)

        combined = Path(utils.task_dir(part["task_id"])) / "combined.mp4"
        if not combined.is_file():
            candidates = sorted(
                Path(utils.task_dir(part["task_id"])).glob("*.mp4"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise RuntimeError(
                    f"render produced no .mp4 in {utils.task_dir(part['task_id'])}"
                )
            combined = candidates[0]
            logger.warning(f"[render] combined.mp4 missing; using {combined.name}")

        final_path = jobs_dir / f"{slug}_part{part['n']}.mp4"
        shutil.copyfile(combined, final_path)
        part["video_path"] = str(final_path)
        part["status"] = "rendered"
        save_state(state, jobs_dir)  # persist after every part
        logger.info(f"[render] part {part['n']} -> {final_path}")


# ---------------------------------------------------------------------------
# Stage 6: UPLOAD
# ---------------------------------------------------------------------------


def _build_description(state: dict) -> str:
    from app.services.acquisition import TMDB_ATTRIBUTION

    metadata = (state.get("acquired") or {}).get("metadata") or {}
    overview = metadata.get("overview") or ""
    lines = []
    if overview:
        lines.append(overview)
        lines.append("")
    lines.append("Recap parts:")
    for part in state["parts"]:
        lines.append(f"  Part {part['n']} ({part.get('duration', '?')}s)")
    lines.append("")
    lines.append(TMDB_ATTRIBUTION)
    return "\n".join(lines)


def _apply_uploaded_parts(state: dict, uploaded_parts: Any) -> None:
    """Map uploader-returned part records back onto state["parts"]."""
    if not uploaded_parts:
        return
    by_n = {p["n"]: p for p in state["parts"]}
    for record in uploaded_parts:
        n = None
        video_id = None
        if isinstance(record, dict):
            n = record.get("n") or record.get("part") or record.get("part_number")
            video_id = record.get("video_id") or record.get("id")
        else:
            n = getattr(record, "n", None) or getattr(record, "part_number", None)
            video_id = getattr(record, "video_id", None)
            if n is None and isinstance(record, (list, tuple)) and len(record) >= 2:
                n, video_id = record[0], record[1]
        if n in by_n and video_id:
            by_n[n]["video_id"] = video_id
            by_n[n]["status"] = "uploaded"


def _stage_upload(
    state: dict,
    settings: RecapSettings,
    *,
    year: Optional[int],
    no_upload: bool,
    jobs_dir: Path,
) -> None:
    if no_upload:
        logger.info("[upload] skipped (--no-upload)")
        return
    if state.get("playlist_url"):
        logger.info(f"[upload] already uploaded: {state['playlist_url']}")
        return
    if not settings.youtube_client_secrets:
        raise RuntimeError(
            "YOUTUBE_CLIENT_SECRETS is not set; pass --no-upload to skip upload "
            "or configure the YouTube OAuth client (see README-RECAP.md)."
        )

    pending = [
        p
        for p in state["parts"]
        if p.get("status") == "rendered"
        and p.get("video_path")
        and os.path.isfile(p["video_path"])
        and not p.get("video_id")
    ]
    already = [p for p in state["parts"] if p.get("video_id")]
    if not pending and already:
        logger.info("[upload] all parts already have video IDs")
        return
    if not pending and not already:
        raise RuntimeError("No rendered parts available to upload")

    from app.services.uploader import YouTubeUploader

    def _progress(done: Any, total: Any = None) -> None:
        logger.info(f"[upload] progress: {done}/{total}" if total else f"[upload] {done}")

    uploader = YouTubeUploader(
        client_secrets_path=settings.youtube_client_secrets,
        token_path=settings.youtube_token_path,
        privacy=settings.privacy,
    )
    part_files = [(p["n"], p["video_path"]) for p in pending]
    logger.info(
        f"[upload] uploading {len(part_files)} part(s) as {settings.privacy}"
    )
    result = uploader.upload_movie_parts(
        part_files=part_files,
        movie_title=state["movie_name"],
        year=year,
        description=_build_description(state),
        progress_callback=_progress,
    )
    _apply_uploaded_parts(state, getattr(result, "parts", None))
    for part in state["parts"]:
        if part.get("video_id") and part.get("status") == "rendered":
            part["status"] = "uploaded"
    state["playlist_id"] = getattr(result, "playlist_id", None)
    state["playlist_url"] = getattr(result, "playlist_url", None)
    save_state(state, jobs_dir)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    movie_name: str,
    *,
    year: Optional[int] = None,
    privacy: Optional[str] = None,
    mock_acquire: bool = False,
    mock_video: Optional[str] = None,
    subtitle: Optional[str] = None,
    no_upload: bool = False,
    language: Optional[str] = None,
    max_parts: Optional[int] = None,
    max_part_seconds: Optional[int] = None,
    stage: Optional[str] = None,
    settings: Optional[RecapSettings] = None,
    jobs_dir: Optional[Path] = None,
) -> dict:
    """Run the recap pipeline; returns the final persisted job state.

    Args:
        stage: when given, run only that single stage (debug helper; assumes
            earlier stages have already persisted their state).
    """
    settings = settings or load_settings()
    if privacy:
        settings.privacy = privacy
        if privacy not in ("private", "unlisted", "public"):
            raise ValueError(f"invalid privacy: {privacy!r}")
    if language:
        settings.narration_language = language
    if max_parts:
        settings.max_parts = int(max_parts)
    if max_part_seconds:
        settings.max_part_seconds = int(max_part_seconds)

    slug = slugify(movie_name)
    jobs_dir = Path(jobs_dir) if jobs_dir else default_jobs_dir()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(slug, jobs_dir) or new_state(movie_name)
    state["error"] = None

    stage_funcs = {
        STAGE_ACQUIRE: lambda: _stage_acquire(
            state, settings, year=year, mock=mock_acquire, mock_video_path=mock_video
        ),
        STAGE_SUBTITLES: lambda: _stage_subtitles(
            state, settings, subtitle_arg=subtitle, jobs_dir=jobs_dir
        ),
        STAGE_SCRIPT: lambda: _stage_script(state, settings, jobs_dir=jobs_dir),
        STAGE_SPLIT: lambda: _stage_split(state, settings, jobs_dir=jobs_dir),
        STAGE_RENDER: lambda: _stage_render(state, settings, jobs_dir=jobs_dir),
        STAGE_UPLOAD: lambda: _stage_upload(
            state, settings, year=year, no_upload=no_upload, jobs_dir=jobs_dir
        ),
    }

    if stage:
        if stage not in stage_funcs:
            raise ValueError(
                f"unknown stage {stage!r}; expected one of {sorted(stage_funcs)}"
            )
        stages_to_run = [stage]
    else:
        stages_to_run = list(STAGE_ORDER)

    current_stage = stages_to_run[0]
    try:
        for current_stage in stages_to_run:
            state["stage"] = current_stage
            save_state(state, jobs_dir)
            stage_funcs[current_stage]()
        if not stage:
            state["stage"] = STAGE_DONE
    except Exception as exc:
        state["stage"] = f"failed:{current_stage}"
        state["error"] = f"{type(exc).__name__}: {exc}"
        save_state(state, jobs_dir)
        logger.error(f"recap pipeline failed at stage {current_stage!r}: {exc}")
        raise RuntimeError(
            f"recap pipeline failed at stage {current_stage!r}: {exc}"
        ) from exc

    save_state(state, jobs_dir)
    return state
