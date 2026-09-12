"""YouTube Data API v3 uploader: OAuth2 installed-app flow, uploads, playlists.

Quota note: a video insert costs 1,600 quota units; the default daily quota
is 10,000 units, i.e. roughly one 4-part movie per day. Playlist creation and
playlist item insertion cost 50 units each.

IMPORTANT: the Google Cloud OAuth consent-screen app must be switched from
"Testing" to "In production"; otherwise the issued refresh token expires
every 7 days and uploads start failing with an invalid_grant error.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from loguru import logger

from app.services.acquisition import TMDB_ATTRIBUTION

YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_SCOPE = "https://www.googleapis.com/auth/youtube"

SCOPES = [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_SCOPE]

_VALID_PRIVACY = {"private", "unlisted", "public"}
_CATEGORY_ENTERTAINMENT = "24"
_CHUNK_SIZE = 8 * 1024 * 1024
_MAX_RETRIES = 3
_RETRYABLE_HTTP_CODES = {500, 502, 503, 504}


class UploadError(Exception):
    """Raised for user-actionable YouTube upload failures."""


@dataclass
class UploadedPart:
    part_number: int
    video_id: str
    url: str


@dataclass
class PlaylistResult:
    playlist_id: str
    playlist_url: str
    parts: list[UploadedPart] = field(default_factory=list)


class YouTubeUploader:
    """Uploads movie parts to YouTube and collects them in a playlist."""

    def __init__(
        self,
        client_secrets_path: str,
        token_path: str,
        privacy: str = "unlisted",
    ):
        if not client_secrets_path or not os.path.isfile(client_secrets_path):
            raise UploadError(
                f"Client secrets file not found: {client_secrets_path!r}"
            )
        if privacy not in _VALID_PRIVACY:
            raise UploadError(
                f"Invalid privacy '{privacy}'; expected one of {sorted(_VALID_PRIVACY)}"
            )
        # Default unlisted — NEVER public by default.
        self.client_secrets_path = client_secrets_path
        self.token_path = token_path
        self.privacy = privacy
        self._youtube: Optional[Any] = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _get_credentials(self) -> Credentials:
        """Load/refresh/persist OAuth2 credentials (installed-app flow).

        First run opens a local browser via run_local_server; afterwards the
        persisted refresh token makes the flow headless.
        """
        creds: Optional[Credentials] = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        if creds and creds.valid:
            return creds

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("Refreshed YouTube OAuth token")
            except Exception as exc:
                raise UploadError(
                    "YouTube token refresh failed (invalid_grant?). If the OAuth "
                    "consent app is still in 'Testing' mode the refresh token "
                    "expires every 7 days — switch it to 'In production'."
                ) from exc
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secrets_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            except Exception as exc:
                raise UploadError(f"OAuth2 authorization flow failed: {exc}") from exc
            logger.info("Completed interactive YouTube OAuth authorization")

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.token_path)), exist_ok=True)
            with open(self.token_path, "w", encoding="utf-8") as fh:
                fh.write(creds.to_json())
        except OSError as exc:
            raise UploadError(f"Could not persist YouTube token: {exc}") from exc
        return creds

    def _client(self) -> Any:
        if self._youtube is None:
            creds = self._get_credentials()
            self._youtube = build("youtube", "v3", credentials=creds)
        return self._youtube

    # ------------------------------------------------------------------
    # Low-level API calls
    # ------------------------------------------------------------------

    def upload_video(
        self,
        file_path: str,
        title: str,
        description: str,
        privacy: Optional[str] = None,
    ) -> str:
        """Upload one video via resumable upload; returns the video id."""
        if not file_path or not os.path.isfile(file_path):
            raise UploadError(f"Video file not found: {file_path!r}")
        privacy = privacy or self.privacy
        if privacy not in _VALID_PRIVACY:
            raise UploadError(
                f"Invalid privacy '{privacy}'; expected one of {sorted(_VALID_PRIVACY)}"
            )

        youtube = self._client()
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": _CATEGORY_ENTERTAINMENT,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False,
            },
        }
        media = MediaFileUpload(file_path, chunksize=_CHUNK_SIZE, resumable=True)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

        logger.info(f"Starting upload: {title} ({os.path.basename(file_path)})")
        retries = 0
        response = None
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    logger.info(
                        f"Upload progress {int(status.progress() * 100)}%: {title}"
                    )
            except HttpError as exc:
                if (
                    exc.resp.status in _RETRYABLE_HTTP_CODES
                    and retries < _MAX_RETRIES
                ):
                    retries += 1
                    sleep_s = 2 ** retries
                    logger.warning(
                        f"Retryable HttpError {exc.resp.status} during upload; "
                        f"retry {retries}/{_MAX_RETRIES} in {sleep_s}s"
                    )
                    time.sleep(sleep_s)
                    continue
                raise UploadError(
                    f"YouTube upload failed with HTTP {exc.resp.status}"
                ) from exc
            except IOError as exc:
                if retries < _MAX_RETRIES:
                    retries += 1
                    sleep_s = 2 ** retries
                    logger.warning(
                        f"IOError during upload; retry {retries}/{_MAX_RETRIES} "
                        f"in {sleep_s}s: {exc}"
                    )
                    time.sleep(sleep_s)
                    continue
                raise UploadError(f"YouTube upload failed after retries: {exc}") from exc

        video_id = (response or {}).get("id")
        if not video_id:
            raise UploadError("YouTube upload finished without returning a video id")
        logger.info(f"Upload complete: {title} -> video id {video_id}")
        return video_id

    def create_playlist(
        self,
        title: str,
        description: str,
        privacy: Optional[str] = None,
    ) -> str:
        """Create a playlist; returns the playlist id."""
        privacy = privacy or self.privacy
        if privacy not in _VALID_PRIVACY:
            raise UploadError(
                f"Invalid privacy '{privacy}'; expected one of {sorted(_VALID_PRIVACY)}"
            )
        youtube = self._client()
        try:
            response = (
                youtube.playlists()
                .insert(
                    part="snippet,status",
                    body={
                        "snippet": {"title": title, "description": description},
                        "status": {"privacyStatus": privacy},
                    },
                )
                .execute()
            )
        except HttpError as exc:
            raise UploadError(
                f"Playlist creation failed with HTTP {exc.resp.status}"
            ) from exc
        playlist_id = (response or {}).get("id")
        if not playlist_id:
            raise UploadError("Playlist creation returned no id")
        logger.info(f"Created playlist '{title}' ({playlist_id})")
        return playlist_id

    def add_video_to_playlist(self, playlist_id: str, video_id: str) -> None:
        """Insert an uploaded video into a playlist."""
        if not playlist_id or not video_id:
            raise UploadError("playlist_id and video_id are required")
        youtube = self._client()
        try:
            youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {"kind": "youtube#video", "videoId": video_id},
                    }
                },
            ).execute()
        except HttpError as exc:
            raise UploadError(
                f"Adding video to playlist failed with HTTP {exc.resp.status}"
            ) from exc
        logger.info(f"Added video {video_id} to playlist {playlist_id}")

    # ------------------------------------------------------------------
    # High-level flow
    # ------------------------------------------------------------------

    def upload_movie_parts(
        self,
        part_files: list[tuple[int, str]],
        movie_title: str,
        year: Optional[int],
        description: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PlaylistResult:
        """Upload all parts and collect them in a playlist.

        ``part_files`` is ``[(part_number, file_path), ...]`` in order.
        Each video is inserted into the playlist as its upload completes.
        """
        if not part_files:
            raise UploadError("part_files must not be empty")
        for part_number, file_path in part_files:
            if not os.path.isfile(file_path):
                raise UploadError(
                    f"Part {part_number} file not found: {file_path!r}"
                )

        total = len(part_files)
        year_suffix = f" ({year})" if year else ""
        playlist_title = f"{movie_title}{year_suffix}"

        part_lines = "\n".join(
            f"Part {n}/{total}" for n, _ in part_files
        )
        full_description = (
            f"{description}\n\n{part_lines}\n\n{TMDB_ATTRIBUTION}"
        )

        playlist_id = self.create_playlist(
            playlist_title, full_description, privacy=None
        )
        result = PlaylistResult(
            playlist_id=playlist_id,
            playlist_url=f"https://www.youtube.com/playlist?list={playlist_id}",
            parts=[],
        )

        for part_number, file_path in part_files:
            title = f"{movie_title}{year_suffix} — Part {part_number}/{total}"
            video_id = self.upload_video(
                file_path=file_path,
                title=title,
                description=full_description,
                privacy=None,
            )
            self.add_video_to_playlist(playlist_id, video_id)
            result.parts.append(
                UploadedPart(
                    part_number=part_number,
                    video_id=video_id,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                )
            )
            if progress_callback:
                progress_callback(len(result.parts), total)

        logger.info(
            f"All {total} part(s) uploaded for '{playlist_title}': {result.playlist_url}"
        )
        return result
