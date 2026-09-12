"""Configuration loading for the recap pipeline.

Reads ``.env`` at the repository root (via python-dotenv) and exposes a
:class:`RecapSettings` dataclass. The module is intentionally free of heavy
imports so it can be used from tests and from ``python -m recap --help``
without NarratoAI's runtime config being importable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

VALID_PRIVACY = ("private", "unlisted", "public")

#: Defaults (also documented in .env.example)
DEFAULT_PRIVACY = "unlisted"
DEFAULT_MAX_PARTS = 4
DEFAULT_MAX_PART_SECONDS = 180
DEFAULT_PREFER_QUALITY = "1080p-bluray"
DEFAULT_TOP_N_CANDIDATES = 10
DEFAULT_NARRATION_LANGUAGE = "English"
DEFAULT_TTS_ENGINE = "edge_tts"
DEFAULT_EDGE_VOICE_FALLBACK = "zh-CN-XiaoyiNeural-Female"


class SettingsError(ValueError):
    """Raised when recap settings are invalid."""


def _get(env: Mapping[str, str], key: str, default: str = "") -> str:
    value = env.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _get(env, key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SettingsError(f"{key} must be an integer, got {raw!r}") from exc


@dataclass
class RecapSettings:
    """Settings for the recap pipeline, sourced from environment variables."""

    torbox_api_key: Optional[str] = None
    tmdb_api_key: Optional[str] = None
    prowlarr_url: Optional[str] = None
    prowlarr_api_key: Optional[str] = None
    youtube_client_secrets: Optional[str] = None
    youtube_token_path: str = ""
    privacy: str = DEFAULT_PRIVACY
    max_parts: int = DEFAULT_MAX_PARTS
    max_part_seconds: int = DEFAULT_MAX_PART_SECONDS
    prefer_quality: str = DEFAULT_PREFER_QUALITY
    top_n_candidates: int = DEFAULT_TOP_N_CANDIDATES
    narration_language: str = DEFAULT_NARRATION_LANGUAGE
    tts_engine: str = DEFAULT_TTS_ENGINE
    tts_voice_name: str = ""  # resolved lazily from config.ui when empty

    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.privacy not in VALID_PRIVACY:
            raise SettingsError(
                f"PRIVACY must be one of {VALID_PRIVACY}, got {self.privacy!r}"
            )
        if self.max_parts < 1:
            raise SettingsError(f"MAX_PARTS must be >= 1, got {self.max_parts}")
        if self.max_part_seconds < 10:
            raise SettingsError(
                f"MAX_PART_SECONDS must be >= 10, got {self.max_part_seconds}"
            )
        if self.top_n_candidates < 1:
            raise SettingsError(
                f"TOP_N_CANDIDATES must be >= 1, got {self.top_n_candidates}"
            )
        if not self.youtube_token_path:
            self.youtube_token_path = str(REPO_ROOT / "storage" / "youtube_token.json")
        # Normalise empty strings to None for optional secrets.
        for name in (
            "torbox_api_key",
            "tmdb_api_key",
            "prowlarr_url",
            "prowlarr_api_key",
            "youtube_client_secrets",
        ):
            if getattr(self, name) == "":
                setattr(self, name, None)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "RecapSettings":
        """Build settings from an arbitrary env mapping (test-friendly)."""
        return cls(
            torbox_api_key=_get(env, "TORBOX_API_KEY"),
            tmdb_api_key=_get(env, "TMDB_API_KEY"),
            prowlarr_url=_get(env, "PROWLARR_URL"),
            prowlarr_api_key=_get(env, "PROWLARR_API_KEY"),
            youtube_client_secrets=_get(env, "YOUTUBE_CLIENT_SECRETS"),
            youtube_token_path=_get(env, "YOUTUBE_TOKEN_PATH"),
            privacy=_get(env, "PRIVACY", DEFAULT_PRIVACY) or DEFAULT_PRIVACY,
            max_parts=_get_int(env, "MAX_PARTS", DEFAULT_MAX_PARTS),
            max_part_seconds=_get_int(env, "MAX_PART_SECONDS", DEFAULT_MAX_PART_SECONDS),
            prefer_quality=_get(env, "PREFER_QUALITY", DEFAULT_PREFER_QUALITY)
            or DEFAULT_PREFER_QUALITY,
            top_n_candidates=_get_int(env, "TOP_N_CANDIDATES", DEFAULT_TOP_N_CANDIDATES),
            narration_language=_get(env, "NARRATION_LANGUAGE", DEFAULT_NARRATION_LANGUAGE)
            or DEFAULT_NARRATION_LANGUAGE,
            tts_engine=_get(env, "TTS_ENGINE", DEFAULT_TTS_ENGINE) or DEFAULT_TTS_ENGINE,
            tts_voice_name=_get(env, "TTS_VOICE_NAME"),
        )

    def resolve_tts_voice_name(self) -> str:
        """Resolve the TTS voice, falling back to NarratoAI's UI config.

        Lazy-imports ``app.config`` so that importing this module never
        requires the NarratoAI runtime config to exist.
        """
        if self.tts_voice_name:
            return self.tts_voice_name
        try:
            from app.config import config as app_config

            voice = app_config.ui.get("edge_voice_name", "") or ""
        except Exception:
            voice = ""
        return voice or DEFAULT_EDGE_VOICE_FALLBACK


def load_settings(env_file: Optional[Path] = None) -> RecapSettings:
    """Load ``.env`` from the repo root and build :class:`RecapSettings`.

    Existing environment variables take precedence over ``.env`` values
    (python-dotenv default: ``override=False``).
    """
    from dotenv import load_dotenv

    dotenv_path = Path(env_file) if env_file else ENV_FILE
    if dotenv_path.is_file():
        load_dotenv(dotenv_path, override=False)
    return RecapSettings.from_env(os.environ)
