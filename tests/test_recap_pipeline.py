"""Unit tests for the recap pipeline helpers.

No network, no credentials, no NarratoAI runtime imports: ``recap.config``
and the pure helpers in ``recap.pipeline`` (slugify, split_into_parts,
state persistence) are import-safe by design.
"""

from __future__ import annotations

import json

import pytest

from recap.config import (
    DEFAULT_MAX_PART_SECONDS,
    DEFAULT_MAX_PARTS,
    RecapSettings,
    SettingsError,
)
from recap.pipeline import (
    load_state,
    new_state,
    save_state,
    slugify,
    split_into_parts,
    state_path,
)


def _items(durations):
    return [
        {"_id": i + 1, "timestamp": "", "picture": "", "narration": f"line {i + 1}",
         "OST": 0, "duration": float(d)}
        for i, d in enumerate(durations)
    ]


class TestSplitIntoParts:
    def test_fits_in_one_part(self):
        items = _items([60, 60, 60])
        parts = split_into_parts(items, max_parts=4, max_seconds=180)
        assert parts is not None
        assert len(parts) == 1
        assert [i["_id"] for i in parts[0]] == [1, 2, 3]

    def test_exact_boundary(self):
        items = _items([90, 90, 90, 90])
        parts = split_into_parts(items, max_parts=2, max_seconds=180)
        assert parts is not None
        assert [[i["_id"] for i in p] for p in parts] == [[1, 2], [3, 4]]

    def test_greedy_multi_part(self):
        # 100+100 > 180, so each part holds exactly one segment.
        items = _items([100, 100, 100])
        parts = split_into_parts(items, max_parts=4, max_seconds=180)
        assert parts is not None
        assert [[i["_id"] for i in p] for p in parts] == [[1], [2], [3]]

    def test_greedy_packs_consecutive(self):
        # 80+80=160 fits; +50 would overflow -> close part.
        # 50+170 overflows -> 50 alone. 170+10=180 fits exactly.
        items = _items([80, 80, 50, 170, 10])
        parts = split_into_parts(items, max_parts=4, max_seconds=180)
        assert parts is not None
        assert [[i["_id"] for i in p] for p in parts] == [[1, 2], [3], [4, 5]]

    def test_returns_none_when_exceeding_max_parts(self):
        items = _items([200, 200, 200, 200, 200])
        assert split_into_parts(items, max_parts=4, max_seconds=180) is None

    def test_single_item_longer_than_part_returns_none(self):
        items = _items([300])
        assert split_into_parts(items, max_parts=4, max_seconds=180) is None

    def test_empty_returns_none(self):
        assert split_into_parts([], max_parts=4, max_seconds=180) is None

    def test_missing_duration_treated_as_zero(self):
        items = [{"_id": 1, "narration": "x"}, {"_id": 2, "narration": "y"}]
        parts = split_into_parts(items, max_parts=2, max_seconds=10)
        assert parts == [items]


class TestSlugify:
    def test_simple(self):
        assert slugify("Upgrade") == "upgrade"

    def test_spaces_and_punctuation(self):
        assert slugify("The Matrix: Reloaded") == "the-matrix-reloaded"

    def test_non_ascii_fallback(self):
        assert slugify("升级 2.0") == "2-0"

    def test_empty_becomes_movie(self):
        assert slugify("!!!") == "movie"


class TestRecapSettings:
    def test_defaults(self):
        settings = RecapSettings.from_env({})
        assert settings.privacy == "unlisted"
        assert settings.max_parts == DEFAULT_MAX_PARTS == 4
        assert settings.max_part_seconds == DEFAULT_MAX_PART_SECONDS == 180
        assert settings.prefer_quality == "1080p-bluray"
        assert settings.top_n_candidates == 10
        assert settings.narration_language == "English"
        assert settings.tts_engine == "edge_tts"
        assert settings.torbox_api_key is None
        assert settings.youtube_token_path.endswith("youtube_token.json")

    def test_parse_from_env_dict(self):
        env = {
            "TORBOX_API_KEY": "tb",
            "TMDB_API_KEY": "tm",
            "PROWLARR_URL": "http://localhost:9696",
            "PROWLARR_API_KEY": "pw",
            "YOUTUBE_CLIENT_SECRETS": "client_secret.json",
            "PRIVACY": "private",
            "MAX_PARTS": "3",
            "MAX_PART_SECONDS": "120",
            "NARRATION_LANGUAGE": "简体中文（中国）",
            "TTS_VOICE_NAME": "en-US-AvaNeural-Female",
        }
        settings = RecapSettings.from_env(env)
        assert settings.torbox_api_key == "tb"
        assert settings.tmdb_api_key == "tm"
        assert settings.prowlarr_url == "http://localhost:9696"
        assert settings.prowlarr_api_key == "pw"
        assert settings.youtube_client_secrets == "client_secret.json"
        assert settings.privacy == "private"
        assert settings.max_parts == 3
        assert settings.max_part_seconds == 120
        assert settings.narration_language == "简体中文（中国）"
        assert settings.tts_voice_name == "en-US-AvaNeural-Female"

    def test_invalid_privacy_rejected(self):
        with pytest.raises(SettingsError):
            RecapSettings.from_env({"PRIVACY": "friends-only"})

    def test_invalid_int_rejected(self):
        with pytest.raises(SettingsError):
            RecapSettings.from_env({"MAX_PARTS": "four"})

    def test_blank_optional_becomes_none(self):
        settings = RecapSettings.from_env({"TORBOX_API_KEY": "  "})
        assert settings.torbox_api_key is None


class TestStatePersistence:
    def test_round_trip(self, tmp_path):
        state = new_state("Test Movie")
        state["stage"] = "render"
        state["acquired"] = {
            "file_path": "C:/tmp/movie.mp4",
            "tmdb_id": 123,
            "imdb_id": "tt0123",
            "metadata": {"title": "Test Movie"},
            "release_title": "Test.Movie.1080p",
        }
        state["parts"] = [
            {"n": 1, "script_path": "a.json", "task_id": "recap-test-movie-p1",
             "video_path": "p1.mp4", "video_id": None, "status": "rendered"}
        ]
        path = save_state(state, tmp_path)
        assert path == state_path("test-movie", tmp_path)
        assert path.is_file()

        loaded = load_state("test-movie", tmp_path)
        assert loaded == state
        # JSON on disk is valid and complete.
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["movie_name"] == "Test Movie"
        assert on_disk["parts"][0]["status"] == "rendered"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_state("does-not-exist", tmp_path) is None

    def test_resume_after_error(self, tmp_path):
        state = new_state("Test Movie")
        state["error"] = "RuntimeError: boom"
        save_state(state, tmp_path)
        loaded = load_state("test-movie", tmp_path)
        assert loaded["error"] == "RuntimeError: boom"
        loaded["error"] = None
        loaded["stage"] = "done"
        save_state(loaded, tmp_path)
        assert load_state("test-movie", tmp_path)["stage"] == "done"


class TestApplyUploadedParts:
    """Lock the contract with app.services.uploader's UploadedPart shape."""

    def test_dataclass_records(self):
        from dataclasses import dataclass

        from recap.pipeline import _apply_uploaded_parts

        @dataclass
        class UploadedPart:
            part_number: int
            video_id: str
            url: str = ""

        state = new_state("Test Movie")
        state["parts"] = [
            {"n": 1, "status": "rendered", "video_id": None},
            {"n": 2, "status": "rendered", "video_id": None},
        ]
        _apply_uploaded_parts(
            state, [UploadedPart(part_number=1, video_id="aaa"), UploadedPart(2, "bbb")]
        )
        assert state["parts"][0]["video_id"] == "aaa"
        assert state["parts"][0]["status"] == "uploaded"
        assert state["parts"][1]["video_id"] == "bbb"

    def test_dict_and_tuple_records(self):
        from recap.pipeline import _apply_uploaded_parts

        state = new_state("Test Movie")
        state["parts"] = [
            {"n": 1, "status": "rendered", "video_id": None},
            {"n": 2, "status": "rendered", "video_id": None},
        ]
        _apply_uploaded_parts(state, [{"part_number": 1, "video_id": "x"}, (2, "y")])
        assert [p["video_id"] for p in state["parts"]] == ["x", "y"]

    def test_none_is_noop(self):
        from recap.pipeline import _apply_uploaded_parts

        state = new_state("Test Movie")
        state["parts"] = [{"n": 1, "status": "rendered", "video_id": None}]
        _apply_uploaded_parts(state, None)
        assert state["parts"][0]["video_id"] is None
