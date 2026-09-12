"""CLI entry point: ``python -m recap "Upgrade" --year 2018``.

Heavy pipeline imports happen inside ``main()`` (after argparse) so that
``--help`` works even when the acquisition/uploader modules or NarratoAI's
runtime config are not available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOCK_VIDEO = REPO_ROOT / "tests" / "assets" / "cc_test_video.mp4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recap",
        description=(
            "Fully automatic movie recap pipeline: acquire a movie, generate "
            "recap narration with NarratoAI, render 1-4 vertical parts and "
            "upload them to a YouTube playlist."
        ),
    )
    parser.add_argument("movie_name", help='Movie title, e.g. "Upgrade"')
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Release year hint for disambiguation (e.g. 2018)",
    )
    parser.add_argument(
        "--privacy",
        choices=["private", "unlisted", "public"],
        default=None,
        help="YouTube privacy status (default: PRIVACY env or 'unlisted'; "
        "never public unless explicitly requested)",
    )
    parser.add_argument(
        "--mock-acquire",
        action="store_true",
        help="Skip TorBox/Prowlarr and use a local mock video",
    )
    parser.add_argument(
        "--mock-video",
        default=str(DEFAULT_MOCK_VIDEO),
        help="Mock video path for --mock-acquire "
        "(default: tests/assets/cc_test_video.mp4)",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Use this .srt instead of embedded-subtitle extraction / fun_asr",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Render parts but skip the YouTube upload stage",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Narration language override (default: NARRATION_LANGUAGE env or English)",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="Override MAX_PARTS (default 4)",
    )
    parser.add_argument(
        "--max-part-seconds",
        type=int,
        default=None,
        help="Override MAX_PART_SECONDS (default 180)",
    )
    parser.add_argument(
        "--stage",
        choices=["acquire", "subtitles", "script", "split", "render", "upload"],
        default=None,
        help="Debug: run only this single stage using persisted state",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Imported lazily so `--help` works without heavy/deferred dependencies.
    from loguru import logger

    from .pipeline import run

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    try:
        state = run(
            args.movie_name,
            year=args.year,
            privacy=args.privacy,
            mock_acquire=args.mock_acquire,
            mock_video=args.mock_video if args.mock_acquire else None,
            subtitle=args.subtitle,
            no_upload=args.no_upload,
            language=args.language,
            max_parts=args.max_parts,
            max_part_seconds=args.max_part_seconds,
            stage=args.stage,
        )
    except Exception as exc:
        logger.error(str(exc))
        return 1

    playlist_url = state.get("playlist_url")
    if playlist_url:
        print(f"\nPlaylist: {playlist_url}")
    elif args.no_upload or args.stage:
        rendered = [p for p in state.get("parts", []) if p.get("video_path")]
        print(f"\nDone (no upload). Rendered parts: {len(rendered)}")
        for part in rendered:
            print(f"  Part {part['n']}: {part['video_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
