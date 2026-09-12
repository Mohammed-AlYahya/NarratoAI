"""recap — fully automatic movie recap pipeline for NarratoAI.

Usage::

    python -m recap "Upgrade" --year 2018

Acquires a movie, generates recap narration via NarratoAI's service layer,
renders 1-4 vertical (1080x1920) recap parts and uploads them to a YouTube
playlist. See README-RECAP.md for setup instructions.

Note on imports: this package deliberately keeps its top-level imports light.
Heavy NarratoAI modules (``app.config``, ``app.services.*``) and the parallel
acquisition/uploader modules are imported lazily inside functions so that
``python -m recap --help`` and the unit tests work even without API
credentials, config.toml, or the acquisition/uploader modules on disk.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
