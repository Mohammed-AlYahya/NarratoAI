# README-RECAP — Fully Automatic Movie Recap Pipeline

`python -m recap "Upgrade"` turns a movie into a YouTube-ready vertical recap series:
**acquire** the film (TorBox + Prowlarr + TMDB) → **generate** recap narration with
NarratoAI's LLM service layer → **render** 1–4 vertical (1080×1920, 9:16, letterbox —
never center-cropped) parts with narration, cut footage, burned subtitles and BGM →
**upload** all parts to a YouTube playlist.

Job state is persisted to `storage/recap_jobs/<movie-slug>.json` after every stage, so
any interrupted run resumes exactly where it stopped.

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. NarratoAI base config (still required)

The pipeline reuses NarratoAI's LLM and TTS configuration from `config.toml`
(created from `config.example.toml` on first import). You must set the text-LLM keys:

```toml
[app]
text_llm_provider = "openai"
text_openai_api_key = "..."
text_openai_model_name = "..."
text_openai_base_url = "https://..."   # optional
```

TTS defaults to `edge_tts`, which needs **zero config** (voice defaults to
`ui.edge_voice_name`, e.g. `zh-CN-XiaoyiNeural-Female`; override with `TTS_VOICE_NAME`).

### 3. Recap `.env`

```bash
cp .env.example .env   # then fill in the keys
```

| Key | Purpose |
| --- | --- |
| `TORBOX_API_KEY` | TorBox account → Settings → API |
| `TMDB_API_KEY` **or** `TMDB_READ_ACCESS_TOKEN` | TMDB account → Settings → API. Only one is needed: the v4 **Read Access Token** (preferred) or the v3 **API Key** |
| `PROWLARR_URL` / `PROWLARR_API_KEY` | Your **self-hosted** Prowlarr instance (install from prowlarr.com; default `http://localhost:9696`). API key under Settings → General |
| `YOUTUBE_CLIENT_SECRETS` | Path to the Google OAuth `client_secret.json` |
| `PRIVACY` | `unlisted` (default), `private` or `public` |
| `MAX_PARTS` / `MAX_PART_SECONDS` | Recap budget, default `4` × `180`s |
| `PREFER_QUALITY` | Release preference, default `1080p-bluray` |

### 4. YouTube Data API v3 (one-time)

1. Google Cloud Console → create a project → enable **YouTube Data API v3**.
2. Create OAuth credentials (Desktop app) and download `client_secret.json`.
3. **Important:** on the OAuth consent screen, switch the app from **"Testing" to
   "In production"**. In Testing mode refresh tokens expire every 7 days and the
   pipeline will keep asking you to re-authenticate.
4. First run opens a browser once for consent; afterwards the token at
   `storage/youtube_token.json` is reused headlessly.

**Quota note:** one video insert costs 1,600 quota units and the default quota is
10,000 units/day — roughly **one 4-part movie per day** on a fresh project.

## Usage

```bash
# Real run
python -m recap "Upgrade" --year 2018

# Acceptance run (no TorBox/Prowlarr, private upload)
python -m recap "Test Movie" --mock-acquire --privacy private

# Render only, no upload
python -m recap "Upgrade" --year 2018 --no-upload

# Provide your own subtitles
python -m recap "Upgrade" --subtitle path/to/movie.srt

# Debug a single stage using persisted state
python -m recap "Upgrade" --stage render
```

## How subtitles are sourced

The LLM needs the movie's dialogue as text. The pipeline tries, in order:

1. `--subtitle <file.srt>` CLI argument (highest priority).
2. Embedded subtitle stream extraction via ffmpeg. The first several subtitle
   streams are probed and the first **text-based** one (SRT/ASS/VTT) is used —
   BluRay **PGS streams are bitmap images and cannot be converted to text**, so
   PGS-only releases fall through to option 3/4. SDH captions with sound-effect
   markers (`[GLASS SHATTERING]`) are fine and give the LLM useful action context.
   (Uses `config.app.ffmpeg_path` when configured, otherwise `ffmpeg` on PATH;
   skipped with a warning when no ffmpeg is available.)
3. A configured NarratoAI **fun_asr** backend in `config.toml`
   (`[fun_asr] auto_transcribe_enabled = true` with `backend = "local" | "firered" | "bailian"`).
4. Otherwise the run stops with a clear error asking for `--subtitle` or fun_asr.

## The recap format

- 1080×1920 (9:16) vertical; widescreen footage is **letterboxed** (pillar/bar padding,
  never center-cropped).
- At most `MAX_PARTS` parts (default 4), each at most `MAX_PART_SECONDS` (default 180s).
- The story is **guaranteed to conclude within the part budget**: segments are measured
  by real TTS audio duration (never by word count), packed greedily into parts, and when
  the story overflows, the narration copy is regenerated shorter (compress-and-regenerate,
  up to 3 attempts) — a dangling ending is never emitted.
- Narration loud, BGM ≈ 0.25, original audio ≈ 0.1 to approximate ducking.
- `ORIGINAL_SOUND_RATIO` (default `0`) controls how many recap segments play with
  the movie's **original audio** instead of voiceover — set it to `20`–`30` to let
  key scenes run with original dialogue/sound between narration blocks.

## TMDB attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.
