# MachGen vs Reactor: generative video inference benchmark

A small, reproducible benchmark of two generative-video inference providers, run 12–15 October 2026:

- **MachGen** returns finished files through a request/poll API, on kernel-optimised open models.
- **Reactor** runs models as live, steerable WebRTC streams.

It measures speed (queue, GPU and delivery time per request), real-time behaviour (time to first frame,
frame rate, freezes, control-to-screen latency), cost, and a visual quality check on matched prompts.
The results page is `report/benchmark.html`. Open it in a browser; it needs no server.

> **Read before quoting.** Samples are small (6 clips per variant, 12–15 images, one session per live
> model), every run came from one laptop, and prices and capacity change. Treat the numbers as a
> snapshot, not a ranking. See [Limits](#limits).

## Headline results

| MiniMax H3 Turbo, 5.17 s clip, 1344×768, same prompts and seeds | MachGen H3 Turbo | Reactor H3 Reference Turbo Realtime |
|---|---|---|
| Generation per clip | 2.76–2.89 s (reported by MachGen) | 2.05–2.38 s (measured client-side) |
| First clip watchable | 8.1 s | 2.4 s |
| Slowest clip | 73.2 s (67 s in queue) | 2.4 s |
| All 6 clips ready | 130.7 s | 13.0 s |
| Cost per clip | $0.05 promo, $0.10 list | $0.07–0.20 (session-time billing) |

**Step-matched run (15 Oct, evening).** MachGen's default H3 Turbo adapter is undisclosed, so it was
pinned to LightX2V's 8-step adapter to match Reactor's H3 Reference Turbo, which most likely runs
LightX2V's 8-step reference-to-video distillation (unconfirmed by Reactor):

| Per clip | MachGen H3 Turbo, 8-step | Reactor H3 Reference Turbo |
|---|---|---|
| Generation | 3.32–3.94 s, median 3.79 s (reported) | 2.04–2.22 s, median 2.12 s (client-side) |
| Seconds per step | 0.47 s | 0.27 s if 8 steps, 0.53 s if 4 steps |
| Queue waits | 10 s, 11 s, 74 s | none |

At 8 steps MachGen's GPU time rose 37% over its default (2.77 s), consistent with the default running
about 6 steps. Whether Reactor's engine is faster per step depends on its undisclosed step count.

Other findings, with charts, are in the report: MachGen queue behaviour under load, promo vs list
pricing, an earlier FastH3 comparison, and Reactor's live models (LingBot World 2, Helios, SANA-Streaming).

## Setup

Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Or with plain pip: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

Copy `.env.example` to `.env` and fill in your own keys. `.env` is git-ignored; never commit it.

```bash
cp .env.example .env
.venv/bin/python check_keys.py   # free calls only: MachGen balance, Reactor token mint, fal model search
```

## Running the tests

Every command writes raw results to `results/raw/` and media to `outputs/`. Costs are approximate list
prices at the time of the run.

### Request/response speed (MachGen, optionally fal)

`bench/run.py` runs matched suites from `bench/suites.py` (same prompts, seeds and output settings per
provider). It checks the planned spend against `--budget` before submitting anything.

```bash
# MiniMax H3 Turbo, 6 clips, one at a time (~$0.30 at MachGen's promo price)
.venv/bin/python -m bench.run --suite minimax-h3-turbo-768p-5s --n 6 --providers machgen --concurrency 1 --budget 1

# Same, pinned to LightX2V's 8-step adapter for a step-matched comparison
.venv/bin/python -m bench.run --suite minimax-h3-turbo-8step-768p-5s --n 6 --providers machgen --concurrency 1 --budget 1

# FLUX.2-dev images, 15 sequential then 12 with 4 in flight (~$0.10)
.venv/bin/python -m bench.run --suite flux2-dev --n 15 --providers machgen --concurrency 1 --budget 1
.venv/bin/python -m bench.run --suite flux2-dev --n 12 --providers machgen --concurrency 4 --budget 1

# Add a dry run to see the plan and cost without submitting
.venv/bin/python -m bench.run --suite minimax-h3-768p-5s --n 6 --providers machgen --dry-run
```

Suites: `flux2-dev`, `minimax-h3-768p-5s`, `minimax-h3-turbo-768p-5s`, `ltx23-pro-1080p-6s`,
`wan22-a14b-720p-5s`. `--providers machgen,fal` also runs fal on the same suite. fal was part of the
first round and is kept as an optional provider; the published report covers MachGen and Reactor only.

### Reactor clip models (FastH3, H3 Reference Turbo Realtime)

`bench/fasth3.py` enqueues the same prompts and seeds in one session with autoplay on, and times
enqueue → built → first frame → played out. The session token caps the session length, so a crash
cannot keep a GPU billing.

```bash
# H3 Reference Turbo Realtime, 6 clips (~$1.20 at $2.10/min of session)
.venv/bin/python -m bench.fasth3 --model reactor/h3-reference-to-video-turbo-realtime --clips 6 --max-session 75

# FastH3 (a FastVideo distilled H3), 6 clips (~$1.30)
.venv/bin/python -m bench.fasth3 --model reactor/fast-h3 --clips 6
```

### Reactor live models

`bench/realtime_reactor.py` measures session start, first frame, frame rate, frame-gap freezes and
control-to-screen latency.

```bash
.venv/bin/python -m bench.realtime_reactor --scenario helios  --seconds 60   # prompt change mid-stream (~$0.10)
.venv/bin/python -m bench.realtime_reactor --scenario lingbot --seconds 60   # "look right" detected by motion (~$0.45)
.venv/bin/python -m bench.realtime_reactor --scenario sana    --seconds 45   # marker square sent through the model (~$0.08)
```

### Experimental, not yet run

`bench/h3turbo_r2v.py` runs MiniMax H3 Turbo reference-to-video on both providers with the same
reference images. It has not been run, so treat it as untested.

`bench/realtime_fal.py` measures fal's real-time WebSocket image endpoint (fast-lcm-diffusion).

## What is measured, and how

| Metric | MachGen | Reactor |
|---|---|---|
| Queue time | `queue_time_secs` from the task status | none observed (a GPU is held per session) |
| Generation time | `generation_time_secs`, reported by MachGen | gap between consecutive `clip_generated` events, measured client-side |
| Time until watchable | submit → file fully downloaded | enqueue → `clip_started` |
| Cost | per-task billing API | session seconds in `ready` × published rate |

MachGen's task status also exposes `execution_provider` and `overflow_converted`. The harness logs both
on every job; no job in these runs was moved to partner GPUs.

The same seed produces different images on each provider, so quality is compared by eye on matched
prompts (`report/*side_by_side.jpg`), not scored.

## Repository layout

```
bench/            benchmark code (providers, suites, runners)
check_keys.py     verifies API keys with free calls
results/raw/      raw per-job and per-session results (JSONL / JSON / CSV)
report/           results page (benchmark.html), its data files, side-by-side frame sheets
assets/           reference image used by the LingBot World 2 test
```

Generated videos and images (`outputs/`) are not included because of their size. Re-running a test
recreates them.

## Limits

- Small samples and one client location. The network round trip to Reactor was 119–187 ms.
- Runs on 12 and 15 October 2026 only. MachGen says its GPUs are overloaded at peak times, which
  matches the queue waits measured here; results at other times of day may differ.
- Neither provider publishes the denoising step count of its Turbo variant. FastH3 is a FastVideo
  distilled checkpoint (4-step V1 or 8-step V2; Reactor does not say which).
- MachGen's generation time is self-reported; Reactor's is measured from the client and includes
  network and pipeline overhead.
- MachGen's sync endpoint was not available to this account (it requires moderation off), so MachGen
  was measured on its queue API.
- Prices: MachGen H3 Turbo was on promotion ($0.01/s vs $0.02/s list). Reactor bills per second of
  session time, so its per-clip cost depends on how busy the session is kept.
