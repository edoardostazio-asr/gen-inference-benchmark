"""Measure Reactor real-time sessions: startup, first frame, sustained FPS, stalls, and
control-to-screen latency.

    python -m bench.realtime_reactor --scenario helios   --seconds 60
    python -m bench.realtime_reactor --scenario lingbot  --seconds 60
    python -m bench.realtime_reactor --scenario sana     --seconds 60

Control-to-screen latency, per scenario:
  helios   prompt switch mid-stream -> first chunk generated with the new prompt
  lingbot  `look right` command      -> first frames whose content visibly pans (phase correlation)
  sana     marker square in the frames we send flips black/white -> same flip seen in the output
           (glass-to-glass round trip through the model)

The session token is minted with max_session_duration_seconds = --seconds + 60, so a crashed script
cannot leave a GPU billing. Billing only runs while the session is `ready`.
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import time
from datetime import datetime, timezone

import httpx
import numpy as np
from dotenv import load_dotenv
from reactor_sdk import Reactor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

MODELS = {"helios": "reactor/helios", "lingbot": "reactor/lingbot-world-2", "sana": "reactor/sana-streaming"}
PROMPT_A = "A slow drone flight over a misty pine forest at sunrise, soft volumetric light"
PROMPT_B = "A slow drone flight over a neon-lit cyberpunk city at night, rain and reflections"
LINGBOT_PROMPT = "A red fox standing in a snowy birch forest at dawn, soft golden light"
LINGBOT_IMAGE = os.path.join(ROOT, "assets", "lingbot_ref.jpg")
SANA_PROMPT = "Add a subtle warm film grain to the whole scene"
STALL_GAP_S = 0.25          # an inter-frame gap above this is visible as a freeze
SANA_W, SANA_H, SANA_FPS = 1280, 704, 24
MARKER = (slice(40, 240), slice(40, 240))  # marker square (rows, cols) in input and output
MARKER_PERIOD_S = 4.0


def mint_jwt(model, max_seconds):
    r = httpx.post(
        "https://api.reactor.inc/tokens",
        headers={"Reactor-API-Key": os.environ["REACTOR_API_KEY"]},
        json={"expires_after": 3600, "authorization_details": [{
            "type": "session", "resources": {"models": {"match": [model]}},
            "constraints": {"max_sessions": 1, "max_session_duration_seconds": max_seconds}}]},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["jwt"]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def pan_shift(a, b):
    """Horizontal shift in pixels between two small grayscale frames (phase correlation)."""
    f = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    r = np.abs(np.fft.ifft2(f / (np.abs(f) + 1e-9)))
    _, dx = np.unravel_index(np.argmax(r), r.shape)
    return dx - a.shape[1] if dx > a.shape[1] // 2 else dx


def sana_frame(white, base):
    frame = base.copy()
    frame[MARKER] = 255 if white else 0
    return frame


async def main(args):
    scenario, model = args.scenario, MODELS[args.scenario]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reactor = Reactor(model_name=model, jwt=mint_jwt(model, args.seconds + 60))

    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0  # noqa: E731
    status_log, messages, frames, stats = [], [], [], []
    small = []               # (t, downscaled grayscale) for lingbot pan detection
    marker_out = []          # (t, mean luminance of marker region) for sana
    marker_in = []           # (t, white?) transitions we sent, for sana
    ready = asyncio.Event()
    marks = {"t_start_sent": None, "t_generation_started": None, "t_first_frame": None,
             "t_ctrl_sent": None, "t_switch_accepted": None, "t_switch_visible": None}

    def on_status(s):
        status_log.append((round(now(), 3), str(s)))
        if str(s) == "ready":
            ready.set()

    def on_message(m):
        t = now()
        kind = m.get("type") if isinstance(m, dict) else None
        messages.append((round(t, 3), kind, m if kind != "state" else None))
        data = (m.get("data") or {}) if isinstance(m, dict) else {}
        if kind == "generation_started" and marks["t_generation_started"] is None:
            marks["t_generation_started"] = t
        if scenario == "helios" and marks["t_ctrl_sent"]:
            if kind == "prompt_accepted" and data.get("prompt") == PROMPT_B:
                marks["t_switch_accepted"] = marks["t_switch_accepted"] or t
            if kind == "chunk_complete" and data.get("active_prompt") == PROMPT_B:
                marks["t_switch_visible"] = marks["t_switch_visible"] or t

    reactor.on("status_changed", on_status)
    reactor.on("message", on_message)
    publisher = None

    try:
        await reactor.connect()
        await asyncio.wait_for(ready.wait(), timeout=300)
        output = reactor.track("main_video")

        @output.on_frame
        def _frame(rgb):
            t = now()
            if marks["t_first_frame"] is None:
                marks["t_first_frame"] = t
            h, w = rgb.shape[:2]
            frames.append((round(t, 4), w, h))
            if scenario == "lingbot":
                small.append((t, rgb[::8, ::8].mean(axis=2).astype(np.float32)))
            elif scenario == "sana":
                sy, sx = h / SANA_H, w / SANA_W
                region = rgb[int(MARKER[0].start * sy):int(MARKER[0].stop * sy),
                             int(MARKER[1].start * sx):int(MARKER[1].stop * sx)]
                marker_out.append((t, float(region.mean())))

        if scenario == "helios":
            await reactor.send_command("set_seed", {"seed": 1234})
            await reactor.send_command("set_prompt", {"prompt": PROMPT_A})
        elif scenario == "lingbot":
            ref = await reactor.upload_file(LINGBOT_IMAGE)
            await reactor.send_command("set_image", {"image": ref})
            await reactor.send_command("set_seed", {"seed": 1234})
            await reactor.send_command("set_prompt", {"prompt": LINGBOT_PROMPT})
        elif scenario == "sana":
            camera = await reactor.track("camera").publish()
            rng = np.random.default_rng(0)
            base = (rng.integers(90, 170, (SANA_H // 8, SANA_W // 8, 3), dtype=np.uint8)
                    .repeat(8, 0).repeat(8, 1))  # blocky texture so the model has content to keep

            async def publish_loop():
                white, last_flip = False, now()
                marker_in.append((last_flip, white))
                while True:
                    if now() - last_flip >= MARKER_PERIOD_S:
                        white, last_flip = not white, now()
                        marker_in.append((last_flip, white))
                    camera.push_frame(sana_frame(white, base))
                    await asyncio.sleep(1 / SANA_FPS)

            publisher = asyncio.create_task(publish_loop())
            await asyncio.sleep(1)  # let the model see a stable camera before starting
            await reactor.send_command("set_anchor_interval", {"chunks": 1})  # re-ground on the source every chunk
            await reactor.send_command("set_prompt", {"prompt": SANA_PROMPT})

        marks["t_start_sent"] = now()
        await reactor.send_command("start", {})

        ctrl_at = marks["t_start_sent"] + args.seconds / 2
        end_at = marks["t_start_sent"] + args.seconds
        while now() < end_at:
            if scenario in ("helios", "lingbot") and marks["t_ctrl_sent"] is None and now() >= ctrl_at:
                marks["t_ctrl_sent"] = now()
                if scenario == "helios":
                    await reactor.send_command("set_prompt", {"prompt": PROMPT_B})
                else:
                    await reactor.send_command("set_rotation_speed_deg", {"rotation_speed_deg": 30.0})
                    await reactor.send_command("set_look_horizontal", {"look_horizontal": "right"})
            try:
                st = await reactor.get_stats()
                stats.append({"t": round(now(), 2), "rtt_ms": getattr(st, "rtt_ms", None),
                              "candidate": getattr(st, "candidate_type", None)})
            except Exception:
                pass
            await asyncio.sleep(1)
    finally:
        if publisher:
            publisher.cancel()
        await reactor.disconnect()  # non-recoverable: releases the GPU and stops billing

    # --- generic stream metrics ------------------------------------------------------------
    t_ready = next((t for t, s in status_log if s == "ready"), None)
    t_connecting = next((t for t, s in status_log if s == "connecting"), 0.0)
    ff = marks["t_first_frame"]
    times = [f[0] for f in frames]
    gaps = [b - a for a, b in zip(times, times[1:])]
    steady = [t for t in times if ff and t > ff + 5]  # skip first 5 s of ramp-up
    ms = lambda q: pct(gaps, q) * 1000 if gaps else None  # noqa: E731
    summary = {
        "run_id": run_id, "scenario": scenario, "model": model, "seconds": args.seconds,
        "status_log": status_log,
        "time_to_ready_s": (t_ready - t_connecting) if t_ready else None,
        "first_frame_after_start_s": (ff - marks["t_start_sent"]) if ff else None,
        "frames": len(frames),
        "resolution": f"{frames[-1][1]}x{frames[-1][2]}" if frames else None,
        "steady_fps": (len(steady) - 1) / (steady[-1] - steady[0]) if len(steady) > 2 else None,
        "gap_p50_ms": ms(.5), "gap_p95_ms": ms(.95), "gap_p99_ms": ms(.99),
        "gap_max_ms": max(gaps) * 1000 if gaps else None,
        "stalls_over_250ms": sum(g > STALL_GAP_S for g in gaps),
        "rtt_ms_median": statistics.median([s["rtt_ms"] for s in stats if s["rtt_ms"] is not None])
        if any(s["rtt_ms"] is not None for s in stats) else None,
    }

    # --- control-to-screen latency ---------------------------------------------------------
    if scenario == "helios" and marks["t_ctrl_sent"]:
        sw = marks["t_ctrl_sent"]
        summary["control_ack_s"] = (marks["t_switch_accepted"] - sw) if marks["t_switch_accepted"] else None
        summary["control_visible_s"] = (marks["t_switch_visible"] - sw) if marks["t_switch_visible"] else None
    elif scenario == "lingbot" and marks["t_ctrl_sent"] and len(small) > 10:
        sw = marks["t_ctrl_sent"]
        shifts = [(b[0], pan_shift(a[1], b[1])) for a, b in zip(small, small[1:])]
        before = [abs(s) for t, s in shifts if sw - 8 < t < sw]
        thresh = max(1, (pct(before, .95) or 0) + 1)
        after = [(t, s) for t, s in shifts if t > sw]
        visible = next((after[k][0] for k in range(len(after) - 2)
                        if all(abs(after[k + j][1]) >= thresh for j in range(3))), None)
        summary.update(control_visible_s=(visible - sw) if visible else None, pan_threshold_px=thresh,
                       idle_shift_p95_px=pct(before, .95))
    elif scenario == "sana" and marker_out:
        lat = []
        # The edit shifts tones, so split at the midpoint of what actually came back, not at 127.
        lo, hi = pct([v for _, v in marker_out], .05), pct([v for _, v in marker_out], .95)
        mid = (lo + hi) / 2
        summary.update(marker_lum_p05=lo, marker_lum_p95=hi)
        lum = [(t, v > mid) for t, v in marker_out]
        for (_, prev_white), (tb, white) in zip(lum, lum[1:]):
            if white != prev_white:
                sent = [ti for ti, wi in marker_in if wi == white and ti <= tb]
                if sent:
                    lat.append(tb - sent[-1])
        summary.update(roundtrip_samples=len(lat), roundtrip_p50_s=pct(lat, .5), roundtrip_p95_s=pct(lat, .95),
                       roundtrip_min_s=min(lat) if lat else None)

    raw_dir = os.path.join(ROOT, "results", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, f"reactor_{scenario}_{run_id}.json"), "w") as f:
        json.dump({**summary, "stats": stats, "messages": messages[:400], "marker_in": marker_in,
                   "marker_out": [(round(t, 3), round(v, 1)) for t, v in marker_out]}, f, indent=1, default=str)
    with open(os.path.join(raw_dir, f"reactor_{scenario}_{run_id}_frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "width", "height"])
        w.writerows(frames)
    for k, v in summary.items():
        if k != "status_log":
            print(f"{k:28s} {round(v, 3) if isinstance(v, float) else v}")
    print(f"{'status_log':28s} {status_log}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=sorted(MODELS), default="helios")
    ap.add_argument("--seconds", type=int, default=60, help="generation time after `start`")
    asyncio.run(main(ap.parse_args()))
