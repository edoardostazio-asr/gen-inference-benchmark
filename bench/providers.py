"""Async clients for fal and MachGen that time every phase of a request-response generation.

Client-side phases (all from time.perf_counter, in seconds relative to submit start):
  t_ack            submit request answered (job accepted, id returned)
  t_first_running  first poll that sees the job executing (end of visible queueing)
  t_done           first poll that sees a terminal status (accuracy bounded by poll interval)
  t_download       output file fully downloaded -> the end-to-end latency a user feels
Server-reported fields are copied as-is when a provider exposes them.
"""
import asyncio
import base64
import os
import time
from datetime import datetime, timezone

MACHGEN_BASE = "https://api.machgen.ai/api/v0"
FAL_QUEUE = "https://queue.fal.run"
POLL_IMAGE_S = 0.25
POLL_VIDEO_S = 0.5
JOB_TIMEOUT_S = 900


def _record(provider, suite, i, prompt, seed):
    return {
        "provider": provider, "suite": suite, "i": i, "prompt": prompt, "seed": seed,
        "wall_start": datetime.now(timezone.utc).isoformat(),
        "status": None, "error": None, "job_id": None,
        "t_ack": None, "t_first_running": None, "t_done": None, "t_download": None,
        "server_queue_s": None, "server_gen_s": None, "server_upload_s": None,
        "execution_provider": None, "overflow_converted": None,
        "cost_usd": None, "cost_source": None, "output_path": None, "output_bytes": None,
    }


async def _download(client, url, path, headers=None):
    async with client.stream("GET", url, headers=headers, follow_redirects=True) as r:
        r.raise_for_status()
        n = 0
        with open(path, "wb") as f:
            async for chunk in r.aiter_bytes():
                f.write(chunk)
                n += len(chunk)
    return n


async def run_machgen(client, suite_name, suite, i, prompt, seed, out_dir):
    rec = _record("machgen", suite_name, i, prompt, seed)
    headers = {"Authorization": f"Bearer {os.environ['MACHGEN_API_KEY']}"}
    poll = POLL_IMAGE_S if suite["kind"] == "image" else POLL_VIDEO_S
    body = {**suite["machgen"], "prompt": prompt, "seed": seed}
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{MACHGEN_BASE}/generate", json=body, headers=headers)
        rec["t_ack"] = time.perf_counter() - t0
        if r.status_code >= 300:
            rec.update(status="SUBMIT_ERROR", error=f"HTTP {r.status_code}: {r.text[:300]}")
            return rec
        task_id = r.json()["task_id"]
        rec["job_id"] = task_id
        while True:
            if time.perf_counter() - t0 > JOB_TIMEOUT_S:
                rec.update(status="TIMEOUT")
                return rec
            s = (await client.get(f"{MACHGEN_BASE}/tasks/{task_id}", headers=headers)).json()
            if s["status"] == "RUNNING" and rec["t_first_running"] is None:
                rec["t_first_running"] = time.perf_counter() - t0
            if s["status"] in ("COMPLETED", "FAILED"):
                rec["t_done"] = time.perf_counter() - t0
                break
            await asyncio.sleep(poll)
        rec.update(
            status=s["status"], error=s.get("error_msg"),
            server_queue_s=s.get("queue_time_secs"), server_gen_s=s.get("generation_time_secs"),
            server_upload_s=s.get("upload_time_secs"), execution_provider=s.get("execution_provider"),
            overflow_converted=s.get("overflow_converted"),
        )
        if s["status"] == "COMPLETED":
            url = s["task_output"].get("video") or s["task_output"].get("image")
            ext = "mp4" if suite["kind"] == "video" else "jpg"
            path = os.path.join(out_dir, f"machgen_{suite_name}_{i:03d}.{ext}")
            rec["output_bytes"] = await _download(client, url, path, headers=headers)
            rec["t_download"] = time.perf_counter() - t0
            rec["output_path"] = path
        b = await client.get(f"{MACHGEN_BASE}/tasks/{task_id}/billing", headers=headers)
        if b.status_code == 200:
            # Gross is the list price; net subtracts MachGen's "generation credit" rebate.
            bill = b.json()
            rec.update(cost_usd=bill["generation_cost_micros"] / 1e6, cost_source="billing_api",
                       cost_net_usd=bill["net_cost_micros"] / 1e6)
    except Exception as e:  # keep the run going; the failure is part of the data
        rec.update(status=rec["status"] or "CLIENT_ERROR", error=f"{type(e).__name__}: {e}")
    return rec


async def run_machgen_sync(client, suite_name, suite, i, prompt, seed, out_dir):
    """One blocking call that returns the image bytes: the lowest-latency path MachGen offers."""
    rec = _record("machgen", suite_name, i, prompt, seed)
    headers = {"Authorization": f"Bearer {os.environ['MACHGEN_API_KEY']}"}
    t0 = time.perf_counter()
    try:
        # Sync delivery requires moderation off; fal runs with its safety checker off too.
        r = await client.post(f"{MACHGEN_BASE}/generate/sync", headers=headers,
                              json={**suite["machgen"], "prompt": prompt, "seed": seed, "moderate": False})
        rec["t_download"] = rec["t_done"] = time.perf_counter() - t0
        rec["job_id"] = r.headers.get("X-Machgen-Task-Id")
        if r.status_code == 200:
            path = os.path.join(out_dir, f"machgen_{suite_name}_{i:03d}.jpg")
            with open(path, "wb") as f:
                f.write(r.content)
            rec.update(status="COMPLETED", output_path=path, output_bytes=len(r.content),
                       cost_usd=suite["est_cost"]["machgen"], cost_source="unit_price_estimate")
        else:  # 202 = fell back to the async path; counts as a sync-mode miss
            rec.update(status=f"HTTP_{r.status_code}",
                       error=r.headers.get("X-Machgen-Sync-Fallback-Reason") or r.text[:300])
    except Exception as e:
        rec.update(status="CLIENT_ERROR", error=f"{type(e).__name__}: {e}")
    return rec


async def run_fal_sync(client, suite_name, suite, i, prompt, seed, out_dir, unit_cost):
    """fal's blocking endpoint (fal.run) with the image inlined: its lowest-latency HTTP path."""
    rec = _record("fal", suite_name, i, prompt, seed)
    headers = {"Authorization": f"Key {os.environ['FAL_KEY']}"}
    t0 = time.perf_counter()
    try:
        r = await client.post(f"https://fal.run/{suite['fal']['endpoint']}", headers=headers,
                              json={**suite["fal"]["args"], "prompt": prompt, "seed": seed, "sync_mode": True})
        if r.status_code != 200:
            rec.update(status=f"HTTP_{r.status_code}", error=r.text[:300], t_done=time.perf_counter() - t0)
            return rec
        img = r.json()["images"][0]["url"]
        if img.startswith("data:"):
            data = base64.b64decode(img.split(",", 1)[1])
        else:  # endpoint ignored sync_mode; fetch the file so timing still covers delivery
            data = (await client.get(img)).content
        rec["t_download"] = rec["t_done"] = time.perf_counter() - t0
        path = os.path.join(out_dir, f"fal_{suite_name}_{i:03d}.jpg")
        with open(path, "wb") as f:
            f.write(data)
        rec.update(status="COMPLETED", output_path=path, output_bytes=len(data),
                   cost_usd=unit_cost, cost_source="unit_price_estimate")
    except Exception as e:
        rec.update(status="CLIENT_ERROR", error=f"{type(e).__name__}: {e}")
    return rec


async def run_fal(client, suite_name, suite, i, prompt, seed, out_dir, unit_cost):
    rec = _record("fal", suite_name, i, prompt, seed)
    headers = {"Authorization": f"Key {os.environ['FAL_KEY']}"}
    poll = POLL_IMAGE_S if suite["kind"] == "image" else POLL_VIDEO_S
    body = {**suite["fal"]["args"], "prompt": prompt, "seed": seed}
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{FAL_QUEUE}/{suite['fal']['endpoint']}", json=body, headers=headers)
        rec["t_ack"] = time.perf_counter() - t0
        if r.status_code >= 300:
            rec.update(status="SUBMIT_ERROR", error=f"HTTP {r.status_code}: {r.text[:300]}")
            return rec
        sub = r.json()
        rec["job_id"] = sub["request_id"]
        while True:
            if time.perf_counter() - t0 > JOB_TIMEOUT_S:
                rec.update(status="TIMEOUT")
                return rec
            s = (await client.get(sub["status_url"], headers=headers)).json()
            if s.get("status") == "IN_PROGRESS" and rec["t_first_running"] is None:
                rec["t_first_running"] = time.perf_counter() - t0
            if s.get("status") == "COMPLETED":
                rec["t_done"] = time.perf_counter() - t0
                break
            await asyncio.sleep(poll)
        rec["server_gen_s"] = (s.get("metrics") or {}).get("inference_time")
        res = await client.get(sub["response_url"], headers=headers)
        if res.status_code >= 300:
            rec.update(status="FAILED", error=f"HTTP {res.status_code}: {res.text[:300]}")
            return rec
        out = res.json()
        media = out.get("video") or (out.get("images") or [None])[0]
        rec["status"] = "COMPLETED"
        ext = "mp4" if suite["kind"] == "video" else "jpg"
        path = os.path.join(out_dir, f"fal_{suite_name}_{i:03d}.{ext}")
        rec["output_bytes"] = await _download(client, media["url"], path)
        rec["t_download"] = time.perf_counter() - t0
        rec["output_path"] = path
        # fal does not return a per-request charge; estimate from its published unit price.
        rec.update(cost_usd=unit_cost, cost_source="unit_price_estimate")
    except Exception as e:
        rec.update(status=rec["status"] or "CLIENT_ERROR", error=f"{type(e).__name__}: {e}")
    return rec
