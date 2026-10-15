"""Verify each provider key with free, non-billable calls. Never prints key values."""
import os

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
TIMEOUT = 20


def check_machgen():
    r = httpx.get(
        "https://api.machgen.ai/api/v0/billing/account",
        headers={"Authorization": f"Bearer {os.environ['MACHGEN_API_KEY']}"},
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        return f"FAIL HTTP {r.status_code}: {r.text[:200]}"
    d = r.json()
    return f"OK  balance=${d['balance_micros'] / 1e6:.2f}  pending={d.get('pending_tasks')} running={d.get('running_tasks')}"


def check_reactor():
    # Minting a token is free: billing starts only once a session reaches `ready`.
    r = httpx.post(
        "https://api.reactor.inc/tokens",
        headers={"Reactor-API-Key": os.environ["REACTOR_API_KEY"]},
        json={
            "expires_after": 60,
            "authorization_details": [
                {"type": "session", "resources": {"models": {"match": ["reactor/helios"]}},
                 "constraints": {"max_sessions": 1, "max_session_duration_seconds": 60}}
            ],
        },
        timeout=TIMEOUT,
    )
    if r.status_code not in (200, 201):
        return f"FAIL HTTP {r.status_code}: {r.text[:200]}"
    return f"OK  token minted (expires_at={r.json().get('expires_at')})"


def check_fal():
    r = httpx.get(
        "https://api.fal.ai/v1/models",
        params={"q": "ltx", "limit": 1},
        headers={"Authorization": f"Key {os.environ['FAL_KEY']}"},
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        return f"FAIL HTTP {r.status_code}: {r.text[:200]}"
    return "OK  model search authorised"


if __name__ == "__main__":
    for name, fn in [("MachGen", check_machgen), ("Reactor", check_reactor), ("fal", check_fal)]:
        try:
            print(f"{name:8s} {fn()}")
        except Exception as e:  # report and keep checking the others
            print(f"{name:8s} ERROR {type(e).__name__}: {e}")
