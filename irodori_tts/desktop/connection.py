"""Can this computer reach Hugging Face? A light request, not a download.

Only Hugging Face publishes a status page (status.huggingface.co, Better Stack format); it tells
a problem of Hugging Face itself from a network that cannot get there.
"""

from __future__ import annotations

import time

from irodori_tts.desktop.messages import t

from .base_models import default_catalog_base

STATUS_URL = "https://status.huggingface.co/index.json"
TIMEOUT = 5.0


def reason(exc: Exception):
    """A short, readable cause for a failed request."""
    import httpx

    text = str(exc).lower()
    if isinstance(exc, httpx.TimeoutException):
        return t("connection.timeout")
    if "getaddrinfo" in text or "name or service" in text or "nodename" in text:
        return t("connection.no_dns")
    if "certificate" in text or "ssl" in text:
        return t("connection.certificate")
    return t("connection.unreachable")


def hub_status(client) -> str | None:
    """'operational', 'degraded', 'downtime' (Hugging Face's own report), or None."""
    try:
        data = client.get(STATUS_URL).json()
    except Exception:
        return None
    attributes = (data.get("data") or {}).get("attributes") or {}
    return attributes.get("aggregate_state")


def check_hugging_face() -> dict:
    """{"reachable", "latency_ms", "reason", "status"}: one metadata request for the default
    base model, and the official status when that page answers. Never raises: the page
    waits for this answer, and a failure without one left it checking for good."""
    try:
        return _check_hugging_face()
    except Exception as exc:
        return {"reachable": False, "latency_ms": None, "reason": reason(exc), "status": None}


def _check_hugging_face() -> dict:
    import httpx

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        start = time.monotonic()
        try:
            response = client.get(
                f"https://huggingface.co/api/models/{default_catalog_base().repo}"
            )
            latency = round((time.monotonic() - start) * 1000)
            if response.status_code != 200:
                result = {
                    "reachable": False,
                    "latency_ms": latency,
                    "reason": t("connection.http_status", code=response.status_code),
                }
            else:
                result = {"reachable": True, "latency_ms": latency, "reason": None}
        except Exception as exc:
            result = {"reachable": False, "latency_ms": None, "reason": reason(exc)}
        result["status"] = hub_status(client)
    return result
