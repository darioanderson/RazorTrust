from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from uuid import uuid4

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a locally signed Razorpay-shaped webhook fixture to RazorTrust."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--secret", required=True)
    args = parser.parse_args()

    payload = {
        "entity": "event",
        "account_id": "acc_local_smoke",
        "event": "payment.captured",
        "contains": ["payment"],
        "created_at": 1788200000,
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_local_smoke",
                    "entity": "payment",
                    "amount": 10000,
                    "currency": "INR",
                    "status": "captured",
                    "method": "upi",
                }
            }
        },
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    signature = hmac.new(args.secret.encode(), raw, hashlib.sha256).hexdigest()
    event_id = f"evt_local_{uuid4().hex}"
    response = httpx.post(
        f"{args.base_url.rstrip('/')}/v1/integrations/razorpay/webhook",
        content=raw,
        headers={
            "content-type": "application/json",
            "X-Razorpay-Signature": signature,
            "x-razorpay-event-id": event_id,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))
    print("NOTE: this is a local transport/signature smoke fixture, not a live Razorpay event.")


if __name__ == "__main__":
    main()
