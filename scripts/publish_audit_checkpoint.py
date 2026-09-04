from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from razortrust.audit_checkpoint import AuditCheckpoint, verify_audit_checkpoint


def _secret(name: str) -> str | None:
    value = os.getenv(name)
    file_name = os.getenv(f"{name}_FILE")
    if value and file_name:
        raise ValueError(f"configure only one of {name} or {name}_FILE")
    if file_name:
        return Path(file_name).read_text(encoding="utf-8").strip()
    return value


def publish(checkpoint_path: Path, destination: str, public_key: str) -> str:
    payload = checkpoint_path.read_bytes()
    checkpoint = AuditCheckpoint.model_validate(json.loads(payload))
    if not verify_audit_checkpoint(checkpoint, public_key):
        raise ValueError("audit checkpoint signature is invalid")
    digest = hashlib.sha256(payload).hexdigest()
    object_url = f"{destination.rstrip('/')}/{digest}.json"
    request = urllib.request.Request(
        object_url,
        data=payload,
        method="PUT",
        headers={
            "Content-Type": "application/json",
            "Digest": f"sha-256={digest}",
            "If-None-Match": "*",
        },
    )
    token = _secret("RAZORTRUST_WORM_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            if response.status not in {200, 201, 204}:
                raise RuntimeError(f"WORM destination returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        if exc.code in {409, 412}:
            raise RuntimeError("checkpoint already exists; immutable overwrite refused") from exc
        raise RuntimeError(f"WORM destination returned HTTP {exc.code}") from exc
    return object_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and publish one signed audit checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", help="HTTPS WORM object endpoint")
    parser.add_argument("--public-key", default=_secret("RAZORTRUST_AUDIT_PUBLIC_KEY"))
    args = parser.parse_args()
    if not args.public_key:
        parser.error("--public-key or RAZORTRUST_AUDIT_PUBLIC_KEY[_FILE] is required")
    if not args.destination.lower().startswith("https://"):
        parser.error("destination must use HTTPS")
    print(publish(args.checkpoint, args.destination, args.public_key))


if __name__ == "__main__":
    main()
