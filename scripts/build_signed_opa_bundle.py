from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify a signed OPA policy bundle")
    parser.add_argument("--opa", default="opa", help="OPA executable")
    parser.add_argument("--policies", type=Path, default=Path("policies"))
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--verification-key", type=Path, required=True)
    parser.add_argument("--key-id", default="razortrust-policy-current")
    parser.add_argument("--scope", default="razortrust/policies")
    parser.add_argument("--output", type=Path, default=Path("artifacts/opa/razortrust.tar.gz"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="razortrust-opa-claims-") as temp_dir:
        claims_path = Path(temp_dir) / "claims.json"
        claims_path.write_text(
            json.dumps({"scope": args.scope, "iss": "razortrust-policy-build"}),
            encoding="utf-8",
        )
        command = [
            args.opa,
            "build",
            "--bundle",
            str(args.policies),
            "--signing-key",
            str(args.signing_key),
            "--verification-key-id",
            args.key_id,
            "--claims-file",
            str(claims_path),
            "--signing-alg",
            "ES256",
            "--output",
            str(args.output),
        ]
        subprocess.run(command, check=True)
    subprocess.run(
        [
            args.opa,
            "run",
            "--verification-key",
            str(args.verification_key),
            "--verification-key-id",
            args.key_id,
            "--signing-alg",
            "ES256",
            "--scope",
            args.scope,
            "--bundle",
            str(args.output),
        ],
        check=True,
        input="",
        text=True,
        timeout=5,
    )


if __name__ == "__main__":
    main()
