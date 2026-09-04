from __future__ import annotations

import base64
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .audit import canonical_json


def generate_release_keypair() -> tuple[str, str]:
    """Generate a development keypair for signing release artifacts.

    Production keys belong in a KMS or HSM. This helper is deliberately not
    exposed through the API.
    """
    private_key = Ed25519PrivateKey.generate()
    private_raw = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(private_raw).decode(), base64.b64encode(public_raw).decode()


def sign_manifest(manifest: dict[str, Any], private_key_b64: str) -> str:
    private_key = Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(private_key_b64, validate=True)
    )
    return base64.b64encode(private_key.sign(canonical_json(manifest))).decode()


def verify_manifest(manifest: dict[str, Any], signature_b64: str, public_key_b64: str) -> bool:
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(public_key_b64, validate=True)
        )
        public_key.verify(
            base64.b64decode(signature_b64, validate=True),
            canonical_json(manifest),
        )
    except (InvalidSignature, ValueError):
        return False
    return True
