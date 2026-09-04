# Policy signing

OPA bundle signing follows OPA's supported JWT bundle algorithms. OPA does not list Ed25519 as a native
bundle-signing algorithm, so `scripts/build_signed_opa_bundle.py` uses ES256 rather than claiming an
unsupported primitive. Build keys come from CI/KMS/Vault-mounted files and are not stored in the
repository. The example runtime configuration pins a key ID and scope. OPA activates a new bundle only
after verification; a failed update leaves the current bundle active and must trigger an alert.

Key rotation is additive: publish the next public key out-of-band, sign a bundle with its new key ID,
observe successful activation, then retire the previous key after the rollback window. Never overwrite
a key under an existing key ID.

Official OPA references:

- <https://www.openpolicyagent.org/docs/management-bundles>
- <https://www.openpolicyagent.org/docs/configuration>
