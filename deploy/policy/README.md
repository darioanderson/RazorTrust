# Signed policy bundle slot

Place the CI-produced `bundle.tar.gz` here before starting the staging Compose stack. Build it with
`scripts/build_signed_opa_bundle.py`; distribute the ES256 public key separately through the
`OPA_PUBLIC_KEY_FILE` Compose secret. OPA will reject an invalid update and retain its last verified
bundle. No private signing key belongs in this directory or image.
