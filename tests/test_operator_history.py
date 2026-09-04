from __future__ import annotations

from razortrust.operator_history import (
    REQUIRED_COLUMNS,
    HistoryImportValidationError,
    OperatorHistoryStore,
    audit_csv_bytes,
)


def _full_csv() -> bytes:
    header = ",".join(REQUIRED_COLUMNS) + ",fraud_label\n"
    row = (
        "tx_1,order_1,2026-08-01T10:00:00+00:00,"
        "2026-08-01T10:00:01+00:00,1000,INR,captured,card,"
        "device-pseudonym-0001,IN-DL,0,,,,,,,0\n"
    )
    return (header + row).encode()


def test_full_contract_is_recognized() -> None:
    report = audit_csv_bytes(_full_csv())
    assert report.recognized_family == "RAZORTRUST_FULL_HISTORY_CONTRACT"
    assert report.live_13_feature_contract_compatible is True


def test_ulb_schema_is_recognized_but_not_live_compatible() -> None:
    payload = b"Time,V1,V2,Amount,Class\n1,0.1,0.2,50.0,1\n"
    report = audit_csv_bytes(payload)
    assert report.recognized_family == "ULB_EUROPEAN_CREDITCARD_2013"
    assert report.live_13_feature_contract_compatible is False
    assert "customer_geo" in report.missing_required_columns


def test_ieee_schema_is_recognized_but_not_live_compatible() -> None:
    payload = (
        b"TransactionID,isFraud,TransactionDT,TransactionAmt,ProductCD\n"
        b"2987003,1,86499,50,W\n"
    )
    report = audit_csv_bytes(payload)
    assert report.recognized_family == "IEEE_CIS_VESTA"
    assert report.live_13_feature_contract_compatible is False


def test_import_rejects_unattested_history(tmp_path) -> None:
    store = OperatorHistoryStore(tmp_path)
    try:
        store.import_csv_bytes(
            _full_csv(),
            dataset_name="real history",
            account_id="merchant_real",
            source_kind="OPERATOR_SUPPLIED_REAL_HISTORY",
            source_description="merchant export",
            source_url=None,
            user_attested_real_data=False,
        )
    except HistoryImportValidationError as exc:
        assert "attested" in str(exc)
    else:
        raise AssertionError("unattested history was accepted")


def test_import_hashes_and_removes_raw_device_identifier(tmp_path) -> None:
    store = OperatorHistoryStore(tmp_path)
    manifest = store.import_csv_bytes(
        _full_csv(),
        dataset_name="real history",
        account_id="merchant_real",
        source_kind="OPERATOR_SUPPLIED_REAL_HISTORY",
        source_description="merchant export",
        source_url=None,
        user_attested_real_data=True,
    )
    raw = (tmp_path / manifest.dataset_id / "records.jsonl").read_text()
    assert "device-pseudonym-0001" not in raw
    assert manifest.provider_verified is False
    assert manifest.production_action_eligible is False
