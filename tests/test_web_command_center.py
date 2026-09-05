from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "razortrust" / "web"


class _StrictEnoughHtmlParser(HTMLParser):
    pass


def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def test_live_workflow_contains_governed_primary_sections() -> None:
    html = _read("index.html")

    assert 'id="main-content"' in html
    assert "Live Case Processing Workflow" in html

    assert "PUBLIC REAL-WORLD BENCHMARK" in html
    assert 'id="benchmark-case-select"' in html
    assert 'id="stage-grid"' in html
    assert 'id="conclusion-panel"' in html
    assert "CONCLUSION LAYER" in html
    assert 'id="dispute-training-note"' in html
    assert 'id="history-json"' in html
    assert 'id="history-json-file"' in html
    assert 'id="history-real-attestation"' in html
    assert 'id="case-details"' in html
    assert 'id="service-health"' in html
    assert 'id="model-runtime"' in html

    assert "HUMAN_ONLY" in html
    assert "AUTO RELEASE OFF" in html

    assert "TRUTH CONTRACT" in html
    assert "Production action eligible <strong>NO</strong>" in html
    assert "Final authority <strong>HUMAN_ONLY</strong>" in html

    assert "Sequence LSTM" not in html
    assert "v3B false RELEASE" not in html
    assert "v3B RELEASE recall" not in html


def test_frontend_preserves_human_gate_and_native_keyboard_paths() -> None:
    html = _read("index.html")
    script = _read("app.js")

    # Use native accessible controls instead of pseudo-button divs.
    assert "<button" in html
    assert "<input" in html
    assert "<select" in html

    assert 'class="skip-link"' in html
    assert 'aria-live="polite"' in html

    # Explicit identifier field supports keyboard execution.
    assert 'event.key === "Enter"' in script

    # Public benchmark executes through backend endpoints.
    assert "/v1/public-benchmark/ulb/cases" in script
    assert "/execute" in script

    # Research benchmark stays human-gated.
    assert "research only / HUMAN_ONLY" in script
    assert '"HUMAN_ONLY"' in script

    assert "benchmark_recommendation" in script
    assert "Preparing verified ULB data" in script
    assert "renderConclusion(result.conclusion)" in script
    assert "DISPUTE PIPELINE READY - AWAITING MATURE LABELS" in script
    assert "ULB fraud labels are" in script
    assert "not being misrepresented as disputes" in script
    assert "/v1/operator-history/import" in script
    assert "user_attested_real_data" in script
    assert "historyJsonAsCsv" in script
    assert "All case stores responded" in script
    assert 'api("/health/ready")' in script


def test_css_has_visible_focus_and_minimum_control_targets() -> None:
    css = _read("app.css")

    assert ":focus-visible" in css

    assert (
        "outline: 3px solid rgba(91, 140, 255, 0.75)"
        in css
    )

    assert "outline-offset: 2px" in css

    # Current workflow exceeds the previous 40px project minimum.
    assert "min-height: 42px" in css

    assert "--blue: #5b8cff" in css
    assert "var(--border)" not in css

    assert css.count("{") == css.count("}")


def test_html_parses_and_source_is_ascii_safe() -> None:
    for name in ("index.html", "checkout-r4c.html"):
        content = _read(name)

        parser = _StrictEnoughHtmlParser()
        parser.feed(content)
        parser.close()

        assert all(
            ord(character) < 128
            for character in content
        )

    for name in ("app.js", "app.css"):
        content = _read(name)

        assert all(
            ord(character) < 128
            for character in content
        )


def test_r4c_checkout_keeps_test_mode_and_server_verification_contract() -> None:
    checkout = _read("checkout-r4c.html")
    assert "checkout.razorpay.com/v1/checkout.js" in checkout
    assert "/v1/integrations/razorpay/checkout/orders" in checkout
    assert "/v1/integrations/razorpay/checkout/verify" in checkout
    assert "RAZORPAY TEST MODE" in checkout
    assert "SHADOW ONLY" in checkout
    assert "NO SETTLEMENT RELEASE" in checkout
