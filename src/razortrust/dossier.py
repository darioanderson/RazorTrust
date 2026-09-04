from __future__ import annotations

from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .domain import DecisionResponse, HoldCase

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#667085")
LIME = colors.HexColor("#B6F13D")
PANEL = colors.HexColor("#F5F7FA")
LINE = colors.HexColor("#D8DEE8")
RED = colors.HexColor("#B42318")


def build_evidence_dossier(
    hold: HoldCase,
    decision: DecisionResponse,
    audit_records: list[dict[str, Any]],
) -> bytes:
    """Create a deterministic, human-readable case dossier from recorded facts only."""
    stream = BytesIO()
    document = SimpleDocTemplate(
        stream,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=f"RazorTrust evidence dossier {hold.hold_id}",
        author="RazorTrust",
    )
    styles = _styles()
    story: list[Any] = [
        Table(
            [
                [
                    Paragraph(
                        ("<b>RAZORTRUST</b><br/><font size='8'>EVIDENCE DOSSIER</font>"),
                        styles["brand"],
                    ),
                    Paragraph(
                        (
                            f"Case {escape(str(hold.hold_id)[:8].upper())}<br/>"
                            f"{decision.created_at:%Y-%m-%d %H:%M UTC}"
                        ),
                        styles["meta_right"],
                    ),
                ]
            ],
            colWidths=[110 * mm, 55 * mm],
        ),
        Spacer(1, 10 * mm),
        Paragraph("Decision summary", styles["h1"]),
        Paragraph(
            "AI recommends. Human authorizes. RazorTrust records the complete proof chain.",
            styles["lede"],
        ),
        Spacer(1, 4 * mm),
        _summary_table(hold, decision, styles),
        Spacer(1, 5 * mm),
        Paragraph("Why this decision", styles["h2"]),
        Paragraph(_decision_explanation(decision), styles["body"]),
        Spacer(1, 3 * mm),
        _probability_table(decision, styles),
        Spacer(1, 7 * mm),
        Paragraph("Contributing signals", styles["h2"]),
        _signal_table(decision, styles),
        Spacer(1, 7 * mm),
        KeepTogether(
            [
                Paragraph("Evidence and recommendation", styles["h2"]),
                _evidence_section(decision, audit_records, styles),
            ]
        ),
        Spacer(1, 5 * mm),
        Paragraph("Attribution and signed audit timeline", styles["h1"]),
        Paragraph(
            "Each entry is linked to the previous record by SHA-256. Actor identity "
            "and artifact versions are retained with the decision.",
            styles["lede"],
        ),
        Spacer(1, 5 * mm),
        _attribution_table(decision, audit_records, styles),
        Spacer(1, 5 * mm),
        *_timeline(audit_records, styles),
        Spacer(1, 8 * mm),
        Paragraph("Verification", styles["h2"]),
        _verification_table(decision, audit_records, styles),
        Spacer(1, 5 * mm),
        Paragraph(
            "Research and demonstration use only. This dossier does not authorize "
            "movement of funds. A named human reviewer must approve any real action.",
            styles["warning"],
        ),
    ]
    document.build(
        story,
        onFirstPage=_page_footer,
        onLaterPages=_page_footer,
    )
    return stream.getvalue()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "brand": ParagraphStyle(
            "brand", parent=base["BodyText"], textColor=INK, fontSize=14, leading=16
        ),
        "meta_right": ParagraphStyle(
            "meta_right",
            parent=base["BodyText"],
            textColor=MUTED,
            fontSize=8,
            leading=12,
            alignment=TA_RIGHT,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], textColor=INK, fontSize=22, leading=26, spaceAfter=5
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], textColor=INK, fontSize=13, leading=17, spaceAfter=5
        ),
        "lede": ParagraphStyle(
            "lede", parent=base["BodyText"], textColor=MUTED, fontSize=9.5, leading=14
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], textColor=INK, fontSize=9, leading=13
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], textColor=MUTED, fontSize=7.5, leading=10
        ),
        "warning": ParagraphStyle(
            "warning",
            parent=base["BodyText"],
            textColor=RED,
            backColor=colors.HexColor("#FEF3F2"),
            borderColor=colors.HexColor("#FECDCA"),
            borderWidth=0.5,
            borderPadding=8,
            fontSize=8,
            leading=12,
        ),
    }


def _summary_table(hold: HoldCase, decision: DecisionResponse, styles: dict[str, Any]) -> Table:
    risk_score = 1.0 - decision.probabilities.release
    values = [
        ["Merchant", hold.merchant_id, "Decision", decision.decision],
        ["Payment / settlement", hold.source_event_id, "Risk score", f"{risk_score:.1%}"],
        ["Triggered", hold.triggered_at.strftime("%Y-%m-%d %H:%M UTC"), "State", hold.state],
        ["Case ID", str(hold.hold_id), "Trace ID", decision.trace_id],
    ]
    return _key_value_table(values, styles)


def _probability_table(decision: DecisionResponse, styles: dict[str, Any]) -> Table:
    values = decision.probabilities
    rows: list[list[Any]] = [
        ["RELEASE", f"{values.release:.1%}"],
        ["EVIDENCE_NEEDED", f"{values.evidence_needed:.1%}"],
        ["ESCALATE", f"{values.escalate:.1%}"],
        [
            "Confidence / uncertainty",
            f"{max(values.release, values.evidence_needed, values.escalate):.1%}",
        ],
    ]
    return _key_value_table(rows, styles, widths=[62 * mm, 103 * mm])


def _signal_table(decision: DecisionResponse, styles: dict[str, Any]) -> Table:
    rows: list[list[Any]] = [["Signal", "Observed", "Direction", "Contribution"]]
    for signal in decision.top_features[:8]:
        rows.append(
            [
                Paragraph(escape(signal.feature.replace("_", " ").title()), styles["body"]),
                f"{signal.observed_value:.4f}",
                signal.direction.replace("toward_", ""),
                f"{signal.contribution_value:+.4f}",
            ]
        )
    if len(rows) == 1:
        rows.append(["No model-level signals available", "-", "-", "-"])
    table = Table(rows, colWidths=[67 * mm, 28 * mm, 42 * mm, 28 * mm], repeatRows=1)
    table.setStyle(_table_style(header=True))
    return table


def _evidence_section(
    decision: DecisionResponse,
    audit_records: list[dict[str, Any]],
    styles: dict[str, Any],
) -> Table:
    evidence = [
        record for record in audit_records if record.get("event_type") == "EVIDENCE_SUBMITTED"
    ]
    evidence_hashes = []
    for record in evidence:
        payload = record.get("payload", {})
        if isinstance(payload, dict):
            evidence_hashes.append(str(payload.get("content_sha256", "recorded in audit entry")))
    guidance = decision.merchant_guidance
    recommendation = guidance.next_step if guidance else _decision_explanation(decision)
    rows: list[list[Any]] = [
        ["Evidence status", f"{len(evidence)} supporting document reference(s) recorded"],
        ["Document hashes", "\n".join(evidence_hashes) or "No document hash recorded"],
        ["Recommended action", recommendation],
        ["Evidence round", str(decision.evidence_round)],
    ]
    return _key_value_table(rows, styles, widths=[43 * mm, 122 * mm])


def _attribution_table(
    decision: DecisionResponse,
    audit_records: list[dict[str, Any]],
    styles: dict[str, Any],
) -> Table:
    actors = []
    for record in audit_records:
        actor = record.get("actor")
        if isinstance(actor, dict):
            actors.append(f"{actor.get('type', 'UNKNOWN')}:{actor.get('id', 'unknown')}")
        elif record.get("event_type") == "RISK_DECISION":
            actors.append("AI:risk-decision-service")
    rows = [
        ["Human / agent attribution", ", ".join(dict.fromkeys(actors)) or "Not recorded"],
        ["Model", decision.model_version],
        ["Feature set", decision.feature_schema_version],
        ["Policy", decision.policy_version],
        ["Cost matrix", f"{decision.cost_matrix_version} / {decision.cost_matrix_sha256}"],
    ]
    return _key_value_table(rows, styles, widths=[43 * mm, 122 * mm])


def _timeline(audit_records: list[dict[str, Any]], styles: dict[str, Any]) -> list[Any]:
    flow: list[Any] = [Paragraph("Timeline", styles["h2"])]
    if not audit_records:
        flow.append(Paragraph("No audit entries are available.", styles["body"]))
        return flow
    for record in audit_records:
        actor = record.get("actor") or {}
        actor_text = actor.get("id", "system") if isinstance(actor, dict) else "system"
        title = escape(f"{record.get('sequence_no', '?')}. {record.get('event_type', 'EVENT')}")
        details = (
            f"{escape(str(record.get('timestamp', 'timestamp unavailable')))} | "
            f"actor {escape(str(actor_text))}<br/>"
            f"hash {escape(str(record.get('record_hash', 'not available')))}"
        )
        flow.append(
            KeepTogether(
                [
                    Paragraph(f"<b>{title}</b>", styles["body"]),
                    Paragraph(details, styles["small"]),
                    Spacer(1, 2 * mm),
                ]
            )
        )
    return flow


def _verification_table(
    decision: DecisionResponse,
    audit_records: list[dict[str, Any]],
    styles: dict[str, Any],
) -> Table:
    audit_head = audit_records[-1].get("record_hash") if audit_records else decision.audit_head_hash
    rows: list[list[Any]] = [
        ["Audit head", audit_head],
        ["Decision audit reference", decision.audit_head_hash],
        ["Audit entries", str(len(audit_records))],
        ["Proof chain", "SHA-256 hash linked; verify against the authoritative audit ledger"],
    ]
    return _key_value_table(rows, styles, widths=[43 * mm, 122 * mm])


def _key_value_table(
    rows: list[list[Any]],
    styles: dict[str, Any],
    widths: list[float] | None = None,
) -> Table:
    rendered = []
    for row in rows:
        rendered.append(
            [
                Paragraph(f"<b>{escape(str(value))}</b>", styles["small"])
                if index % 2 == 0
                else Paragraph(escape(str(value)), styles["body"])
                for index, value in enumerate(row)
            ]
        )
    table = Table(rendered, colWidths=widths or [31 * mm, 51.5 * mm, 31 * mm, 51.5 * mm])
    table.setStyle(_table_style())
    return table


def _table_style(*, header: bool = False) -> TableStyle:
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
            ]
        )
    return TableStyle(commands)


def _decision_explanation(decision: DecisionResponse) -> str:
    if decision.decision == "RELEASE":
        return (
            "RELEASE was allowed because calibrated legitimate probability cleared the "
            "policy threshold and no fail-closed guardrail overrode it."
        )
    if decision.decision == "EVIDENCE_NEEDED":
        return (
            "RELEASE was not allowed. The system needs corroborating evidence before a "
            "human reviewer can authorize the settlement."
        )
    return (
        "RELEASE was not allowed because the risk and policy signals require a named "
        "human reviewer to decide the next action."
    )


def _page_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8 * mm, "RazorTrust - Confidential research dossier")
    canvas.drawRightString(width - 18 * mm, 8 * mm, f"Page {document.page}")
    canvas.setFillColor(LIME)
    canvas.rect(18 * mm, 14.5 * mm, 18 * mm, 1.2 * mm, fill=1, stroke=0)
    canvas.restoreState()
