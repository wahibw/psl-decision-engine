# utils/pdf_generator.py
# Generates a 2-page A4 PDF pre-match brief using reportlab.
#
# Page 1: Playing 11, toss, weather summary, bowling plan
# Page 2: Opposition batting order, batting scenario cards, matchup notes
#
# Public API:
#   generate_pdf(brief, output_path) -> Path

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, KeepTogether, HRFlowable,
    NextPageTemplate, PageBreak,
)
from reportlab.platypus.flowables import BalancedColumns

from engine.decision_engine import PreMatchBrief

# ---------------------------------------------------------------------------
# COLOUR PALETTE  (dark orange theme adapted for print)
# ---------------------------------------------------------------------------

C_ORANGE    = colors.HexColor("#E87722")
C_DARK      = colors.HexColor("#1A1A1A")
C_DARK_GRAY = colors.HexColor("#2D2D2D")
C_MID_GRAY  = colors.HexColor("#555555")
C_LIGHT     = colors.HexColor("#F5F5F5")
C_WHITE     = colors.white
C_RED       = colors.HexColor("#CC3333")
C_AMBER     = colors.HexColor("#E8A020")
C_GREEN     = colors.HexColor("#2E7D32")
C_BLUE      = colors.HexColor("#1565C0")

PAGE_W, PAGE_H = A4
MARGIN = 1.5 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


# ---------------------------------------------------------------------------
# STYLES
# ---------------------------------------------------------------------------

def _build_styles() -> dict:
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=C_WHITE,
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_LIGHT,
            alignment=TA_LEFT,
        ),
        "section_header": ParagraphStyle(
            "section_header",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=C_ORANGE,
            spaceBefore=6,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=8,
            textColor=C_DARK,
            spaceAfter=2,
            leading=11,
        ),
        "body_small": ParagraphStyle(
            "body_small",
            fontName="Helvetica",
            fontSize=7,
            textColor=C_MID_GRAY,
            leading=10,
        ),
        "bold": ParagraphStyle(
            "bold",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=C_DARK,
        ),
        "toss_big": ParagraphStyle(
            "toss_big",
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=C_ORANGE,
            alignment=TA_CENTER,
        ),
        "danger_high": ParagraphStyle(
            "danger_high",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=C_RED,
        ),
        "danger_med": ParagraphStyle(
            "danger_med",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=C_AMBER,
        ),
        "danger_low": ParagraphStyle(
            "danger_low",
            fontName="Helvetica",
            fontSize=8,
            textColor=C_GREEN,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# PAGE DECORATORS
# ---------------------------------------------------------------------------

def _header_footer(canvas, doc, brief: PreMatchBrief) -> None:
    """Draw header bar and footer on every page."""
    canvas.saveState()

    # Header bar
    canvas.setFillColor(C_DARK)
    canvas.rect(0, PAGE_H - 2.2 * cm, PAGE_W, 2.2 * cm, fill=1, stroke=0)

    canvas.setFillColor(C_ORANGE)
    canvas.rect(0, PAGE_H - 2.2 * cm, 0.4 * cm, 2.2 * cm, fill=1, stroke=0)

    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(MARGIN, PAGE_H - 1.35 * cm, f"{brief.our_team}  vs  {brief.opposition}")

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(C_LIGHT)
    date_str = brief.match_datetime.strftime("%d %b %Y  %H:%M")
    canvas.drawString(MARGIN, PAGE_H - 1.85 * cm, f"{brief.venue}   |   {date_str}")

    # Page number (right side)
    canvas.setFillColor(C_LIGHT)
    canvas.setFont("Helvetica", 8)
    page_label = f"Page {doc.page} of 2   |   PSL Decision Engine"
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.5 * cm, page_label)

    # Footer line
    canvas.setStrokeColor(C_ORANGE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 1.2 * cm, PAGE_W - MARGIN, 1.2 * cm)
    canvas.setFillColor(C_MID_GRAY)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(MARGIN, 0.7 * cm, "CONFIDENTIAL — PSL Decision Intelligence")
    canvas.drawRightString(PAGE_W - MARGIN, 0.7 * cm, brief.generated_at.strftime("Generated %d %b %Y %H:%M"))

    canvas.restoreState()


# ---------------------------------------------------------------------------
# SECTION BUILDERS
# ---------------------------------------------------------------------------

def _section_header(text: str, styles: dict) -> list:
    return [
        Paragraph(text.upper(), styles["section_header"]),
        HRFlowable(width=CONTENT_W, thickness=0.5, color=C_ORANGE, spaceAfter=4),
    ]


def _weather_section(brief: PreMatchBrief, styles: dict) -> list:
    elems = _section_header("Weather & Conditions", styles)
    wi = brief.weather_impact
    cond_rows = [
        ["Spinner penalty", f"{wi.spinner_penalty:.2f}x",
         "Swing bonus",     f"{wi.swing_bonus:.2f}x"],
        ["Pace/bounce",     f"{wi.pace_bounce_bonus:.2f}x",
         "Yorker reliability", f"{wi.yorker_reliability:.2f}x"],
        ["Dew onset over",  str(wi.dew_onset_over) if wi.dew_onset_over else "None",
         "D/L planning",    "YES" if wi.dl_planning_needed else "No"],
    ]
    t = Table(cond_rows, colWidths=[3.5*cm, 2*cm, 3.8*cm, 2*cm])
    t.setStyle(TableStyle([
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,0), (-1,-1), C_DARK),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [C_LIGHT, C_WHITE]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 3*mm))

    for w in brief.weather_warnings[:3]:
        # Strip emoji for PDF rendering
        clean = w.encode("ascii", "ignore").decode("ascii").strip()
        if clean:
            elems.append(Paragraph(f"* {clean}", styles["body_small"]))
    return elems


def _toss_section(brief: PreMatchBrief, styles: dict) -> list:
    elems = _section_header("Toss Recommendation", styles)
    elems.append(Paragraph(brief.toss.recommendation, styles["toss_big"]))
    elems.append(Spacer(1, 2*mm))
    for r in brief.toss.reasoning:
        elems.append(Paragraph(f"* {r}", styles["body"]))
    if brief.toss.dl_note:
        elems.append(Paragraph(f"D/L: {brief.toss.dl_note}", styles["body_small"]))
    return elems


def _xi_section(brief: PreMatchBrief, styles: dict) -> list:
    elems = _section_header("Playing 11 Recommendation", styles)
    if not brief.xi_options:
        elems.append(Paragraph("No XI generated.", styles["body"]))
        return elems

    primary = brief.xi_options[0]
    rows = [["#", "Player", "Role", "Score"]]
    for p in primary.players:
        rows.append([
            str(p.batting_position),
            p.player_name,
            p.role,
            f"{p.score:.0f}",
        ])
    rows.append(["", primary.constraint_note, "", ""])

    col_w = [0.8*cm, 6.5*cm, 4*cm, 1.5*cm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("BACKGROUND",  (0,0), (-1,0), C_DARK),
        ("TEXTCOLOR",   (0,0), (-1,0), C_WHITE),
        ("ROWBACKGROUNDS", (0,1), (-1,-2), [C_LIGHT, C_WHITE]),
        ("FONTNAME",    (0,-1), (-1,-1), "Helvetica-Oblique"),
        ("FONTSIZE",    (0,-1), (-1,-1), 7),
        ("TEXTCOLOR",   (0,-1), (-1,-1), C_MID_GRAY),
        ("SPAN",        (1,-1), (2,-1)),
        ("GRID",        (0,0), (-1,-2), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("ALIGN",       (0,0), (0,-1), "CENTER"),
        ("ALIGN",       (3,0), (3,-1), "CENTER"),
    ]))
    elems.append(t)

    # Alt XI note
    if len(brief.xi_options) > 1:
        elems.append(Spacer(1, 2*mm))
        elems.append(Paragraph(
            f"Alternatives: {brief.xi_options[1].description} and {brief.xi_options[2].description} available on screen.",
            styles["body_small"]
        ))
    return elems


def _bowling_plan_section(brief: PreMatchBrief, styles: dict) -> list:
    elems = _section_header("Over-by-Over Bowling Plan", styles)
    if not brief.bowling_plan:
        elems.append(Paragraph("No bowling plan generated.", styles["body"]))
        return elems

    # Build 4-column layout: Over | Bowler | Phase | Note
    rows = [["Over", "Primary Bowler", "Ph", "Note"]]
    for oa in brief.bowling_plan.overs:
        note = oa.weather_note[:35] if oa.weather_note else oa.reason[:35]
        rows.append([str(oa.over), oa.primary_bowler, oa.phase, note])

    # Single compact table: over pairs side by side (1+11, 2+12, ...)
    paired_rows = [["Ov", "Bowler", "Ph", "Ov", "Bowler", "Ph"]]
    plan_overs = brief.bowling_plan.overs
    for i in range(10):
        lo = plan_overs[i]   if i < len(plan_overs) else None
        ro = plan_overs[i+10] if i+10 < len(plan_overs) else None
        paired_rows.append([
            str(lo.over) if lo else "",
            lo.primary_bowler if lo else "",
            lo.phase if lo else "",
            str(ro.over) if ro else "",
            ro.primary_bowler if ro else "",
            ro.phase if ro else "",
        ])

    col_w = [0.8*cm, 5.8*cm, 1.0*cm, 0.8*cm, 5.8*cm, 1.0*cm]
    t = Table(paired_rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7),
        ("BACKGROUND",  (0,0), (-1,0), C_DARK),
        ("TEXTCOLOR",   (0,0), (-1,0), C_WHITE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LIGHT, C_WHITE]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ("LINEAFTER",   (2,0), (2,-1), 1.0, C_ORANGE),
        ("TOPPADDING",  (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ("ALIGN",       (0,0), (0,-1), "CENTER"),
        ("ALIGN",       (3,0), (3,-1), "CENTER"),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 2*mm))

    # Key decisions
    for kd in brief.bowling_plan.key_decisions[:3]:
        elems.append(Paragraph(f"* {kd}", styles["body_small"]))
    return elems


def _opposition_section(brief: PreMatchBrief, styles: dict) -> list:
    elems = _section_header("Opposition Batting Order Prediction", styles)
    op = brief.opposition_order
    if not op.predicted_order:
        elems.append(Paragraph("No opposition data available.", styles["body"]))
        return elems

    rows = [["Pos", "Player", "Arrival", "Style", "Danger", "Key Note"]]
    for pb in op.predicted_order:
        danger_style = (
            styles["danger_high"] if pb.danger_rating == "High"
            else styles["danger_med"] if pb.danger_rating == "Medium"
            else styles["danger_low"]
        )
        note_clean = pb.key_note[:45] if pb.key_note else ""
        rows.append([
            str(pb.position),
            pb.player_name,
            pb.arrival_over_range.split("—")[-1].strip()[:18],
            pb.batting_style[:5],
            Paragraph(pb.danger_rating, danger_style),
            note_clean,
        ])

    col_w = [0.7*cm, 4.5*cm, 3.0*cm, 1.2*cm, 1.5*cm, 6.6*cm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7),
        ("BACKGROUND",  (0,0), (-1,0), C_DARK),
        ("TEXTCOLOR",   (0,0), (-1,0), C_WHITE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LIGHT, C_WHITE]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",  (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ("ALIGN",       (0,0), (0,-1), "CENTER"),
        ("ALIGN",       (4,0), (4,-1), "CENTER"),
    ]))
    elems.append(t)

    # Bowling implications
    elems.append(Spacer(1, 2*mm))
    for imp in op.bowling_implications[:3]:
        elems.append(Paragraph(f"* {imp}", styles["body_small"]))
    return elems


def _scenarios_section(brief: PreMatchBrief, styles: dict) -> list:
    elems = _section_header("Batting Scenario Cards", styles)
    if not brief.batting_scenarios:
        elems.append(Paragraph("No scenario data.", styles["body"]))
        return elems

    # 2x2 grid of scenario cards
    card_data = []
    for sc in brief.batting_scenarios:
        card_elems = []
        # Card header
        card_elems.append(Paragraph(
            f"[{sc.scenario_id}] {sc.name.upper()}",
            ParagraphStyle("ch", fontName="Helvetica-Bold", fontSize=8,
                           textColor=C_WHITE, backColor=C_DARK_GRAY,
                           spaceAfter=2, leftIndent=3),
        ))
        card_elems.append(Paragraph(sc.key_message, ParagraphStyle(
            "cm", fontName="Helvetica-Oblique", fontSize=7,
            textColor=C_ORANGE, spaceAfter=3, leading=9,
        )))
        # Top 5 batters for this scenario
        for bi in sc.batting_order[:5]:
            card_elems.append(Paragraph(
                f"{bi.position}. {bi.player_name}  [{bi.role_in_card}]",
                ParagraphStyle("cp", fontName="Helvetica", fontSize=7,
                               textColor=C_DARK, leading=10),
            ))
        if sc.weather_note:
            card_elems.append(Paragraph(
                f"* {sc.weather_note[:60]}",
                ParagraphStyle("wn", fontName="Helvetica-Oblique", fontSize=6.5,
                               textColor=C_MID_GRAY, spaceBefore=2, leading=9),
            ))
        card_data.append(card_elems)

    # Arrange in 2 columns
    col_w = (CONTENT_W - 0.3*cm) / 2
    grid = Table(
        [[card_data[0], card_data[1]],
         [card_data[2], card_data[3]]],
        colWidths=[col_w, col_w],
    )
    grid.setStyle(TableStyle([
        ("BOX",         (0,0), (0,0), 0.5, C_ORANGE),
        ("BOX",         (1,0), (1,0), 0.5, C_ORANGE),
        ("BOX",         (0,1), (0,1), 0.5, C_ORANGE),
        ("BOX",         (1,1), (1,1), 0.5, C_ORANGE),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
    ]))
    elems.append(grid)
    return elems


def _matchup_section(brief: PreMatchBrief, styles: dict) -> list:
    elems = _section_header("Key Matchup Notes", styles)
    if not brief.matchup_notes:
        elems.append(Paragraph(
            "No matchups with sufficient H2H data (>= 8 balls) found for this fixture.",
            styles["body_small"]
        ))
        return elems

    for mn in brief.matchup_notes:
        adv_color = C_GREEN if mn.advantage == "bowler" else C_RED
        conf_text = f"[{mn.confidence}]"
        elems.append(Paragraph(
            f"<font color='#{adv_color.hexval()[2:]}'>{conf_text}</font>  {mn.note}",
            styles["body"],
        ))
        elems.append(Spacer(1, 1.5*mm))
    return elems


# ---------------------------------------------------------------------------
# MAIN EXPORT FUNCTION
# ---------------------------------------------------------------------------

def generate_pdf(
    brief:       PreMatchBrief,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a 2-page A4 PDF pre-match brief.

    Args:
        brief:       PreMatchBrief from decision_engine.generate_prematch_brief()
        output_path: Destination path (default: data/processed/brief_<team>_<date>.pdf)

    Returns:
        Path to the generated PDF.
    """
    if output_path is None:
        proj_root   = Path(__file__).resolve().parent.parent
        date_str    = brief.match_datetime.strftime("%Y%m%d")
        team_slug   = brief.our_team.replace(" ", "_")
        output_path = proj_root / "data" / "processed" / f"brief_{team_slug}_{date_str}.pdf"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()

    # Build page frames
    frame = Frame(
        MARGIN, 1.8 * cm,
        CONTENT_W,
        PAGE_H - 2.2 * cm - 1.8 * cm,
        id="main",
    )

    def _on_page(canvas, doc):
        _header_footer(canvas, doc, brief)

    page_template = PageTemplate(id="main", frames=[frame], onPage=_on_page)

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        pageTemplates=[page_template],
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=2.4 * cm,
        bottomMargin=2.0 * cm,
    )

    # --- Build story ---
    story = []

    # PAGE 1
    story += _toss_section(brief, styles)
    story.append(Spacer(1, 4*mm))
    story += _weather_section(brief, styles)
    story.append(Spacer(1, 4*mm))
    story += _xi_section(brief, styles)
    story.append(Spacer(1, 4*mm))
    story += _bowling_plan_section(brief, styles)

    # PAGE 2
    story.append(PageBreak())
    story += _opposition_section(brief, styles)
    story.append(Spacer(1, 4*mm))
    story += _scenarios_section(brief, styles)
    story.append(Spacer(1, 4*mm))
    story += _matchup_section(brief, styles)

    doc.build(story)
    return output_path


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime
    from engine.decision_engine import generate_prematch_brief
    from utils.situation import WeatherImpact

    squad = [
        "Fakhar Zaman", "Abdullah Shafique", "Sikandar Raza",
        "Shaheen Shah Afridi", "Liam Dawson", "Mohammad Hafeez",
        "Rashid Khan", "Haris Rauf", "Zaman Khan",
        "Agha Salman", "Mohammad Nawaz", "Sahibzada Farhan",
    ]

    weather = WeatherImpact(
        spinner_penalty    = 0.60,
        swing_bonus        = 1.15,
        pace_bounce_bonus  = 1.05,
        yorker_reliability = 0.90,
        dl_planning_needed = False,
        dew_onset_over     = 13,
        warnings           = ["Heavy dew expected from over 13. Spinner use restricted."],
    )

    brief = generate_prematch_brief(
        our_team       = "Lahore Qalandars",
        opposition     = "Karachi Kings",
        venue          = "Gaddafi Stadium, Lahore",
        match_datetime = datetime(2025, 3, 20, 19, 0),
        our_squad      = squad,
        weather_impact = weather,
    )

    out = generate_pdf(brief)
    print(f"PDF generated: {out}  ({out.stat().st_size // 1024} KB)")
