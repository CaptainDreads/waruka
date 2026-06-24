# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Regenerate the user-facing PDF docs from their Markdown sources.

The committed PDFs (`docs/cli_reference.pdf`, `docs/gui_walkthrough.pdf`)
are derived from the `.md` files next to them. Run this after editing
either Markdown source so the PDFs stay in sync:

    python scripts/build_docs.py                 # rebuild both PDFs
    python scripts/build_docs.py SRC OUT TITLE   # rebuild one

Supports a lean subset of Markdown: headings (h1-h4), paragraphs,
bullet/numbered lists, pipe tables, fenced code blocks, inline code,
bold, italic, images, and horizontal rules -- enough for these docs.
Requires `reportlab` and `Pillow` (see requirements.txt).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default doc set: (markdown source, pdf output, running-footer title).
DOCS = [
    ("docs/cli_reference.md", "docs/cli_reference.pdf", "Waruka CLI Reference (v1.0.0)"),
    ("docs/gui_walkthrough.md", "docs/gui_walkthrough.pdf", "Waruka GUI Reference (v1.0.0)"),
]


def _styles():
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.5, leading=13, spaceAfter=4, alignment=TA_LEFT)
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=20, leading=26, spaceBefore=18, spaceAfter=10,
        textColor=colors.HexColor("#1a1a2e"))
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=15, leading=20, spaceBefore=14, spaceAfter=8,
        textColor=colors.HexColor("#16213e"))
    h3 = ParagraphStyle(
        "H3", parent=styles["Heading3"], fontName="Helvetica-Bold",
        fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=6,
        textColor=colors.HexColor("#0f3460"))
    h4 = ParagraphStyle(
        "H4", parent=styles["Heading4"], fontName="Helvetica-Bold",
        fontSize=11, leading=14, spaceBefore=8, spaceAfter=5)
    code = ParagraphStyle(
        "Code", parent=styles["Code"], fontName="Courier",
        fontSize=8.5, leading=11, leftIndent=8, rightIndent=8,
        spaceBefore=4, spaceAfter=6,
        backColor=colors.HexColor("#f2f2f6"),
        borderColor=colors.HexColor("#cfcfd6"),
        borderWidth=0.4, borderPadding=6)
    list_item = ParagraphStyle(
        "ListItem", parent=body, leftIndent=18, bulletIndent=4, spaceAfter=2)
    table_cell = ParagraphStyle(
        "TableCell", parent=body, fontSize=8.5, leading=11,
        spaceAfter=0, spaceBefore=0)
    table_header = ParagraphStyle(
        "TableHeader", parent=table_cell, fontName="Helvetica-Bold",
        textColor=colors.whitesmoke)
    return dict(body=body, h1=h1, h2=h2, h3=h3, h4=h4, code=code,
                list_item=list_item, table_cell=table_cell,
                table_header=table_header)


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(t: str) -> str:
    """Convert Markdown inline syntax to reportlab Paragraph XML markup."""
    t = _esc(t)
    code_spans: list[str] = []

    def _stash(m):
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans) - 1}\x00"

    t = re.sub(r"`([^`]+)`", _stash, t)
    t = re.sub(r"\*\*([^*]+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)

    def _restore(m):
        idx = int(m.group(1))
        return (f'<font face="Courier" backColor="#f2f2f6">'
                f'{code_spans[idx]}</font>')

    return re.sub(r"\x00CODE(\d+)\x00", _restore, t)


def _parse(src: Path, st: dict) -> list:
    """Parse a Markdown file into a list of reportlab flowables."""
    lines = src.read_text(encoding="utf-8").splitlines()
    flow: list = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped in ("---", "***", "___"):
            flow.append(Spacer(1, 6))
            flow.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc"),
                                   thickness=0.6, spaceBefore=2, spaceAfter=8))
            i += 1
            continue
        img_m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if img_m:
            path = img_m.group(2)
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(os.path.abspath(src)), path)
            if os.path.exists(path):
                try:
                    img = Image(path)
                    scale = min(1.0, (170 * mm) / img.drawWidth)
                    img.drawWidth *= scale
                    img.drawHeight *= scale
                    flow += [Spacer(1, 4), img, Spacer(1, 6)]
                except Exception:
                    flow.append(Paragraph(f"<i>(image: {path})</i>", st["body"]))
            else:
                flow.append(Paragraph(f"<i>(missing image: {path})</i>", st["body"]))
            i += 1
            continue
        for prefix, style in (("# ", "h1"), ("## ", "h2"),
                              ("### ", "h3"), ("#### ", "h4")):
            if stripped.startswith(prefix):
                flow.append(Paragraph(_inline(stripped[len(prefix):]), st[style]))
                i += 1
                break
        else:
            if stripped.startswith("```"):
                i += 1
                buf = []
                while i < n and not lines[i].strip().startswith("```"):
                    buf.append(lines[i])
                    i += 1
                i += 1
                flow.append(Preformatted("\n".join(buf), st["code"]))
                continue
            if stripped.startswith("|") and stripped.endswith("|") and i + 1 < n:
                nxt = lines[i + 1].strip()
                if nxt.startswith("|") and "---" in nxt:
                    rows = [[c.strip() for c in stripped.strip("|").split("|")]]
                    i += 2
                    while (i < n and lines[i].strip().startswith("|")
                           and lines[i].strip().endswith("|")):
                        rows.append([c.strip() for c in
                                     lines[i].strip().strip("|").split("|")])
                        i += 1
                    data = [[Paragraph(_inline(c),
                                       st["table_header"] if ri == 0 else st["table_cell"])
                             for c in r] for ri, r in enumerate(rows)]
                    ncols = len(rows[0])
                    tbl = Table(data, colWidths=[(170 * mm) / ncols] * ncols,
                                repeatRows=1)
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                         [colors.HexColor("#fbfbfd"), colors.HexColor("#f2f2f6")]),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cfcfd6")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]))
                    flow += [Spacer(1, 4), tbl, Spacer(1, 6)]
                    continue
            bullet_m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
            if bullet_m:
                depth = len(bullet_m.group(1)) // 2
                para = Paragraph(_inline(bullet_m.group(2)), st["list_item"],
                                 bulletText="•")
                para.style.leftIndent = 18 + depth * 14
                flow.append(para)
                i += 1
                continue
            num_m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
            if num_m:
                depth = len(num_m.group(1)) // 2
                para = Paragraph(_inline(num_m.group(3)), st["list_item"],
                                 bulletText=f"{num_m.group(2)}.")
                para.style.leftIndent = 18 + depth * 14
                flow.append(para)
                i += 1
                continue
            buf = [stripped]
            i += 1
            while i < n and lines[i].strip() and not (
                    lines[i].strip().startswith("#")
                    or lines[i].strip().startswith("```")
                    or lines[i].strip().startswith("|")
                    or re.match(r"^(\s*)[-*]\s+", lines[i])
                    or re.match(r"^(\s*)\d+\.\s+", lines[i])):
                buf.append(lines[i].strip())
                i += 1
            flow.append(Paragraph(_inline(" ".join(buf)), st["body"]))
    return flow


def build_pdf(src: str | Path, out: str | Path, title: str) -> None:
    src, out = Path(src), Path(out)
    st = _styles()
    flow = _parse(src, st)

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(15 * mm, 10 * mm, title)
        canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=title, author="Waruka")
    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    print(f"wrote {out}")


def main(argv: list[str]) -> int:
    if len(argv) == 3:
        build_pdf(argv[0], argv[1], argv[2])
    elif not argv:
        for rel_src, rel_out, title in DOCS:
            build_pdf(REPO_ROOT / rel_src, REPO_ROOT / rel_out, title)
    else:
        print("usage: python scripts/build_docs.py [SRC OUT TITLE]")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
