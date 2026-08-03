#!/usr/bin/env python3
"""
diff_to_pdf.py — Intelligent side-by-side diff of two Python files,
rendered as a dark-mode PDF.

"Intelligent" means the diff is computed with difflib's SequenceMatcher
over the two full line sequences, so lines that were merely *shifted*
(moved up/down because other lines were added/removed elsewhere) are
recognized as unchanged rather than reported as a spurious delete+insert
pair. Only lines whose *content* actually changed are marked.

Layout: two columns, old file on the left (deletions in light red),
new file on the right (additions in light green), unchanged lines
shown on both sides at matching row height, cream text throughout.

Usage:
    python diff_to_pdf.py old_file.py new_file.py -o diff_report.pdf
    python diff_to_pdf.py old_file.py new_file.py --context 3
"""

import argparse
import difflib
import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

# ---------------------------------------------------------------- #
# Palette — dark mode
# ---------------------------------------------------------------- #
BG_COLOR       = (0.02, 0.02, 0.02)     # near-black background
CREAM          = (0.93, 0.91, 0.82)     # unchanged code
LIGHT_GREEN    = (0.62, 0.89, 0.62)     # added lines
LIGHT_RED      = (0.94, 0.55, 0.55)     # removed lines
DIM_CREAM      = (0.50, 0.49, 0.44)     # line numbers / gutter
ACCENT_LINE    = (0.30, 0.30, 0.28)     # rule under header
DIVIDER_COLOR  = (0.22, 0.22, 0.21)     # center column divider
TINT_GREEN     = (0.08, 0.14, 0.08)
TINT_RED       = (0.16, 0.08, 0.08)
GAP_COLOR      = (0.55, 0.68, 0.85)

FONT_MONO      = "Courier"
FONT_MONO_BOLD = "Courier-Bold"
FONT_SIZE      = 8.3
LINE_HEIGHT    = 11.4

PAGE_W, PAGE_H = landscape(LETTER)
MARGIN_L       = 34
MARGIN_R       = 34
MARGIN_TOP     = 56
MARGIN_BOTTOM  = 36
COL_GAP        = 16

GUTTER_W       = 28   # width for line number
MARKER_W       = 10   # width for +/-/space marker

COL_W          = (PAGE_W - MARGIN_L - MARGIN_R - COL_GAP) / 2
LEFT_X         = MARGIN_L
RIGHT_X        = MARGIN_L + COL_W + COL_GAP
LEFT_TEXT_X    = LEFT_X + GUTTER_W + MARKER_W + 4
RIGHT_TEXT_X   = RIGHT_X + GUTTER_W + MARKER_W + 4
LEFT_TEXT_W    = COL_W - GUTTER_W - MARKER_W - 6
RIGHT_TEXT_W   = COL_W - GUTTER_W - MARKER_W - 6


# ---------------------------------------------------------------- #
# Diff computation — produces paired rows for side-by-side display
# ---------------------------------------------------------------- #
def compute_pairs(old_lines, new_lines):
    """
    Returns a list of pair-rows:
        (ltag, lno, ltext, rtag, rno, rtext)
    ltag/rtag in {'equal', 'delete', 'insert', 'blank'}.
    Uses SequenceMatcher so relocated (shifted) but unchanged lines are
    matched as 'equal' rather than delete+insert.
    """
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pairs.append(("equal", i1 + k + 1, old_lines[i1 + k],
                              "equal", j1 + k + 1, new_lines[j1 + k]))
        elif tag == "delete":
            for k in range(i1, i2):
                pairs.append(("delete", k + 1, old_lines[k], "blank", None, ""))
        elif tag == "insert":
            for k in range(j1, j2):
                pairs.append(("blank", None, "", "insert", k + 1, new_lines[k]))
        elif tag == "replace":
            old_count, new_count = i2 - i1, j2 - j1
            common = min(old_count, new_count)
            for k in range(common):
                pairs.append(("delete", i1 + k + 1, old_lines[i1 + k],
                              "insert", j1 + k + 1, new_lines[j1 + k]))
            for k in range(i1 + common, i2):
                pairs.append(("delete", k + 1, old_lines[k], "blank", None, ""))
            for k in range(j1 + common, j2):
                pairs.append(("blank", None, "", "insert", k + 1, new_lines[k]))
    return pairs


def summarize(pairs):
    added = sum(1 for p in pairs if p[3] == "insert")
    removed = sum(1 for p in pairs if p[0] == "delete")
    unchanged = sum(1 for p in pairs if p[0] == "equal")
    return added, removed, unchanged


# ---------------------------------------------------------------- #
# Context collapsing (optional)
# ---------------------------------------------------------------- #
def apply_context(pairs, context):
    if context is None:
        return pairs
    n = len(pairs)
    is_change = [p[0] != "equal" for p in pairs]
    keep = [False] * n
    for i, ch in enumerate(is_change):
        if ch:
            for j in range(max(0, i - context), min(n, i + context + 1)):
                keep[j] = True
    out = []
    i = 0
    while i < n:
        if keep[i]:
            out.append(pairs[i])
            i += 1
        else:
            start = i
            while i < n and not keep[i]:
                i += 1
            out.append(("gap", None, str(i - start), "gap", None, ""))
    return out


# ---------------------------------------------------------------- #
# Text wrapping
# ---------------------------------------------------------------- #
def wrap_text(text, font, size, max_width):
    if not text:
        return [""]
    if stringWidth(text, font, size) <= max_width:
        return [text]

    leading_ws = len(text) - len(text.lstrip(" "))
    indent = " " * min(leading_ws, 16)
    words = text.split(" ")
    lines, current = [], ""
    for idx, w in enumerate(words):
        candidate = (current + " " + w) if current else w
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            if stringWidth(w, font, size) > max_width:
                chunk = ""
                for ch in w:
                    if stringWidth(chunk + ch, font, size) <= max_width:
                        chunk += ch
                    else:
                        lines.append(chunk)
                        chunk = ch
                current = chunk
            else:
                current = w
        if idx == len(words) - 1 and current:
            lines.append(current)
    if not lines:
        lines = [text]
    out = [lines[0]]
    for l in lines[1:]:
        out.append(indent + l)
    return out


# ---------------------------------------------------------------- #
# PDF rendering
# ---------------------------------------------------------------- #
class SideBySideDiffPDF:
    def __init__(self, path, old_name, new_name, added, removed, unchanged):
        self.c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
        self.old_name = old_name
        self.new_name = new_name
        self.added = added
        self.removed = removed
        self.unchanged = unchanged
        self.page_num = 0
        self.y = 0
        self._new_page()

    def _new_page(self):
        if self.page_num > 0:
            self.c.showPage()
        self.page_num += 1
        self.c.setFillColorRGB(*BG_COLOR)
        self.c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        self._draw_header()
        self.y = PAGE_H - MARGIN_TOP - 4

    def _draw_header(self):
        c = self.c
        c.setFont(FONT_MONO_BOLD, 12)
        c.setFillColorRGB(*CREAM)
        c.drawString(MARGIN_L, PAGE_H - 28, "Diff Report")

        stats = f"{self.added} added   {self.removed} removed   {self.unchanged} unchanged"
        c.setFont(FONT_MONO, 8.5)
        c.setFillColorRGB(*DIM_CREAM)
        c.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 28, stats)
        c.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 40, f"page {self.page_num}")

        c.setFont(FONT_MONO_BOLD, 9)
        c.setFillColorRGB(*LIGHT_RED)
        c.drawString(LEFT_X, PAGE_H - 42, f"- {self.old_name}")
        c.setFillColorRGB(*LIGHT_GREEN)
        c.drawString(RIGHT_X, PAGE_H - 42, f"+ {self.new_name}")

        c.setStrokeColorRGB(*ACCENT_LINE)
        c.setLineWidth(0.6)
        c.line(MARGIN_L, PAGE_H - MARGIN_TOP, PAGE_W - MARGIN_R, PAGE_H - MARGIN_TOP)

    def _ensure_space(self, n_lines=1):
        if self.y - n_lines * LINE_HEIGHT < MARGIN_BOTTOM:
            self._new_page()

    def _draw_divider_segment(self, top_y, n_lines):
        self.c.setStrokeColorRGB(*DIVIDER_COLOR)
        self.c.setLineWidth(0.5)
        x = LEFT_X + COL_W + COL_GAP / 2
        self.c.line(x, top_y - n_lines * LINE_HEIGHT + 3, x, top_y + 3)

    def draw_gap(self, count):
        self._ensure_space(1)
        c = self.c
        label = f"⋮  {count} unchanged line(s) hidden"
        c.setFillColorRGB(*GAP_COLOR)
        c.setFont(FONT_MONO, FONT_SIZE)
        c.drawCentredString(PAGE_W / 2, self.y, label)
        self._draw_divider_segment(self.y + LINE_HEIGHT, 1)
        self.y -= LINE_HEIGHT

    def _draw_side(self, x_col, text_x, text_w, tag, lno, text, n_rows, tint):
        c = self.c
        if tag in ("delete", "insert"):
            c.setFillColorRGB(*tint)
            c.rect(x_col - 3, self.y - (n_rows - 1) * LINE_HEIGHT - 2,
                   COL_W + 3, n_rows * LINE_HEIGHT, fill=1, stroke=0)

        if tag == "blank":
            return

        color = LIGHT_GREEN if tag == "insert" else (LIGHT_RED if tag == "delete" else CREAM)
        marker = "+" if tag == "insert" else ("-" if tag == "delete" else " ")

        c.setFillColorRGB(*DIM_CREAM)
        c.setFont(FONT_MONO, FONT_SIZE)
        c.drawRightString(x_col + GUTTER_W - 4, self.y, str(lno) if lno else "")

        c.setFillColorRGB(*color)
        c.setFont(FONT_MONO_BOLD, FONT_SIZE)
        c.drawString(x_col + GUTTER_W, self.y, marker)

        c.setFont(FONT_MONO, FONT_SIZE)
        wrapped = wrap_text(text.rstrip("\n"), FONT_MONO, FONT_SIZE, text_w) or [""]
        yy = self.y
        for line in wrapped:
            c.drawString(text_x, yy, line)
            yy -= LINE_HEIGHT

    def draw_pair(self, ltag, lno, ltext, rtag, rno, rtext):
        left_wrapped = wrap_text(ltext.rstrip("\n"), FONT_MONO, FONT_SIZE, LEFT_TEXT_W) \
            if ltag != "blank" else [""]
        right_wrapped = wrap_text(rtext.rstrip("\n"), FONT_MONO, FONT_SIZE, RIGHT_TEXT_W) \
            if rtag != "blank" else [""]
        n_rows = max(len(left_wrapped), len(right_wrapped), 1)

        self._ensure_space(n_rows)

        self._draw_side(LEFT_X, LEFT_TEXT_X, LEFT_TEXT_W, ltag, lno, ltext, n_rows, TINT_RED)
        self._draw_side(RIGHT_X, RIGHT_TEXT_X, RIGHT_TEXT_W, rtag, rno, rtext, n_rows, TINT_GREEN)
        self._draw_divider_segment(self.y + LINE_HEIGHT, n_rows)

        self.y -= LINE_HEIGHT * n_rows

    def save(self):
        self.c.showPage()
        self.c.save()


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Render an intelligent (shift-aware) side-by-side diff "
                    "of two Python files as a dark-mode PDF."
    )
    ap.add_argument("old_file", type=Path, help="Original / 'before' file")
    ap.add_argument("new_file", type=Path, help="Modified / 'after' file")
    ap.add_argument("-o", "--output", type=Path, default=None,
                     help="Output PDF path (default: <old>_vs_<new>_diff.pdf)")
    ap.add_argument("--context", type=int, default=None,
                     help="Collapse unchanged runs longer than this many "
                          "lines around each change (like diff -U).")
    args = ap.parse_args()

    if not args.old_file.exists():
        sys.exit(f"error: {args.old_file} does not exist")
    if not args.new_file.exists():
        sys.exit(f"error: {args.new_file} does not exist")

    old_lines = args.old_file.read_text(encoding="utf-8", errors="replace").splitlines()
    new_lines = args.new_file.read_text(encoding="utf-8", errors="replace").splitlines()

    pairs = compute_pairs(old_lines, new_lines)
    added, removed, unchanged = summarize(pairs)

    if args.context is not None:
        pairs = apply_context(pairs, args.context)

    out_path = args.output or Path(f"{args.old_file.stem}_vs_{args.new_file.stem}_diff.pdf")

    pdf = SideBySideDiffPDF(out_path, args.old_file.name, args.new_file.name,
                             added, removed, unchanged)

    for ltag, lno, ltext, rtag, rno, rtext in pairs:
        if ltag == "gap":
            pdf.draw_gap(ltext)
        else:
            pdf.draw_pair(ltag, lno, ltext, rtag, rno, rtext)

    pdf.save()
    print(f"Wrote {out_path}  (+{added} / -{removed} / ={unchanged})")


if __name__ == "__main__":
    main()
