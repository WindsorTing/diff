#!/usr/bin/env python3
"""
diff_to_pdf.py — Intelligent side-by-side diff of two Python files,
rendered as a dark-mode PDF, with substantive-change highlighting and
a hyperlinked function index for data-analysis / plotting code.

"Intelligent" means the diff is computed with difflib's SequenceMatcher
over the two full line sequences, so lines that were merely *shifted*
(moved up/down because other lines were added/removed elsewhere) are
recognized as unchanged rather than reported as a spurious delete+insert
pair. Only lines whose *content* actually changed are marked.

Layout: two columns, old file on the left (deletions in light red),
new file on the right (additions in light green), unchanged lines
shown on both sides at matching row height, cream text throughout.

Substantive-change highlighting: changed lines (+/-) that match a
heuristic set of patterns for numerical computation, data loading/
saving, subsetting, or plotting calls get a yellow star. Comments,
docstrings, imports, prints/logging, and pure renames/formatting are
left unstarred. This is a heuristic, not a static analyzer — skim
the whole diff, use the stars as a guide.

Function index: page 1 lists every `def` found in either file, with
its line number in the old and/or new file, each hyperlinked to that
exact line further in the document (also mirrored in the PDF's
bookmark/outline sidebar). `def` lines are never collapsed by
--context, so the index always resolves to something visible.

Scope annotations: every starred (substantive) line gets a small
"→ in <function>()" and, if it's a continuation of a multi-line call,
"call: <name>(…)" note directly beneath it, so you know which
function/call a significant change belongs to without scrolling.

Usage:
    python diff_to_pdf.py old_file.py new_file.py -o diff_report.pdf
    python diff_to_pdf.py old_file.py new_file.py --context 3
    python diff_to_pdf.py old_file.py new_file.py --no-highlight
    python diff_to_pdf.py old_file.py new_file.py \\
        --highlight-pattern '\\bmy_custom_transform\\('
"""

import argparse
import difflib
import math
import re
import sys
from collections import defaultdict
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
ANNOTATION_COLOR = (0.58, 0.72, 0.88)   # scope/call breadcrumb text
STAR_YELLOW    = (1.00, 0.84, 0.20)
STAR_TINT_ADD  = (0.16, 0.16, 0.05)     # extra glow tint for starred lines
STAR_TINT_DEL  = (0.20, 0.13, 0.05)

FONT_MONO      = "Courier"
FONT_MONO_BOLD = "Courier-Bold"
FONT_SIZE      = 8.3
ANNOT_SIZE     = 7.1
LINE_HEIGHT    = 11.4

PAGE_W, PAGE_H = landscape(LETTER)
MARGIN_L       = 34
MARGIN_R       = 34
MARGIN_TOP     = 62
MARGIN_BOTTOM  = 36
COL_GAP        = 16

GUTTER_W       = 28   # width for line number
MARKER_W       = 10   # width for +/-/space marker
STAR_W         = 12   # width reserved for the significance star

COL_W          = (PAGE_W - MARGIN_L - MARGIN_R - COL_GAP) / 2
LEFT_X         = MARGIN_L
RIGHT_X        = MARGIN_L + COL_W + COL_GAP
LEFT_TEXT_X    = LEFT_X + GUTTER_W + MARKER_W + STAR_W + 4
RIGHT_TEXT_X   = RIGHT_X + GUTTER_W + MARKER_W + STAR_W + 4
LEFT_TEXT_W    = COL_W - GUTTER_W - MARKER_W - STAR_W - 6
RIGHT_TEXT_W   = COL_W - GUTTER_W - MARKER_W - STAR_W - 6


# ---------------------------------------------------------------- #
# Significance heuristics — flags lines likely to affect computed
# results or rendered figures, as opposed to framing/cosmetic code.
# ---------------------------------------------------------------- #
_NOISE_PATTERNS = [
    re.compile(r'^\s*#'),                          # comments
    re.compile(r'^\s*(\'\'\'|""")'),                # docstring delimiter lines
    re.compile(r'^\s*(import|from)\s+\S'),          # imports
    re.compile(r'^\s*(print|logging\.\w+|logger\.\w+)\s*\('),  # logging/prints
    re.compile(r'^\s*$'),                           # blank
]

_SIGNIFICANT_PATTERNS = [
    # numpy / scipy / sklearn / generic numeric namespaces
    r'\bnp\.\w+\s*\(',
    r'\bnumpy\.\w+\s*\(',
    r'\bscipy\b',
    r'\bstats\.\w+\s*\(',
    r'\bsklearn\b',
    r'\.(fit|predict|transform|fit_transform|fit_predict|score)\s*\(',
    # pandas data transforms / aggregations
    r'\bpd\.\w+\s*\(',
    r'\bpandas\.\w+\s*\(',
    r'\.(mean|sum|std|var|median|min|max|corr|cov|count|nunique|quantile|'
    r'cumsum|cumprod|diff|pct_change|rolling|resample|groupby|pivot_table|'
    r'pivot|merge|join|concat|apply|applymap|map|filter|query|fillna|'
    r'dropna|interpolate|astype|clip|round|abs|sort_values|rank|zscore|'
    r'reindex|melt|crosstab)\s*\(',
    # dataset loading / saving — the source of truth for what's actually
    # analyzed or plotted, so a changed path/format/method here is treated
    # as substantive even though it's not itself a "calculation"
    r'\.(read_csv|read_excel|read_json|read_parquet|read_sql|read_pickle|'
    r'read_hdf|read_feather|read_table|read_html|read_xml|read_orc)\s*\(',
    r'\.(to_csv|to_excel|to_json|to_parquet|to_sql|to_pickle|to_hdf|'
    r'to_feather|to_numpy|to_records)\s*\(',
    r'\b(np\.)?(save|savez|savez_compressed|savetxt|load|loadtxt|'
    r'genfromtxt|fromfile)\s*\(',
    r'\bopen\s*\(',
    r'\b(json|pickle|yaml|joblib)\.(load|loads|dump|dumps)\s*\(',
    r'\.imsave\s*\(',
    # user-defined load/save/read/write wrappers — catches both the def
    # line and every call site by name, since teams often wrap pd.read_csv
    # etc. in their own load_/save_ helper rather than calling it directly
    r'\b(load|save|read|write)_\w*\s*\(',
    # subsetting — the other major pivot point: which rows/columns/samples
    # actually make it into the analysis. Covers label/positional indexing,
    # boolean-mask filtering, slicing, sampling, and row/dup dropping.
    r'\.(loc|iloc|at|iat)\s*\[',
    r'\.(where|mask|isin|sample|head|tail|nlargest|nsmallest|drop|'
    r'drop_duplicates|nth|take|truncate)\s*\(',
    r'\w+\[.+[<>=!]=?[^\[\]=]*\]',           # boolean-mask indexing, e.g. df[df['x'] > 5]
    r'\[\s*-?[\w.]*\s*:\s*-?[\w.]*\s*(:\s*-?[\w.]*\s*)?\]',  # slice notation a:b / a:b:c
    # plotting calls — matplotlib / seaborn / plotly
    r'\b(plt|ax|axes|fig|sns|seaborn|px|go|plotly)\.\w*'
    r'(plot|scatter|hist|bar|barh|pie|imshow|contour|contourf|boxplot|'
    r'violinplot|heatmap|lineplot|scatterplot|barplot|histplot|kdeplot|'
    r'regplot|pairplot|Figure|subplots|surface|errorbar|fill_between|'
    r'quiver|streamplot)\w*\s*\(',
    r'\.savefig\s*\(',
    r'\bset_(xlim|ylim|xlabel|ylabel|title)\s*\(',   # axis/scale changes affect figure reading
    # arithmetic assignment on data (heuristic): "x = ... <op> ..." with a
    # math operator, excluding pure default-arg/decorator/comparison lines
    re.compile(r'^\s*[\w\.\[\]\'\", ]+\s=\s.*[-+*/%^]|^\s*[\w\.\[\]]+\s*[-+*/%]=\s'),
]
_SIGNIFICANT_RE = [re.compile(p) if isinstance(p, str) else p
                    for p in _SIGNIFICANT_PATTERNS]


def is_significant(line, extra_patterns=None):
    """
    Heuristic: True if this line looks like it performs a calculation on
    data, loads/saves/subsets it, or produces/configures a plot, as
    opposed to comments, imports, logging, blank lines, or pure
    formatting/renames.
    """
    if not line.strip():
        return False
    for pat in _NOISE_PATTERNS:
        if pat.match(line):
            return False
    for pat in _SIGNIFICANT_RE:
        if pat.search(line):
            return True
    if extra_patterns:
        for pat in extra_patterns:
            if pat.search(line):
                return True
    return False


# ---------------------------------------------------------------- #
# Structural analysis — function/class scope tracking, used for both
# the hyperlinked function index and the inline scope annotations.
# ---------------------------------------------------------------- #
_DEF_RE = re.compile(r'^\s*def\s+(\w+)\s*\(')
_CLASS_RE = re.compile(r'^\s*class\s+(\w+)')
_CALL_OPEN_RE = re.compile(r'([\w.]+)\s*\(\s*$')
_CALL_ANY_RE = re.compile(r'([A-Za-z_][\w.]*)\s*\(')


def analyze_structure(lines):
    """
    Single pass over a file's lines. Returns:
      scope_map — per-line qualified enclosing function/class.method name
                  (None outside any def/class)
      defs      — list of (qualified_name, 1-based lineno) for every
                  `def` line, in file order
    """
    scope_map = []
    defs = []
    stack = []  # list of (indent, qualified_name)
    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped:
            while stack and indent <= stack[-1][0]:
                stack.pop()
        current = stack[-1][1] if stack else None
        m = _DEF_RE.match(line)
        c = _CLASS_RE.match(line)
        if m:
            name = m.group(1)
            qualified = f"{current}.{name}" if current else name
            scope_map.append(qualified)
            defs.append((qualified, i + 1))
            stack.append((indent, qualified))
        elif c:
            name = c.group(1)
            qualified = f"{current}.{name}" if current else name
            scope_map.append(qualified)
            stack.append((indent, qualified))
        else:
            scope_map.append(current)
    return scope_map, defs


def is_def_line(text):
    return bool(text) and bool(_DEF_RE.match(text))


def find_enclosing_call(lines, idx):
    """
    If lines[idx] doesn't itself show a call opening, look back a few
    lines for the nearest multi-line call header it's likely a
    continuation of (e.g. a lone keyword-argument line inside a
    plt.scatter(...) spanning several lines).
    """
    line = lines[idx]
    if _CALL_ANY_RE.search(line):
        return None
    for back in range(1, 9):
        j = idx - back
        if j < 0:
            break
        m = _CALL_OPEN_RE.search(lines[j])
        if m:
            return m.group(1)
        if lines[j].strip() and not lines[j].rstrip().endswith((",", "(", "[", "\\")):
            break
    return None


def build_annotation(lines, scope_map, lineno):
    """Breadcrumb for a significant line: enclosing function + (if it's a
    bare continuation line) the multi-line call it belongs to."""
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return None
    scope = scope_map[idx] if idx < len(scope_map) else None
    call = find_enclosing_call(lines, idx)
    parts = []
    if scope:
        parts.append(f"in {scope}()")
    if call:
        parts.append(f"call: {call}(\u2026)")
    return "   \u2022   ".join(parts) if parts else None


def build_def_index(old_lines, new_lines, old_defs, new_defs):
    """Merge old/new `def` occurrences by name into a unified, sorted
    index for the function table. Duplicate names within one file are
    paired positionally and labeled (#2), (#3), ... for clarity."""
    old_map = defaultdict(list)
    for name, ln in old_defs:
        old_map[name].append(ln)
    new_map = defaultdict(list)
    for name, ln in new_defs:
        new_map[name].append(ln)

    entries = []
    for name in sorted(set(old_map) | set(new_map)):
        olds, news = old_map.get(name, []), new_map.get(name, [])
        n = max(len(olds), len(news))
        for i in range(n):
            label = name if n == 1 else f"{name} (#{i + 1})"
            entries.append({
                "label": label,
                "old_lineno": olds[i] if i < len(olds) else None,
                "new_lineno": news[i] if i < len(news) else None,
            })
    return entries


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


def summarize(pairs, extra_patterns=None):
    added = sum(1 for p in pairs if p[3] == "insert")
    removed = sum(1 for p in pairs if p[0] == "delete")
    unchanged = sum(1 for p in pairs if p[0] == "equal")
    substantive = sum(
        1 for p in pairs
        if (p[0] == "delete" and is_significant(p[2], extra_patterns))
        or (p[3] == "insert" and is_significant(p[5], extra_patterns))
    )
    return added, removed, unchanged, substantive


# ---------------------------------------------------------------- #
# Context collapsing (optional) — `def` lines are always kept so the
# function index can always link somewhere visible, and so a starred
# change deep in a function still shows its header nearby.
# ---------------------------------------------------------------- #
def apply_context(pairs, context):
    if context is None:
        return pairs
    n = len(pairs)
    keep = [False] * n
    for i, p in enumerate(pairs):
        if p[0] != "equal" or is_def_line(p[2]) or is_def_line(p[5]):
            keep[i] = True
    for i in list(range(n)):
        if pairs[i][0] != "equal":
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
# Text wrapping / truncation
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


def truncate_to_width(text, font, size, max_width):
    if stringWidth(text, font, size) <= max_width:
        return text
    ell = "\u2026"
    while text and stringWidth(text + ell, font, size) > max_width:
        text = text[:-1]
    return (text + ell) if text else ell


# ---------------------------------------------------------------- #
# Small vector star (avoids relying on a Unicode glyph in base14 fonts)
# ---------------------------------------------------------------- #
def _star_points(cx, cy, outer_r, inner_r, rotation_deg=-90):
    pts = []
    for i in range(10):
        angle = math.radians(rotation_deg + i * 36)
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def draw_star(c, cx, cy, outer_r, color):
    pts = _star_points(cx, cy, outer_r, outer_r * 0.42)
    p = c.beginPath()
    p.moveTo(*pts[0])
    for pt in pts[1:]:
        p.lineTo(*pt)
    p.close()
    c.setFillColorRGB(*color)
    c.drawPath(p, fill=1, stroke=0)


# ---------------------------------------------------------------- #
# PDF rendering
# ---------------------------------------------------------------- #
class SideBySideDiffPDF:
    def __init__(self, path, old_name, new_name, added, removed, unchanged,
                 substantive, old_lines, new_lines, old_scope_map,
                 new_scope_map, def_entries, highlight=True,
                 extra_patterns=None):
        self.c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
        self.old_name = old_name
        self.new_name = new_name
        self.added = added
        self.removed = removed
        self.unchanged = unchanged
        self.substantive = substantive
        self.highlight = highlight
        self.old_lines = old_lines
        self.new_lines = new_lines
        self.old_scope_map = old_scope_map
        self.new_scope_map = new_scope_map
        self.def_entries = def_entries
        self.extra_patterns = extra_patterns
        self.old_def_anchor = {e["old_lineno"]: f"defold{i}"
                                for i, e in enumerate(def_entries) if e["old_lineno"]}
        self.new_def_anchor = {e["new_lineno"]: f"defnew{i}"
                                for i, e in enumerate(def_entries) if e["new_lineno"]}
        self.page_num = 0
        self.y = 0
        self._diff_started = False
        self._draw_function_index()
        self._new_page()

    # -- page management --------------------------------------------------
    def _bg_page(self):
        self.page_num += 1
        self.c.setFillColorRGB(*BG_COLOR)
        self.c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    def _new_page(self):
        if self.page_num > 0:
            self.c.showPage()
        self._bg_page()
        if not self._diff_started:
            self.c.bookmarkPage("diff_top")
            self._diff_started = True
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

        if self.highlight:
            draw_star(c, PAGE_W - MARGIN_R - stringWidth(
                f"{self.substantive} substantive change(s)", FONT_MONO, 8.5) - 8,
                PAGE_H - 50.5, 3.4, STAR_YELLOW)
            c.setFillColorRGB(*STAR_YELLOW)
            c.setFont(FONT_MONO, 8.5)
            c.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 52,
                               f"{self.substantive} substantive change(s)")

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
        label = f"\u2022 \u2022 \u2022  {count} unchanged line(s) hidden  \u2022 \u2022 \u2022"
        c.setFillColorRGB(*GAP_COLOR)
        c.setFont(FONT_MONO, FONT_SIZE)
        c.drawCentredString(PAGE_W / 2, self.y, label)
        self._draw_divider_segment(self.y + LINE_HEIGHT, 1)
        self.y -= LINE_HEIGHT

    # -- function index (page 1) ------------------------------------------
    def _toc_new_page(self, first=False):
        if self.page_num > 0:
            self.c.showPage()
        self._bg_page()
        c = self.c
        c.setFont(FONT_MONO_BOLD, 14)
        c.setFillColorRGB(*CREAM)
        c.drawString(MARGIN_L, PAGE_H - 30, "Function Index")
        c.setFont(FONT_MONO, 8.5)
        c.setFillColorRGB(*DIM_CREAM)
        c.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 30, f"page {self.page_num}")
        if first:
            c.drawString(MARGIN_L, PAGE_H - 44,
                          f"{len(self.def_entries)} function definition(s) "
                          f"\u2014 click a line number to jump")
        c.setStrokeColorRGB(*ACCENT_LINE)
        c.setLineWidth(0.6)
        c.line(MARGIN_L, PAGE_H - 52, PAGE_W - MARGIN_R, PAGE_H - 52)

        name_x = MARGIN_L
        old_x = MARGIN_L + 360
        new_x = old_x + 100
        c.setFont(FONT_MONO_BOLD, 8.5)
        c.setFillColorRGB(*DIM_CREAM)
        c.drawString(name_x, PAGE_H - 66, "FUNCTION")
        c.setFillColorRGB(*LIGHT_RED)
        c.drawString(old_x, PAGE_H - 66, "OLD")
        c.setFillColorRGB(*LIGHT_GREEN)
        c.drawString(new_x, PAGE_H - 66, "NEW")

        self.y = PAGE_H - 82
        return name_x, old_x, new_x

    def _draw_function_index(self):
        c = self.c
        c.bookmarkPage("toc_top")
        c.addOutlineEntry("Function Index", "toc_top", level=0)

        if self.def_entries:
            name_x, old_x, new_x = self._toc_new_page(first=True)
            row_h = 14
            name_w = old_x - name_x - 10

            for idx, entry in enumerate(self.def_entries):
                if self.y < MARGIN_BOTTOM + row_h:
                    name_x, old_x, new_x = self._toc_new_page(first=False)

                if entry["old_lineno"] is None:
                    name_color = LIGHT_GREEN     # added function
                elif entry["new_lineno"] is None:
                    name_color = LIGHT_RED       # removed function
                else:
                    name_color = CREAM

                c.setFont(FONT_MONO, 8.6)
                c.setFillColorRGB(*name_color)
                c.drawString(name_x, self.y, truncate_to_width(
                    entry["label"], FONT_MONO, 8.6, name_w))

                outline_key = f"defnew{idx}" if entry["new_lineno"] else f"defold{idx}"
                c.addOutlineEntry(entry["label"], outline_key, level=1)

                for x, lineno, anchor, color in (
                    (old_x, entry["old_lineno"], f"defold{idx}", LIGHT_RED),
                    (new_x, entry["new_lineno"], f"defnew{idx}", LIGHT_GREEN),
                ):
                    if lineno:
                        txt = str(lineno)
                        c.setFillColorRGB(*color)
                        c.drawString(x, self.y, txt)
                        w = stringWidth(txt, FONT_MONO, 8.6)
                        c.linkRect("", anchor, (x - 2, self.y - 2, x + w + 2, self.y + 9),
                                   relative=0)
                    else:
                        c.setFillColorRGB(*DIM_CREAM)
                        c.drawString(x, self.y, "\u2014")

                self.y -= row_h

        c.addOutlineEntry("Diff", "diff_top", level=0)

    # -- diff row rendering -------------------------------------------------
    def _draw_side(self, side, x_col, text_x, text_w, tag, lno, text, n_rows,
                    tint, star_tint, significant, annotation):
        c = self.c
        anchor_map = self.old_def_anchor if side == "left" else self.new_def_anchor
        if tag != "blank" and lno and lno in anchor_map:
            c.bookmarkHorizontalAbsolute(anchor_map[lno], self.y + 9)

        if tag in ("delete", "insert"):
            bg = star_tint if (significant and self.highlight) else tint
            c.setFillColorRGB(*bg)
            c.rect(x_col - 3, self.y - (n_rows - 1) * LINE_HEIGHT - 2,
                   COL_W + 3, n_rows * LINE_HEIGHT, fill=1, stroke=0)
            if significant and self.highlight:
                c.setFillColorRGB(*STAR_YELLOW)
                edge_x = (x_col - 3) if side == "left" else (x_col - 3 + COL_W + 3 - 2)
                c.rect(edge_x, self.y - (n_rows - 1) * LINE_HEIGHT - 2, 2,
                       n_rows * LINE_HEIGHT, fill=1, stroke=0)

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

        if significant and self.highlight:
            draw_star(c, x_col + GUTTER_W + MARKER_W + 5, self.y + 2.8, 3.4, STAR_YELLOW)

        c.setFont(FONT_MONO_BOLD if (significant and self.highlight) else FONT_MONO, FONT_SIZE)
        c.setFillColorRGB(*color)
        wrapped = wrap_text(text.rstrip("\n"), FONT_MONO, FONT_SIZE, text_w) or [""]
        yy = self.y
        for line in wrapped:
            c.drawString(text_x, yy, line)
            yy -= LINE_HEIGHT

        if annotation:
            c.setFont(FONT_MONO, ANNOT_SIZE)
            c.setFillColorRGB(*ANNOTATION_COLOR)
            ann = truncate_to_width(f"\u2192 {annotation}", FONT_MONO, ANNOT_SIZE, text_w)
            c.drawString(text_x, yy, ann)

    def draw_pair(self, ltag, lno, ltext, rtag, rno, rtext):
        left_wrapped = wrap_text(ltext.rstrip("\n"), FONT_MONO, FONT_SIZE, LEFT_TEXT_W) \
            if ltag != "blank" else [""]
        right_wrapped = wrap_text(rtext.rstrip("\n"), FONT_MONO, FONT_SIZE, RIGHT_TEXT_W) \
            if rtag != "blank" else [""]

        l_sig = ltag == "delete" and is_significant(ltext, self.extra_patterns)
        r_sig = rtag == "insert" and is_significant(rtext, self.extra_patterns)

        ann_l = build_annotation(self.old_lines, self.old_scope_map, lno) if l_sig else None
        ann_r = build_annotation(self.new_lines, self.new_scope_map, rno) if r_sig else None

        left_total = len(left_wrapped) + (1 if ann_l else 0)
        right_total = len(right_wrapped) + (1 if ann_r else 0)
        n_rows = max(left_total, right_total, 1)

        self._ensure_space(n_rows)

        self._draw_side("left", LEFT_X, LEFT_TEXT_X, LEFT_TEXT_W, ltag, lno, ltext,
                         n_rows, TINT_RED, STAR_TINT_DEL, l_sig, ann_l)
        self._draw_side("right", RIGHT_X, RIGHT_TEXT_X, RIGHT_TEXT_W, rtag, rno, rtext,
                         n_rows, TINT_GREEN, STAR_TINT_ADD, r_sig, ann_r)
        self._draw_divider_segment(self.y + LINE_HEIGHT, n_rows)

        self.y -= LINE_HEIGHT * n_rows

    def save(self):
        self.c.showPage()
        self.c.save()


# ---------------------------------------------------------------- #
# Output path resolution
# ---------------------------------------------------------------- #
def resolve_output_path(explicit_output, old_file, new_file):
    """
    -o/--output always wins. Otherwise: if both source files live in the
    same directory, save the PDF there. If they live in different
    directories, there's no obvious "right" folder, so pop a native save
    dialog and let the person choose. If no display is available (or the
    dialog is dismissed), fall back to a terminal prompt, and if that
    isn't interactive either, fall back to the current working directory.
    """
    if explicit_output:
        return explicit_output

    old_dir = old_file.resolve().parent
    new_dir = new_file.resolve().parent
    default_name = f"{old_file.stem}_vs_{new_file.stem}_diff.pdf"

    if old_dir == new_dir:
        return old_dir / default_name

    print(f"note: {old_file.name} and {new_file.name} are in different "
          f"directories ({old_dir} vs {new_dir}) \u2014 pick a save location.",
          file=sys.stderr)

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.asksaveasfilename(
            title="Save diff PDF as...",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=default_name,
            initialdir=str(old_dir),
        )
        root.destroy()
        if chosen:
            return Path(chosen)
        print("note: save dialog was cancelled; falling back to a prompt.",
              file=sys.stderr)
    except Exception as e:
        print(f"note: could not open a save dialog ({e}); falling back to a prompt.",
              file=sys.stderr)

    try:
        typed = input(f"Save path for diff PDF [default: ./{default_name}]: ").strip()
        if typed:
            return Path(typed)
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)

    fallback = Path.cwd() / default_name
    print(f"note: using default location: {fallback}", file=sys.stderr)
    return fallback


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Render an intelligent (shift-aware) side-by-side diff "
                    "of two Python files as a dark-mode PDF, with a "
                    "hyperlinked function index and yellow-star highlighting "
                    "for changes that affect calculations, data, or plots."
    )
    ap.add_argument("old_file", type=Path, help="Original / 'before' file")
    ap.add_argument("new_file", type=Path, help="Modified / 'after' file")
    ap.add_argument("-o", "--output", type=Path, default=None,
                     help="Output PDF path. Default: saved alongside the "
                          "source files if they share a directory; "
                          "otherwise you'll be prompted (save dialog, "
                          "falling back to a terminal prompt) to pick one.")
    ap.add_argument("--context", type=int, default=None,
                     help="Collapse unchanged runs longer than this many "
                          "lines around each change (like diff -U). "
                          "`def` lines are always kept regardless.")
    ap.add_argument("--no-highlight", action="store_true",
                     help="Disable substantive-change star highlighting "
                          "(and its scope/call annotations).")
    ap.add_argument("--highlight-pattern", action="append", default=[],
                     metavar="REGEX",
                     help="Extra regex pattern (repeatable) that marks a "
                          "changed line as substantive, on top of the "
                          "built-in patterns.")
    args = ap.parse_args()

    if not args.old_file.exists():
        sys.exit(f"error: {args.old_file} does not exist")
    if not args.new_file.exists():
        sys.exit(f"error: {args.new_file} does not exist")

    extra_patterns = [re.compile(p) for p in args.highlight_pattern]

    old_lines = args.old_file.read_text(encoding="utf-8", errors="replace").splitlines()
    new_lines = args.new_file.read_text(encoding="utf-8", errors="replace").splitlines()

    old_scope_map, old_defs = analyze_structure(old_lines)
    new_scope_map, new_defs = analyze_structure(new_lines)
    def_entries = build_def_index(old_lines, new_lines, old_defs, new_defs)

    pairs = compute_pairs(old_lines, new_lines)
    added, removed, unchanged, substantive = summarize(pairs, extra_patterns)

    if args.context is not None:
        pairs = apply_context(pairs, args.context)

    out_path = resolve_output_path(args.output, args.old_file, args.new_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = SideBySideDiffPDF(out_path, args.old_file.name, args.new_file.name,
                             added, removed, unchanged, substantive,
                             old_lines, new_lines, old_scope_map, new_scope_map,
                             def_entries, highlight=not args.no_highlight,
                             extra_patterns=extra_patterns)

    for ltag, lno, ltext, rtag, rno, rtext in pairs:
        if ltag == "gap":
            pdf.draw_gap(ltext)
        else:
            pdf.draw_pair(ltag, lno, ltext, rtag, rno, rtext)

    pdf.save()
    print(f"Wrote {out_path}  (+{added} / -{removed} / ={unchanged} / "
          f"\u2605{substantive} substantive / {len(def_entries)} function(s) indexed)")


if __name__ == "__main__":
    main()
