"""DOCX export — TOS + CILOs + exam items.

Extracted from the 250-line `_build_docx` in dashboard.py. Split into focused
methods so each section is independently readable and testable.

Page: long bond paper 8.5 × 13 in, 1-inch margins, Arial 11pt black-and-white.
"""
from __future__ import annotations

from io import BytesIO
from typing import List, Optional

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# Column widths for the TOS table, in twips (1 inch = 1440 twips).
_COL_W = [
    int(2.00 * 1440),   # Content Outline
    int(0.65 * 1440),   # Hours
    int(1.10 * 1440),   # FAM
    int(1.10 * 1440),   # INT
    int(1.10 * 1440),   # CRE
    int(0.55 * 1440),   # Total
]  # sum = 9360 twips = 6.5 in (matches body width after 1" margins)


def build_docx(
    *,
    title: str,
    cilos: List[str],
    topics: List[dict],
    quizzes: List[dict],
    fam_pct: int,
    int_pct: int,
    cre_pct: int,
    total_items: int,
) -> BytesIO:
    """Public entry point. Returns an in-memory .docx."""
    return _DocxBuilder(
        title=title, cilos=cilos, topics=topics, quizzes=quizzes,
        fam_pct=fam_pct, int_pct=int_pct, cre_pct=cre_pct,
        total_items=total_items,
    ).build()


# ──────────────────────────────────────────────────────────────────────
class _DocxBuilder:
    """Owns the Document and writes the three sections in order."""

    _BLACK = RGBColor(0, 0, 0)

    def __init__(
        self, *, title, cilos, topics, quizzes,
        fam_pct, int_pct, cre_pct, total_items,
    ):
        self.title = title
        self.cilos = cilos
        self.topics = topics
        self.quizzes = quizzes
        self.fam_pct = fam_pct
        self.int_pct = int_pct
        self.cre_pct = cre_pct
        self.total_items = total_items

        self.doc = Document()
        self._configure_page()
        self._configure_default_style()

    # ── Public ─────────────────────────────────────────────────
    def build(self) -> BytesIO:
        if self.cilos:
            self._add_cilos_section()
        self._add_tos_table()
        self._add_exam_items()

        buf = BytesIO()
        self.doc.save(buf)
        buf.seek(0)
        return buf

    # ── Page / style config ────────────────────────────────────
    def _configure_page(self) -> None:
        sec = self.doc.sections[0]
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(13)
        sec.top_margin = sec.bottom_margin = Inches(1)
        sec.left_margin = sec.right_margin = Inches(1)

    def _configure_default_style(self) -> None:
        normal = self.doc.styles["Normal"]
        normal.font.name = "Arial"
        normal.font.size = Pt(11)
        normal.font.color.rgb = self._BLACK

    # ── Low-level helpers ──────────────────────────────────────
    def _run(self, para, text: str, *, bold=False, size=11, italic=False):
        r = para.add_run(text or "")
        r.font.name = "Arial"
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = self._BLACK
        return r

    def _add_para(
        self,
        text: str = "",
        *,
        bold: bool = False,
        size: int = 11,
        align=WD_ALIGN_PARAGRAPH.LEFT,
        italic: bool = False,
        space_before: int = 0,
        space_after: int = 4,
    ):
        p = self.doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        if text:
            self._run(p, text, bold=bold, size=size, italic=italic)
        return p

    def _cell_write(
        self, cell, text: str,
        *, bold=False, size=9,
        align=WD_ALIGN_PARAGRAPH.CENTER, italic=False,
        valign=WD_ALIGN_VERTICAL.CENTER,
    ) -> None:
        cell.vertical_alignment = valign
        lines = (text or "").split("\n")

        p0 = cell.paragraphs[0]
        p0.alignment = align
        p0.paragraph_format.space_before = Pt(1)
        p0.paragraph_format.space_after = Pt(1)
        p0.paragraph_format.left_indent = Inches(0.05)
        p0.paragraph_format.right_indent = Inches(0.05)
        self._run(p0, lines[0], bold=bold, size=size, italic=italic)

        for line in lines[1:]:
            p = cell.add_paragraph()
            p.alignment = align
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.left_indent = Inches(0.05)
            p.paragraph_format.right_indent = Inches(0.05)
            self._run(p, line, bold=bold, size=size, italic=italic)

    @staticmethod
    def _shade_cell(cell, hex_color: str = "E0E0E0") -> None:
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tc_pr.append(shd)

    @staticmethod
    def _set_cell_width(cell, twips: int) -> None:
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_w = OxmlElement("w:tcW")
        tc_w.set(qn("w:w"), str(twips))
        tc_w.set(qn("w:type"), "dxa")
        tc_pr.append(tc_w)

    @staticmethod
    def _set_table_width(table, twips: int) -> None:
        tbl_w = OxmlElement("w:tblW")
        tbl_w.set(qn("w:w"), str(twips))
        tbl_w.set(qn("w:type"), "dxa")
        table._tbl.tblPr.append(tbl_w)

    @staticmethod
    def _underline_paragraph(p) -> None:
        p_pr = p._p.get_or_add_pPr()
        borders = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), "000000")
        borders.append(bottom)
        p_pr.append(borders)

    # ── Section 1: CILOs ───────────────────────────────────────
    def _add_cilos_section(self) -> None:
        self._add_para(
            "Cognitive Objectives / Behavioral Dimensions / Thinking Skills",
            bold=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER,
            space_after=2,
        )
        self._add_para(
            "Intended Learning Outcomes (CILO):",
            bold=True, size=10, space_before=2, space_after=3,
        )
        for i, c in enumerate(self.cilos, 1):
            p = self.doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(2)
            self._run(p, f"{i}.  {c}", size=10)
        self._add_para(space_after=8)

    # ── Section 2: TOS Table ───────────────────────────────────
    def _add_tos_table(self) -> None:
        self._add_para(
            "TABLE OF SPECIFICATION",
            bold=True, size=11,
            align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=6,
        )

        n = len(self.topics)
        tbl = self.doc.add_table(rows=3 + n + 1, cols=6)
        tbl.style = "Table Grid"
        self._set_table_width(tbl, sum(_COL_W))

        # Set column widths on every cell (required for vertical-merge accuracy).
        for row in tbl.rows:
            for ci, cell in enumerate(row.cells):
                self._set_cell_width(cell, _COL_W[ci])

        # Merge "Content Outline" & "Hours" across the three header rows.
        tbl.cell(0, 0).merge(tbl.cell(2, 0))
        tbl.cell(0, 1).merge(tbl.cell(2, 1))

        self._write_tos_header(tbl)
        self._write_tos_data(tbl, start_row=3)
        self._add_para(space_after=14)

    def _write_tos_header(self, tbl) -> None:
        # Row 0 — column titles.
        self._cell_write(tbl.cell(0, 0), "CONTENT OUTLINE",
                         bold=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
        self._cell_write(tbl.cell(0, 1), "NUMBER OF\nHOURS SPENT",
                         bold=True, size=9)
        self._cell_write(tbl.cell(0, 2),
                         "FAMILIARIZATION\n(Remembering / Understanding)",
                         bold=True, size=9)
        self._cell_write(tbl.cell(0, 3),
                         "INTEGRATION\n(Applying / Analyzing)",
                         bold=True, size=9)
        self._cell_write(tbl.cell(0, 4),
                         "CREATION\n(Evaluating / Creating)",
                         bold=True, size=9)
        self._cell_write(tbl.cell(0, 5), "TOTAL", bold=True, size=9)
        for ci in range(6):
            self._shade_cell(tbl.cell(0, ci), "D0D0D0")

        # Row 1 — percentages.
        self._cell_write(tbl.cell(1, 2), f"(Percentage)\n{self.fam_pct}%",
                         size=9, italic=True)
        self._cell_write(tbl.cell(1, 3), f"(Percentage)\n{self.int_pct}%",
                         size=9, italic=True)
        self._cell_write(tbl.cell(1, 4), f"(Percentage)\n{self.cre_pct}%",
                         size=9, italic=True)
        self._cell_write(tbl.cell(1, 5), "100%", size=9, italic=True)
        for ci in range(2, 6):
            self._shade_cell(tbl.cell(1, ci), "EBEBEB")

        # Row 2 — sub-header labels.
        for col, label in ((2, "Item Numbers"), (3, "Item Numbers"),
                           (4, "Item Numbers"), (5, "Total No.\nof Items")):
            self._cell_write(tbl.cell(2, col), label, bold=True, size=9)
        for ci in range(2, 6):
            self._shade_cell(tbl.cell(2, ci), "EBEBEB")

    def _write_tos_data(self, tbl, *, start_row: int) -> None:
        t_hrs = t_fam = t_int = t_cre = t_tot = 0

        for i, t in enumerate(self.topics):
            row = tbl.rows[start_row + i]
            hrs = t.get("hours") or 0
            fam = t.get("fam") or 0
            intg = t.get("int") or 0
            cre = t.get("cre") or 0
            tot = t.get("items") or t.get("quiz_items") or 0
            t_hrs += hrs; t_fam += fam; t_int += intg
            t_cre += cre; t_tot += tot

            self._cell_write(row.cells[0], t.get("topic") or "",
                             bold=True, size=10, align=WD_ALIGN_PARAGRAPH.LEFT)
            self._cell_write(row.cells[1], str(hrs), size=10)
            self._cell_write(row.cells[2], t.get("fam_range") or "—", size=10)
            self._cell_write(row.cells[3], t.get("int_range") or "—", size=10)
            self._cell_write(row.cells[4], t.get("cre_range") or "—", size=10)
            self._cell_write(row.cells[5], str(tot), bold=True, size=10)

        # Totals row.
        footer = tbl.rows[start_row + len(self.topics)]
        self._cell_write(footer.cells[0], "TOTAL:",
                         bold=True, size=10, align=WD_ALIGN_PARAGRAPH.RIGHT)
        self._cell_write(footer.cells[1], str(t_hrs), bold=True, size=10)
        self._cell_write(footer.cells[2], str(t_fam), bold=True, size=10)
        self._cell_write(footer.cells[3], str(t_int), bold=True, size=10)
        self._cell_write(footer.cells[4], str(t_cre), bold=True, size=10)
        self._cell_write(footer.cells[5], str(t_tot), bold=True, size=10)
        for ci in range(6):
            self._shade_cell(footer.cells[ci], "D0D0D0")

    # ── Section 3: Exam Items ──────────────────────────────────
    def _add_exam_items(self) -> None:
        self._add_para(
            "EXAM ITEMS",
            bold=True, size=12,
            align=WD_ALIGN_PARAGRAPH.CENTER,
            space_before=6, space_after=8,
        )

        current_test: Optional[str] = ""
        for idx, q in enumerate(self.quizzes, 1):
            if not isinstance(q, dict):
                continue

            header = q.get("test_header") or ""
            desc = q.get("test_description") or ""
            if header and header != current_test:
                current_test = header
                self._write_test_header(header, desc)

            self._write_question(idx, q)

    def _write_test_header(self, header: str, desc: str) -> None:
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        self._run(p, header, bold=True, size=12)
        if desc:
            self._run(p, f"   —   {desc}", italic=True, size=11)
        self._underline_paragraph(p)

    def _write_question(self, idx: int, q: dict) -> None:
        qtype = (q.get("type") or "").lower()

        p_q = self.doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(4)
        p_q.paragraph_format.space_after = Pt(2)
        p_q.paragraph_format.keep_with_next = True
        self._run(p_q, f"{idx}.  {q.get('question') or ''}", size=11)

        if qtype == "mcq" and q.get("choices"):
            self._write_mcq_choices(q)
        elif qtype in ("truefalse", "tf", "true_false"):
            p = self.doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            self._run(p, "(True or False)", italic=True, size=11)

        self._write_answer_line(q)

    def _write_mcq_choices(self, q: dict) -> None:
        letters = ["a", "b", "c", "d", "e"]
        ans_raw = (q.get("answer") or "").strip()
        ans_low = ans_raw.lower()
        ans_upper = ans_raw.upper().rstrip(".")
        ans_is_letter = ans_upper in ("A", "B", "C", "D")

        for ci, choice in enumerate(q["choices"][:5]):
            letter = letters[ci]
            choice_l = (choice or "").lower()
            if ans_is_letter:
                # v31: answer is a letter — match by index
                is_correct = (ans_upper == letter.upper())
            else:
                # Legacy: answer is the full choice text
                is_correct = (
                    choice_l == ans_low
                    or (ans_low and len(ans_low) > 5 and ans_low in choice_l)
                    or ans_low == letter
                )
            p = self.doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            self._run(p, f"{letter})  {choice or ''}", bold=is_correct, size=11)

    def _write_answer_line(self, q: dict) -> None:
        p = self.doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(7)
        self._run(p, "Answer: ", bold=True, size=11)

        qtype = (q.get("type") or "").lower()
        ans_raw = str(q.get("answer") or "").strip()
        ans_upper = ans_raw.upper().rstrip(".")

        # MCQ + letter answer → display "A) <choice text>"
        if qtype == "mcq" and ans_upper in ("A", "B", "C", "D"):
            choices = q.get("choices") or []
            idx = ord(ans_upper) - ord("A")
            if 0 <= idx < len(choices):
                self._run(p, f"{ans_upper}) {choices[idx]}", size=11)
            else:
                self._run(p, ans_raw or "—", size=11)
        else:
            self._run(p, ans_raw or "—", size=11)

        if q.get("answer_text") and not q.get("_invalid_tf"):
            self._run(p, f"  ({q['answer_text']})", italic=True, size=10)