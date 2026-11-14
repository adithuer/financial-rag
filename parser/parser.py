import argparse
from pathlib import Path

import pandas as pd
from docling.document_converter import DocumentConverter
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table import TableItem

# PaddleOCR-VL's allowed labels between two table parts, mapped to docling labels.
# (Paddle's "number", "seal", "header_image" and "footer_image" have no docling label.)
ALLOWED_AFTER_PREV = {DocItemLabel.PAGE_FOOTER, DocItemLabel.FOOTNOTE}
ALLOWED_BEFORE_CURR = {DocItemLabel.PAGE_HEADER}
MAX_HEADER_ROWS = 5


class Parser:
    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, path: Path) -> DoclingDocument:
        """Parse a document with docling."""
        return self.converter.convert(path).document


class TableExtractor:
    """Extract tables and merge tables that continue across pages (PaddleOCR-VL rules)."""

    def __init__(
        self,
        check_width: bool = True,
        max_width_diff: float = 0.1,
        check_position: bool = False,
        min_bottom: float = 0.8,
        max_top: float = 0.15,
    ):
        self.check_width = check_width
        self.max_width_diff = max_width_diff
        self.check_position = check_position
        self.min_bottom = min_bottom
        self.max_top = max_top

    def extract(self, doc: DoclingDocument) -> list[pd.DataFrame]:
        """Return each (merged) table of a parsed document as a DataFrame."""
        items = [item for item, _ in doc.iterate_items()]
        groups: list[list[TableItem]] = []
        for table in doc.tables:
            if groups and self._continues(doc, items, groups[-1][-1], table):
                groups[-1].append(table)
            else:
                groups.append([table])
        return [self._merge(group) for group in groups]

    def _continues(self, doc, items, prev: TableItem, curr: TableItem) -> bool:
        """Decide whether `curr` continues `prev` on the next page."""
        # Rule 1: last table on one page and first table on the next, both non-empty.
        if curr.prov[0].page_no != prev.prov[0].page_no + 1:
            return False
        if not prev.data.num_rows or not curr.data.num_rows:
            return False
        # Rule 2 (optional): table widths differ by less than 10%.
        if self.check_width and not self._similar_width(prev, curr):
            return False
        # Rule 3: only page footers/footnotes after prev, only page headers before curr.
        if not self._only_allowed_between(items, prev, curr):
            return False
        # Rule 4: same column count, or same cell count in the rows at the page break.
        if not self._columns_match(prev, curr):
            return False
        # Rule 5 (optional): prev ends low on its page, curr starts high on its page.
        if self.check_position and not self._at_page_break(doc, prev, curr):
            return False
        return True

    def _similar_width(self, prev: TableItem, curr: TableItem) -> bool:
        w_prev, w_curr = prev.prov[0].bbox.width, curr.prov[0].bbox.width
        if not w_prev or not w_curr:
            return False
        return abs(w_curr - w_prev) / min(w_prev, w_curr) < self.max_width_diff

    def _only_allowed_between(self, items, prev: TableItem, curr: TableItem) -> bool:
        refs = [item.self_ref for item in items]
        prev_page, curr_page = prev.prov[0].page_no, curr.prov[0].page_no
        for item in items[refs.index(prev.self_ref) + 1 : refs.index(curr.self_ref)]:
            if item.parent and item.parent.cref == prev.self_ref:
                continue  # prev's own caption or footnote
            page = item.prov[0].page_no if item.prov else None
            if page == prev_page and item.label in ALLOWED_AFTER_PREV:
                continue
            if page == curr_page and item.label in ALLOWED_BEFORE_CURR:
                continue
            return False
        return not curr.captions  # a caption above curr starts a new table

    def _columns_match(self, prev: TableItem, curr: TableItem) -> bool:
        if prev.data.num_cols == curr.data.num_cols:
            return True
        first = self._repeated_header_rows(prev, curr)  # first data row of curr
        if first >= curr.data.num_rows:
            return False
        last_cells = self._row_cells(prev, prev.data.num_rows - 1)
        first_cells = self._row_cells(curr, first)

        def span(cells):
            return sum(c.end_col_offset_idx - c.start_col_offset_idx for c in cells)

        return span(last_cells) == span(first_cells) or len(last_cells) == len(first_cells)

    def _at_page_break(self, doc, prev: TableItem, curr: TableItem) -> bool:
        p, c = prev.prov[0], curr.prov[0]
        p_height = doc.pages[p.page_no].size.height
        c_height = doc.pages[c.page_no].size.height
        bottom = p.bbox.to_top_left_origin(p_height).b / p_height
        top = c.bbox.to_top_left_origin(c_height).t / c_height
        return bottom >= self.min_bottom and top <= self.max_top

    def _repeated_header_rows(self, prev: TableItem, curr: TableItem) -> int:
        """Count leading rows of curr identical to prev's (ignoring whitespace), at most 5."""
        count = 0
        for row_prev, row_curr in zip(self._rows(prev)[:MAX_HEADER_ROWS], self._rows(curr)):
            if [_strip(t) for t in row_prev] != [_strip(t) for t in row_curr]:
                break
            count += 1
        return count

    @staticmethod
    def _rows(table: TableItem) -> list[list[str]]:
        return [[cell.text for cell in row] for row in table.data.grid]

    @staticmethod
    def _row_cells(table: TableItem, r: int) -> list:
        """Cells starting in row r; a merged cell counts once, like an HTML <td>."""
        return [
            c
            for j, c in enumerate(table.data.grid[r])
            if c.start_row_offset_idx == r and c.start_col_offset_idx == j
        ]

    def _merge(self, parts: list[TableItem]) -> pd.DataFrame:
        """Join table parts; the first grid row is the header (docling's header flags are unreliable)."""
        rows = self._rows(parts[0])
        for prev, curr in zip(parts, parts[1:]):
            rows += self._rows(curr)[self._repeated_header_rows(prev, curr) :]
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        return pd.DataFrame(rows[1:], columns=rows[0])


def _strip(text: str) -> str:
    return "".join(text.split())


def main():
    cli = argparse.ArgumentParser(description="Extract tables from a PDF.")
    cli.add_argument("pdf", type=Path, help="path to the PDF file")
    pdf = cli.parse_args().pdf

    doc = Parser().parse(pdf)
    tables = TableExtractor().extract(doc)
    print(f"\n=== {pdf.name}: {len(tables)} table(s) ===")
    for i, df in enumerate(tables, 1):
        print(f"\n--- Table {i} ---")
        print(df)


if __name__ == "__main__":
    main()
