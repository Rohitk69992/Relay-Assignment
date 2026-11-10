"""Text and section extraction from research paper PDFs."""

import re
from pathlib import Path
from typing import Dict, List, Optional
from pypdf import PdfReader


class ExtractedPage:
    def __init__(
        self,
        paper_id: str,
        paper_title: str,
        page_number: int,
        text: str,
        dominant_section: str = "General"
    ):
        self.paper_id = paper_id
        self.paper_title = paper_title
        self.page_number = page_number
        self.text = text
        self.dominant_section = dominant_section

    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "page_number": self.page_number,
            "dominant_section": self.dominant_section,
            "text": self.text,
            "char_count": len(self.text)
        }


class PaperExtractor:
    """Extracts structured text from research PDFs, preserving page numbers and section headers."""

    SECTION_REGEX = re.compile(
        r'^(?:(?:[1-9]\d*(?:\.\d+)*\s+[A-Z][A-Za-z0-9\s,\-\(\)]+)|Abstract|References|Conclusion|Acknowledgments)',
        re.MULTILINE
    )

    def __init__(self, papers_dir: Path):
        self.papers_dir = Path(papers_dir)

    def extract_paper(self, pdf_path: Path, paper_id: str, paper_title: str) -> List[ExtractedPage]:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        reader = PdfReader(str(pdf_path))
        extracted_pages: List[ExtractedPage] = []
        current_section = "Abstract"

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = page.extract_text() or ""
            # Normalize whitespace and clean up hyphenated line breaks
            cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', raw_text)
            cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text).strip()

            # Detect headings on this page
            headings = self.SECTION_REGEX.findall(raw_text)
            if headings:
                current_section = headings[0].strip()

            extracted_pages.append(
                ExtractedPage(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page_number=page_num,
                    text=cleaned_text,
                    dominant_section=current_section
                )
            )

        return extracted_pages
