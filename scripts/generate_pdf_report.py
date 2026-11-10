"""Compiles The_Relay_Report.tex into The_Relay_Report.pdf using pdflatex and verifies exact 2-page constraint."""

import subprocess
import sys
from pathlib import Path
from pypdf import PdfReader


def compile_latex(tex_file: str = "The_Relay_Report.tex", output_pdf: str = "The_Relay_Report.pdf") -> int:
    tex_path = Path(tex_file)
    if not tex_path.exists():
        raise FileNotFoundError(f"LaTeX source file {tex_file} not found.")

    print(f"Compiling {tex_file} with pdflatex...")
    # Two passes to resolve outlines and geometry
    for pass_num in (1, 2):
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file],
            capture_output=True,
            text=True
        )
        if res.returncode != 0:
            print(f"Error on pass {pass_num}:")
            print(res.stdout[-600:] if res.stdout else res.stderr)
            raise RuntimeError(f"pdflatex compilation failed with code {res.returncode}")

    reader = PdfReader(output_pdf)
    num_pages = len(reader.pages)
    print(f"Compilation successful: {output_pdf} generated with exactly {num_pages} pages.")

    if num_pages != 2:
        raise ValueError(f"Constraint violation: Expected exactly 2 pages, but got {num_pages}!")

    print("[SUCCESS] Verified: PDF report is formatted properly and strictly bounded to 2 pages.")
    return num_pages


if __name__ == "__main__":
    compile_latex()
