"""PDF ingestion: /Docs -> Markdown in /Ingestion/MID_ingest.

Latin-script pages are processed with PyMuPDF4LLM + RapidOCR. Pages that
contain Arabic are routed to Tesseract using the ara language pack.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf

INGESTION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INGESTION_DIR.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "Docs"
DEFAULT_OUTPUT_DIR = INGESTION_DIR / "MID_ingest"
TESSDATA_DIR = INGESTION_DIR / "tessdata"

TESSDATA_LANGS = ("eng", "fra", "ara")
TESSERACT_LANG = "ara+fra+eng"
TESSDATA_URL = (
    "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{lang}.traineddata"
)

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def _system_tessdata_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get("TESSDATA_PREFIX")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path("/usr/share/tesseract/tessdata"))
    candidates.append(Path("/usr/share/tessdata"))
    for base in (Path("/usr/share/tesseract"), Path("/usr/share/tesseract-ocr")):
        if base.is_dir():
            candidates.extend(sorted(base.glob("*/tessdata")))
    return candidates


def _download_traineddata(lang: str, target: Path) -> None:
    url = TESSDATA_URL.format(lang=lang)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def ensure_tessdata() -> list[str]:
    TESSDATA_DIR.mkdir(parents=True, exist_ok=True)
    system_dirs = _system_tessdata_dirs()
    missing: list[str] = []

    for lang in TESSDATA_LANGS:
        target = TESSDATA_DIR / f"{lang}.traineddata"
        if target.is_file() and target.stat().st_size > 0:
            continue

        source = next(
            (directory / target.name for directory in system_dirs if (directory / target.name).is_file()),
            None,
        )
        if source is not None:
            shutil.copy2(source, target)
            continue

        try:
            _download_traineddata(lang, target)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            missing.append(lang)
            print(
                f"[warning] Could not obtain {lang}.traineddata: {exc}",
                file=sys.stderr,
            )

    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)
    return missing


def _tessdata_available(lang: str) -> bool:
    target = TESSDATA_DIR / f"{lang}.traineddata"
    return target.is_file() and target.stat().st_size > 0


def _page_text(page: pymupdf.Page) -> str:
    try:
        return page.get_text("text")
    except Exception:
        return ""


def _page_has_arabic(page: pymupdf.Page) -> bool:
    return bool(ARABIC_RE.search(_page_text(page)))


def _arabic_stats(text: str) -> tuple[int, float]:
    arabic = len(ARABIC_RE.findall(text))
    visible = len(re.sub(r"\s+", "", text))
    ratio = arabic / visible if visible else 0.0
    return arabic, ratio


def _document_has_arabic_text(doc: pymupdf.Document) -> bool:
    total = 0
    for page in doc:
        total += len(ARABIC_RE.findall(_page_text(page)))
        if total >= 5:
            return True
    return False


def _looks_scanned(
    doc: pymupdf.Document, sample_pages: int, min_chars_per_page: int = 40
) -> bool:
    pages = min(doc.page_count, sample_pages)
    if pages <= 0:
        return False
    total = sum(len(_page_text(doc[index]).strip()) for index in range(pages))
    return total < min_chars_per_page * pages


def _sample_ocr_text(
    doc: pymupdf.Document, sample_pages: int, dpi: int, language: str
) -> str:
    parts: list[str] = []
    tessdata = pymupdf.get_tessdata()
    for index in range(min(doc.page_count, sample_pages)):
        pixmap = doc[index].get_pixmap(dpi=dpi)
        data = pixmap.pdfocr_tobytes(language=language, tessdata=tessdata)
        sample_doc = pymupdf.open("pdf", data)
        try:
            parts.append(sample_doc[0].get_text())
        finally:
            sample_doc.close()
    return "\n".join(parts)


def _detect_ocr_mode(
    doc: pymupdf.Document, sample_pages: int, sample_dpi: int
) -> str:
    if _document_has_arabic_text(doc):
        return "mixed"

    if not _looks_scanned(doc, sample_pages):
        return "default"

    if not _tessdata_available("ara"):
        print(
            "[warning] ara.traineddata is unavailable; scanned Arabic pages will use RapidOCR.",
            file=sys.stderr,
        )
        return "default"

    sample_text = _sample_ocr_text(doc, sample_pages, sample_dpi, TESSERACT_LANG)
    arabic_count, arabic_ratio = _arabic_stats(sample_text)
    if arabic_count >= 10 and arabic_ratio >= 0.02:
        return "scanned"

    return "default"


def _make_ocr_function(
    mode: str, force_tesseract: bool, force_rapidocr: bool
):
    from pymupdf4llm.ocr import rapidocr_api, tesseract_api

    def hybrid_ocr(
        page: pymupdf.Page,
        dpi: int = 150,
        pixmap: pymupdf.Pixmap | None = None,
        language: str = "eng",
        keep_ocr_text: bool = False,
    ):
        if force_tesseract:
            use_tesseract = True
        elif force_rapidocr:
            use_tesseract = False
        elif mode == "scanned":
            use_tesseract = True
        elif mode == "mixed":
            use_tesseract = _page_has_arabic(page)
        else:
            use_tesseract = False

        if use_tesseract:
            if not _tessdata_available("ara"):
                raise RuntimeError(
                    "Arabic OCR requested but ara.traineddata is missing. "
                    "Install tesseract-langpack-ara or allow the loader to download it."
                )
            return tesseract_api.exec_ocr(
                page,
                dpi=dpi,
                pixmap=pixmap,
                language=TESSERACT_LANG,
                keep_ocr_text=keep_ocr_text,
            )

        return rapidocr_api.exec_ocr(
            page,
            dpi=dpi,
            pixmap=pixmap,
            language=language,
            keep_ocr_text=keep_ocr_text,
        )

    return hybrid_ocr


def _convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    ocr_function,
    ocr_dpi: int,
    force_ocr: bool,
    show_progress: bool,
) -> tuple[Path, int]:
    import pymupdf4llm

    markdown = pymupdf4llm.to_markdown(
        str(pdf_path),
        ocr_function=ocr_function,
        use_ocr=True,
        force_ocr=force_ocr,
        ocr_dpi=ocr_dpi,
        write_images=False,
        embed_images=False,
        show_progress=show_progress,
    )
    output_path = output_dir / f"{pdf_path.stem}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path, len(markdown)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs from /Docs to Markdown using RapidOCR and Tesseract."
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help=f"Source PDF directory (default: {DEFAULT_DOCS_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Markdown output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob used to select files in --docs (default: *.pdf)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional list of file names or stems to process",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=200,
        help="DPI used for OCR rendering (default: 200)",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=2,
        help="Number of pages used to detect scanned Arabic (default: 2)",
    )
    parser.add_argument(
        "--sample-dpi",
        type=int,
        default=120,
        help="DPI used for scanned-language detection (default: 120)",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on every page, including pages with extractable text",
    )
    parser.add_argument(
        "--force-tesseract",
        action="store_true",
        help="Always use Tesseract (ara+fra+eng) when OCR runs",
    )
    parser.add_argument(
        "--force-rapidocr",
        action="store_true",
        help="Always use RapidOCR when OCR runs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents whose Markdown output already exists",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show PyMuPDF4LLM progress output",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first conversion error",
    )
    args = parser.parse_args(argv)

    if args.force_tesseract and args.force_rapidocr:
        parser.error("--force-tesseract and --force-rapidocr are mutually exclusive")

    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    missing = ensure_tessdata()
    if "ara" in missing:
        print(
            "[warning] Tesseract Arabic support is unavailable. "
            "Arabic OCR will fail if requested.",
            file=sys.stderr,
        )

    docs_dir = args.docs.expanduser().resolve()
    output_dir = args.out.expanduser().resolve()

    if not docs_dir.is_dir():
        print(f"[error] Source directory not found: {docs_dir}", file=sys.stderr)
        return 1

    pdf_paths = sorted(path for path in docs_dir.glob(args.pattern) if path.is_file())
    if args.files:
        requested = {name.lower() for name in args.files}
        pdf_paths = [
            path
            for path in pdf_paths
            if path.name.lower() in requested or path.stem.lower() in requested
        ]

    if not pdf_paths:
        print(f"[error] No matching documents found in {docs_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for index, pdf_path in enumerate(pdf_paths, start=1):
        output_path = output_dir / f"{pdf_path.stem}.md"
        if args.skip_existing and output_path.exists():
            print(f"[{index}/{len(pdf_paths)}] Skipping {pdf_path.name}")
            continue

        try:
            with pymupdf.open(pdf_path) as doc:
                mode = _detect_ocr_mode(doc, args.sample_pages, args.sample_dpi)

            ocr_function = _make_ocr_function(
                mode, args.force_tesseract, args.force_rapidocr
            )
            written_path, length = _convert_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                ocr_function=ocr_function,
                ocr_dpi=args.ocr_dpi,
                force_ocr=args.force_ocr,
                show_progress=args.progress,
            )
            print(
                f"[{index}/{len(pdf_paths)}] {pdf_path.name} -> "
                f"{written_path.name} ({length} chars, ocr={mode})"
            )
        except Exception as exc:
            failures += 1
            print(f"[error] {pdf_path.name}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
