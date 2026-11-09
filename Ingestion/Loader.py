"""Convert PDFs in Docs/ to Markdown in Ingestion/MID_ingest/.

Conversion is decided page by page:
- pages with a real text layer are extracted with PyMuPDF4LLM;
- text-less pages are OCR'd with RapidOCR after preprocessing;
- digital Arabic PDFs are extracted with arabicpdf (Qalam).

OCR uses RapidOCR with a Latin or Arabic recognition model. Arabic pages can
also be recognized with Tesseract (Arabic and French) using the bundled
tessdata directory. Scanned pages are preprocessed with grayscale, cropping,
denoising, and contrast normalization before recognition.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pymupdf4llm
import pytesseract
from arabicpdf import to_text as arabic_to_text
from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

INGESTION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INGESTION_DIR.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "Docs"
DEFAULT_OUTPUT_DIR = INGESTION_DIR / "MID_ingest"

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u024F]")

MIN_PAGE_TEXT_CHARS = 20
MIN_TEXT_FOR_LANG = 200
ARABIC_RATIO = 0.05
OCR_DPI = 450

# The larger tessdata_best model improves Arabic recognition on CPU at the
# cost of slower OCR. It contains the bundled Arabic model used below.
TESSDATA_DIR = INGESTION_DIR / "tessdata_best"
TESSERACT_LANGS = {"arabic": "ara+fra", "latin": "fra"}
OCR_ENGINE_CHOICES = ("auto", "rapidocr", "tesseract")
DEFAULT_OCR_ENGINE = "auto"

_OCR_ENGINES: dict[str, RapidOCR] = {}


def _doc_text_stats(doc: pymupdf.Document) -> tuple[int, int]:
    arabic = 0
    visible = 0
    for page in doc:
        text = page.get_text("text")
        arabic += len(ARABIC_RE.findall(text))
        visible += len(re.sub(r"\s+", "", text))
    return arabic, visible


def _page_is_blank(page: pymupdf.Page) -> bool:
    if page.get_text("text").strip():
        return False
    if page.get_images(full=True):
        return False
    try:
        return not page.get_drawings()
    except Exception:
        return True


def _get_ocr_engine(language: str) -> RapidOCR:
    if language not in _OCR_ENGINES:
        rec_lang = LangRec.ARABIC if language == "arabic" else LangRec.LATIN
        _OCR_ENGINES[language] = RapidOCR(
            params={
                "Det.lang_type": LangDet.MULTI,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Rec.lang_type": rec_lang,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
        )
    return _OCR_ENGINES[language]


def _tesseract_available() -> bool:
    cmd = pytesseract.pytesseract.tesseract_cmd
    return bool(cmd) and shutil.which(cmd) is not None


def _ocr_tesseract(image: np.ndarray, language: str) -> str:
    if not _tesseract_available():
        raise RuntimeError(
            "Tesseract executable not found. Install it (e.g. 'tesseract-ocr') "
            "or set pytesseract.pytesseract.tesseract_cmd."
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    config = f'--tessdata-dir "{TESSDATA_DIR}" --psm 3'
    if language == "arabic":
        config += " -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(
        gray,
        lang=TESSERACT_LANGS.get(language, "eng"),
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for index, value in enumerate(data["text"]):
        word = value.strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1
        if confidence >= 0:
            confidences.append(confidence)
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        lines.setdefault(key, []).append(word)

    if confidences and sum(confidences) / len(confidences) < 40:
        print(
            "[warning] Low average Tesseract confidence: "
            f"{sum(confidences) / len(confidences):.1f}",
            file=sys.stderr,
        )

    return "\n".join(" ".join(words) for words in lines.values()).strip()


def _resolve_ocr_engine(engine: str, language: str) -> str:
    if engine != "auto":
        return engine
    if language == "arabic" and _tesseract_available():
        return "tesseract"
    return "rapidocr"


def _render_gray(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3
    )
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _crop_borders(
    gray: np.ndarray, pad_ratio: float = 0.01, min_keep: float = 0.25
) -> np.ndarray:
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return gray

    x, y, w, h = cv2.boundingRect(coords)
    height, width = gray.shape
    if w < min_keep * width or h < min_keep * height:
        return gray

    pad = int(pad_ratio * max(height, width))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)
    return gray[y0:y1, x0:x1]


def _quadrant_means(gray: np.ndarray) -> list[float]:
    height, width = gray.shape
    quadrants = [
        gray[: height // 2, : width // 2],
        gray[: height // 2, width // 2 :],
        gray[height // 2 :, : width // 2],
        gray[height // 2 :, width // 2 :],
    ]
    return [float(q.mean()) for q in quadrants if q.size]


def _illumination_uneven(gray: np.ndarray) -> bool:
    means = _quadrant_means(gray)
    return bool(means) and (max(means) - min(means) > 40)


def _denoise(gray: np.ndarray) -> np.ndarray:
    median = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, median)
    background = median > 200
    count = int(np.count_nonzero(background))
    if count == 0:
        return gray
    noise_ratio = np.count_nonzero((diff > 40) & background) / count
    return median if noise_ratio > 0.002 else gray


def _normalize_contrast(gray: np.ndarray) -> np.ndarray:
    if not _illumination_uneven(gray):
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _preprocess_page(page: pymupdf.Page, dpi: int) -> np.ndarray:
    gray = _render_gray(page, dpi)
    gray = _crop_borders(gray)
    gray = _denoise(gray)
    gray = _normalize_contrast(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _ocr_image(
    image: np.ndarray, language: str, engine: str = DEFAULT_OCR_ENGINE
) -> str:
    if _resolve_ocr_engine(engine, language) == "tesseract":
        return _ocr_tesseract(image, language)

    result = _get_ocr_engine(language)(image, use_cls=True)
    texts = getattr(result, "txts", None) or []
    return "\n".join(texts).strip()


def _detect_scanned_language(
    doc: pymupdf.Document,
    dpi: int,
    engine: str = DEFAULT_OCR_ENGINE,
    max_pages: int = 3,
) -> str:
    probe = "rapidocr" if engine == "auto" else engine
    latin_total = 0
    arabic_total = 0
    sampled = 0
    for page in doc:
        if _page_is_blank(page):
            continue
        image = _preprocess_page(page, dpi)
        latin_total += len(LATIN_RE.findall(_ocr_image(image, "latin", probe)))
        arabic_total += len(ARABIC_RE.findall(_ocr_image(image, "arabic", probe)))
        sampled += 1
        if sampled >= max_pages or latin_total + arabic_total >= 200:
            break
    return "arabic" if arabic_total > latin_total else "latin"


def _resolve_language(
    doc: pymupdf.Document,
    arabic: int,
    visible: int,
    dpi: int,
    engine: str = DEFAULT_OCR_ENGINE,
) -> str:
    if visible >= MIN_TEXT_FOR_LANG:
        return "arabic" if arabic / visible >= ARABIC_RATIO else "latin"
    return _detect_scanned_language(doc, dpi, engine)


def _convert_pages(
    doc: pymupdf.Document,
    pdf_path: Path,
    language: str,
    dpi: int,
    engine: str = DEFAULT_OCR_ENGINE,
) -> str:
    text_pages: list[int] = []
    ocr_pages: list[int] = []
    for index, page in enumerate(doc):
        text = page.get_text("text").strip()
        if len(text) >= MIN_PAGE_TEXT_CHARS:
            text_pages.append(index)
        elif _page_is_blank(page):
            continue
        else:
            ocr_pages.append(index)

    chunks: dict[int, str] = {}

    if text_pages:
        page_chunks = pymupdf4llm.to_markdown(
            str(pdf_path),
            pages=text_pages,
            page_chunks=True,
            write_images=False,
            embed_images=False,
        )
        for chunk in page_chunks:
            page_number = int(chunk["metadata"]["page_number"]) - 1
            text = chunk.get("text", "").strip()
            if text:
                chunks[page_number] = text

    for index in ocr_pages:
        image = _preprocess_page(doc[index], dpi)
        text = _ocr_image(image, language, engine)
        if text:
            chunks[index] = text

    return "\n\n".join(
        f"## Page {index + 1}\n\n{chunks.get(index, '')}".rstrip()
        for index in range(doc.page_count)
    ).strip()


def convert_pdf(
    pdf_path: Path,
    ocr_dpi: int = OCR_DPI,
    ocr_engine: str = DEFAULT_OCR_ENGINE,
) -> str:
    with pymupdf.open(pdf_path) as doc:
        if doc.page_count == 0:
            return ""

        arabic, visible = _doc_text_stats(doc)
        if visible >= MIN_TEXT_FOR_LANG and arabic / visible >= ARABIC_RATIO:
            return arabic_to_text(pdf_path)

        language = _resolve_language(doc, arabic, visible, ocr_dpi, ocr_engine)
        return _convert_pages(doc, pdf_path, language, ocr_dpi, ocr_engine)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs from Docs/ to Markdown (Arabic-aware)."
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
        default=OCR_DPI,
        help=f"DPI used to render OCR pages (default: {OCR_DPI})",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=OCR_ENGINE_CHOICES,
        default=DEFAULT_OCR_ENGINE,
        help=(
            "OCR backend: 'auto' uses Tesseract for Arabic when available "
            "and RapidOCR otherwise (default: auto)"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents whose Markdown output already exists",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first conversion error",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

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
            markdown = convert_pdf(pdf_path, args.ocr_dpi, args.ocr_engine)
            output_path.write_text(markdown, encoding="utf-8")
            print(
                f"[{index}/{len(pdf_paths)}] {pdf_path.name} -> "
                f"{output_path.name} ({len(markdown)} chars)"
            )
        except Exception as exc:
            failures += 1
            print(f"[error] {pdf_path.name}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
