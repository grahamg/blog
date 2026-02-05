+++
date = '2026-02-05T00:05:51-06:00'
draft = false
title = 'Recovering Hidden PDF Attachments from the Epstein Document Release'
tags = ["AI", "Python3", "Claude Code", "Claude Opus"]
+++

Following a [challenge posed by Mahmoud Al-Qudsi](https://neosmart.net/blog/recreating-epstein-pdfs-from-raw-encoded-attachments/), I set out to build an automated pipeline for recovering base64-encoded email attachments buried inside the DoJ's Epstein document release. Here's what I found.

## The Problem

When the Department of Justice released thousands of documents related to Jeffrey Epstein, they made a peculiar choice: rather than preserving email attachments digitally, they printed the raw email source — including base64-encoded binary attachments — and then scanned those printouts as JPEG images embedded in PDFs.

The result: PDF files that look like someone printed out `cat email.eml` and ran it through a flatbed scanner. Pages and pages of tiny Courier New text containing base64-encoded data, now trapped as low-quality raster images.

## The Dataset

The files are organized across multiple datasets:

- **Dataset 9**: 6,067 PDFs (the largest collection)
- **Dataset 11**: 50 PDFs
- **Extracted volumes**: VOL00002 through VOL00012, each containing IMAGES directories with individual document PDFs

The target document identified by the blog post is **EFTA00400459**, located in `extracted/VOL00009/IMAGES/0092/`. It's a 76-page PDF containing an email between Boris Nikolic and one of Epstein's assistants, with an attached PDF invitation ("DBC12 One Page Invite with Reply.pdf") encoded as base64 across 75 pages.

## Building the Scanner

My first task was building a tool to automatically detect which pages across thousands of PDFs contain base64-encoded content. The naive approach — checking whether characters fall within the base64 alphabet `[A-Za-z0-9+/=]` — fails spectacularly.

### False Positive Problem

When I first scanned Dataset 11 (50 PDFs), I got 27 hits with scores above 0.90. Exciting — until I looked at the actual content. Every single one was regular email text, not base64 data. A Wizz Air flight itinerary, for example, scored 0.945 because dense English text without detected spaces is almost entirely composed of base64-valid characters.

The OCR at scan resolution (100 DPI) strips most punctuation and spaces, leaving walls of alphanumeric text that look superficially like base64.

### Entropy to the Rescue

The key discriminator turned out to be **Shannon entropy**. Real base64 encoding produces a near-uniform distribution over 64 characters, yielding high entropy (~5.5-6.0 bits per character). English text, even without spaces, has heavily skewed letter frequencies (lots of e/t/a/o, few z/q/x/j), producing lower entropy (~4.0-4.5 bits).

I combined this with common English word detection — if the OCR output contains "the", "and", "from", "your" etc., it's text, not base64. With both filters in place, the re-scan of Dataset 11 correctly returned zero hits: none of those PDFs contain actual base64 attachments.

## Finding the Target

The blog post identified EFTA00400459 as the specific document to target. I found it in the extracted volumes:

```
extracted/VOL00009/IMAGES/0092/EFTA00400459.pdf
```

76 pages, 11.2 MB. Page 1 is the email header and body. Pages 2-76 contain the base64-encoded PDF attachment.

## Three OCR Approaches

### Approach 1: Embedded Text Layer (pdftotext)

The scanned PDFs already have an embedded text layer from whatever OCR the DoJ used during processing. Extracting it is instant:

```bash
pdftotext EFTA00400459.pdf - | head
```

This gives us the base64 data directly, but with significant errors:

- **Result**: 3,843 lines OK, 998 lines failed (**79.4% accuracy**)
- Many invalid characters (commas, brackets, periods scattered throughout)
- The header `JVBERi0x` (= `%PDF-1`) was garbled

### Approach 2: Tesseract with Base64 Whitelist

I re-OCR'd the pages using Tesseract with a character whitelist that restricts output to only valid base64 characters:

```
--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=
```

Pre-processing pipeline:
1. Render at 300-400 DPI using `pdftoppm`
2. Convert to grayscale
3. Upscale 2x with nearest-neighbor interpolation (preserves sharp text edges)
4. Sharpen and binarize with a fixed threshold
5. Apply MinFilter to thicken thin Courier New strokes

The whitelist eliminates spurious commas and brackets, but it can't fix characters that are wrong *within* the base64 alphabet. When Tesseract reads `l` but the actual character is `1`, both are valid base64 — the whitelist doesn't help.

### Approach 3: AWS Textract (Blog Author's Data)

The blog author generously uploaded their AWS Textract OCR results as a ZIP file. Textract is a commercial OCR service that produced significantly better results:

- **Result**: 4,750 lines OK, 10 fixed, 64 lines failed (**98.5% accuracy**)
- Only 50 invalid characters across 362K of base64 text
- The PDF header decoded correctly: `%PDF-1.5`

## The Decode

Using the Textract data, I wrote a line-by-line decoder that:

1. Strips email quoting markers (`> `)
2. Removes EFTA page markers between pages
3. Filters invalid characters
4. Handles internal `=` signs (OCR errors — real base64 only has `=` padding at the very end)
5. Attempts decode per-line with fallback to padding correction

The moment of truth:

```
Total base64 chars: 361,865
Lines OK: 4,750
Lines fixed: 10
Lines failed: 64
First 20 bytes: b'%PDF-1.5\r%\xe2\xe3\xcf\xd3\r\n34 0'

*** FILE TYPE: PDF ***
Saved: EFTA00400459_DBC12_invite.pdf (271,388 bytes)
```

**The header is correct.** `%PDF-1.5` followed by the standard binary comment marker. The file size (271KB) is consistent with the MIME header's `size=276028` (allowing for the ~1.5% of corrupted lines).

## The Wall

Despite 98.5% line-level accuracy, the recovered PDF cannot be rendered. Running `qpdf --check` produces hundreds of structural errors:

```
WARNING: file is damaged
WARNING: can't find startxref
WARNING: Attempting to reconstruct cross-reference table
WARNING: unable to find trailer dictionary
```

The fundamental problem: **base64 is unforgiving**. Each base64 character encodes 6 bits. A single character error corrupts up to 3 bytes of the decoded output. With 64 failed lines scattered across the file, there are at least 64 regions of corrupted bytes — enough to break:

- Cross-reference tables (which store byte offsets — any shift breaks all references)
- Stream length declarations (wrong lengths cause parser errors)
- Flate-compressed content streams (a single wrong byte in a zlib stream corrupts everything after it)
- Dictionary key names (corrupted `/Filter` becomes gibberish)

## What I Learned

1. **Entropy is the best base64 detector.** Character alphabet membership is necessary but not sufficient. Shannon entropy cleanly separates base64 (~5.5+ bits) from English text (~4.5 bits).

2. **Character whitelisting helps but doesn't solve the core problem.** It eliminates invalid characters but can't fix wrong-within-alphabet errors. The Courier New confusions (1/l/I, 0/O, m/rn, 5/S, 8/B) all involve characters that are valid base64.

3. **Line-by-line decoding is essential.** Decoding the entire concatenated base64 as one blob means a single error can cascade. Line-by-line decoding isolates failures to individual 76-character lines (57 bytes each).

4. **Commercial OCR significantly outperforms open-source for this task.** Textract's 98.5% vs. the embedded text layer's 79.4% is the difference between a partially recognizable PDF and complete garbage.

5. **98.5% accuracy is not enough for binary reconstruction.** This is the fundamental insight. For text recovery, 98.5% would be excellent. For binary data where every byte matters, it's insufficient. You need effectively 100% accuracy, which OCR cannot provide at this scan quality.

## The Tool

The recovery pipeline script is `recover_attachments.py` with four modes (supplied below). Disclaimer: It was generated by Claude Opus v4.3, not hand written.

```python3
#!/usr/bin/env python3
"""
Recover base64-encoded email attachments from scanned Epstein document PDFs.

Scans PDFs for pages containing base64-encoded binary data (rendered as
Courier New text in scanned images), OCRs them with Tesseract, corrects
common recognition errors, and reconstructs the original files.

Usage:
    # Scan for base64 pages
    python3 recover_attachments.py scan --dir ./dataset9-pdfs/ --output manifest.json

    # Extract from a specific PDF + page range
    python3 recover_attachments.py extract --pdf file.pdf --pages 5-12 --output ./recovered/

    # Full auto pipeline
    python3 recover_attachments.py auto --dir ./dataset9-pdfs/ --output ./recovered/
"""

import argparse
import base64
import json
import math
import os
import re
import string
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageOps
import pytesseract

# Valid base64 alphabet
B64_CHARS = set(string.ascii_letters + string.digits + "+/=")

# Common Courier New OCR confusions: char -> list of likely intended chars
CONFUSION_MAP = {
    "1": ["l", "I"],
    "l": ["1", "I"],
    "I": ["l", "1"],
    "0": ["O", "o"],
    "O": ["0"],
    "o": ["0", "O"],
    "5": ["S", "s"],
    "S": ["5"],
    "s": ["5", "S"],
    "8": ["B"],
    "B": ["8"],
    "Z": ["2"],
    "2": ["Z"],
    "G": ["6"],
    "6": ["G"],
    "g": ["9"],
    "9": ["g", "q"],
    "q": ["9"],
    "D": ["0"],
    "U": ["V"],
    "V": ["U"],
    "m": ["rn"],
    "rn": ["m"],
}

# Tesseract config for base64 text
TESSERACT_B64_CONFIG = (
    "--psm 6 "
    "-c tessedit_char_whitelist="
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)

# Quick scan config (lower quality, faster)
TESSERACT_SCAN_CONFIG = "--psm 6"

# Magic bytes for file type detection
MAGIC_BYTES = [
    (b"%PDF", ".pdf"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"PK\x03\x04", ".zip"),
    (b"PK\x05\x06", ".zip"),
    (b"\x1f\x8b", ".gz"),
    (b"Rar!", ".rar"),
    (b"\xd0\xcf\x11\xe0", ".doc"),  # OLE2 (doc/xls/ppt)
    (b"\x50\x4b\x03\x04\x14\x00\x06\x00", ".docx"),  # OOXML
]


def detect_file_type(data: bytes) -> str:
    """Detect file type from magic bytes."""
    for magic, ext in MAGIC_BYTES:
        if data[:len(magic)] == magic:
            return ext
    return ".bin"


def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Pre-process a PIL image for optimal Tesseract OCR of base64 text."""
    # Convert to grayscale
    gray = img.convert("L")

    # Upscale 2x with nearest-neighbor (preserves sharp edges of text)
    w, h = gray.size
    upscaled = gray.resize((w * 2, h * 2), Image.NEAREST)

    # Apply a slight sharpen to enhance character edges
    sharpened = upscaled.filter(ImageFilter.SHARPEN)

    # Binarize with a threshold — simpler than adaptive but effective
    # with the Tesseract whitelist doing most of the heavy lifting
    threshold = 160
    binary = sharpened.point(lambda x: 255 if x > threshold else 0, "1")

    # Convert back to L mode for Tesseract
    binary = binary.convert("L")

    # Thicken strokes slightly via min filter (dilates dark pixels)
    thickened = binary.filter(ImageFilter.MinFilter(size=3))

    return thickened


def char_entropy(text: str) -> float:
    """Calculate Shannon entropy of character distribution."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def score_page_as_base64(text: str) -> tuple[float, int]:
    """Score how likely a page's text is base64 content.

    Returns (score, text_length) where score is 0.0-1.0.
    Uses multiple heuristics:
    - Character alphabet membership (must be mostly base64 chars)
    - Shannon entropy (base64 ~5.5-6.0 bits; English text ~4.0-4.5 bits)
    - Absence of common English words
    - Presence of base64-specific chars (+, /, =)
    """
    # Strip whitespace for analysis
    stripped = re.sub(r"\s+", "", text)
    if len(stripped) < 100:
        return 0.0, len(stripped)

    # Check 1: character alphabet (must be high)
    b64_count = sum(1 for c in stripped if c in B64_CHARS)
    alphabet_score = b64_count / len(stripped)
    if alphabet_score < 0.85:
        return 0.0, len(stripped)

    # Check 2: entropy of lowercase version (case-insensitive analysis)
    # Real base64 uses both cases uniformly -> high entropy
    # English text has skewed letter frequencies -> lower entropy
    entropy = char_entropy(stripped.lower())
    # base64 (lowercased) entropy is typically 4.5-5.0
    # English text (lowercased, no spaces) entropy is typically 3.5-4.2
    # We use the case-sensitive version for a better signal
    cs_entropy = char_entropy(stripped)
    # base64 case-sensitive entropy: ~5.5-6.0
    # English text case-sensitive: ~4.0-4.8

    # Check 3: common English words (lowercase the text and search)
    lower = stripped.lower()
    common_words = [
        "the", "and", "for", "that", "with", "this", "from", "your",
        "have", "are", "not", "but", "will", "can", "all", "been",
        "were", "has", "was", "may", "information", "please", "subject",
        "email", "date", "sent", "attachment", "message",
    ]
    word_hits = sum(1 for w in common_words if w in lower)

    # Check 4: presence of +, /, = (common in base64, rare in text)
    special_b64 = sum(1 for c in stripped if c in "+/=")
    special_ratio = special_b64 / len(stripped) if stripped else 0

    # Scoring:
    # - High alphabet score is necessary but not sufficient
    # - High entropy strongly suggests base64
    # - Many English word hits strongly suggest it's NOT base64
    # - Presence of +/= is a mild positive signal

    if word_hits >= 5:
        return 0.0, len(stripped)  # Definitely English text

    if cs_entropy < 4.8 and word_hits >= 2:
        return 0.0, len(stripped)  # Likely English text

    # Combine signals into final score
    # Start with alphabet score, adjust based on entropy
    score = alphabet_score

    # Penalize low entropy (suggests structured text, not random base64)
    if cs_entropy < 4.5:
        score *= 0.3
    elif cs_entropy < 5.0:
        score *= 0.6
    elif cs_entropy < 5.3:
        score *= 0.8

    # Penalize English word presence
    score *= max(0.0, 1.0 - word_hits * 0.15)

    # Small bonus for base64 special chars
    if special_ratio > 0.005:
        score = min(1.0, score * 1.05)

    return score, len(stripped)


def scan_pdf_for_base64(pdf_path: str, scan_dpi: int = 100) -> list[dict]:
    """Scan a PDF for pages containing base64-encoded content.

    Returns list of {page, score, text_length} for pages scoring above threshold.
    """
    results = []
    try:
        images = convert_from_path(pdf_path, dpi=scan_dpi)
    except Exception as e:
        print(f"  Error converting {pdf_path}: {e}", file=sys.stderr)
        return results

    for i, img in enumerate(images, 1):
        try:
            text = pytesseract.image_to_string(img, config=TESSERACT_SCAN_CONFIG)
            score, text_len = score_page_as_base64(text)
            if score > 0.85 and text_len > 500:
                results.append({
                    "page": i,
                    "score": round(score, 4),
                    "text_length": text_len,
                })
        except Exception as e:
            print(f"  Error OCR page {i} of {pdf_path}: {e}", file=sys.stderr)

    return results


def group_consecutive_pages(pages: list[dict]) -> list[dict]:
    """Group consecutive base64 pages into attachment ranges."""
    if not pages:
        return []

    page_nums = [p["page"] for p in pages]
    groups = []

    for _, g in groupby(enumerate(page_nums), lambda ix: ix[1] - ix[0]):
        group_pages = list(g)
        start = group_pages[0][1]
        end = group_pages[-1][1]
        avg_score = sum(
            p["score"] for p in pages
            if start <= p["page"] <= end
        ) / (end - start + 1)
        groups.append({
            "pages": f"{start}-{end}" if start != end else str(start),
            "page_start": start,
            "page_end": end,
            "num_pages": end - start + 1,
            "avg_score": round(avg_score, 4),
        })

    return groups


def extract_base64_from_pages(
    pdf_path: str,
    page_start: int,
    page_end: int,
    extract_dpi: int = 300,
) -> str:
    """Extract and OCR base64 text from specific pages of a PDF."""
    all_text = []

    images = convert_from_path(
        pdf_path,
        dpi=extract_dpi,
        first_page=page_start,
        last_page=page_end,
    )

    for i, img in enumerate(images):
        page_num = page_start + i
        print(f"  Processing page {page_num}...")

        # Pre-process for optimal OCR
        processed = preprocess_image_for_ocr(img)

        # OCR with base64 character whitelist
        text = pytesseract.image_to_string(
            processed,
            config=TESSERACT_B64_CONFIG,
        )

        # Strip whitespace
        clean = re.sub(r"\s+", "", text)
        all_text.append(clean)
        print(f"    Got {len(clean)} base64 characters")

    return "".join(all_text)


def try_decode_base64(text: str) -> tuple[bytes | None, str | None]:
    """Try to decode base64 text, returning (data, error_msg)."""
    # Ensure proper padding
    padding_needed = len(text) % 4
    if padding_needed:
        text += "=" * (4 - padding_needed)

    try:
        data = base64.b64decode(text, validate=True)
        return data, None
    except Exception as e:
        return None, str(e)


def attempt_error_correction(text: str, max_attempts: int = 1000) -> tuple[str, bool]:
    """Try to fix base64 decode errors using character confusion map.

    Returns (corrected_text, success).
    """
    # First try as-is
    data, err = try_decode_base64(text)
    if data is not None:
        return text, True

    print(f"  Initial decode failed: {err}")
    print(f"  Attempting error correction...")

    # Find invalid base64 characters first
    invalid_positions = []
    for i, c in enumerate(text):
        if c not in B64_CHARS:
            invalid_positions.append(i)

    if invalid_positions:
        print(f"  Found {len(invalid_positions)} invalid characters")
        # Replace invalid chars with best guesses
        text_list = list(text)
        for pos in invalid_positions:
            char = text_list[pos]
            if char in CONFUSION_MAP:
                text_list[pos] = CONFUSION_MAP[char][0]
            else:
                # Remove unknown characters
                text_list[pos] = ""
        text = "".join(text_list)

        data, err = try_decode_base64(text)
        if data is not None:
            return text, True

    # Try systematic single-character substitutions at positions near errors
    # Base64 decodes in groups of 4 chars, so find which group fails
    attempts = 0
    for chunk_start in range(0, len(text), 4):
        chunk = text[chunk_start:chunk_start + 4]
        if len(chunk) < 4:
            break

        # Try decoding just this chunk to find problematic groups
        try:
            base64.b64decode(chunk + "==", validate=True)
            continue  # This chunk is fine
        except Exception:
            pass

        # Try substitutions for each char in this chunk
        for offset in range(4):
            pos = chunk_start + offset
            if pos >= len(text):
                break
            original = text[pos]
            candidates = CONFUSION_MAP.get(original, [])
            for replacement in candidates:
                attempts += 1
                if attempts > max_attempts:
                    print(f"  Exhausted {max_attempts} correction attempts")
                    return text, False

                corrected = text[:pos] + replacement + text[pos + 1:]
                data, err = try_decode_base64(corrected)
                if data is not None:
                    print(f"  Fixed: position {pos}, '{original}' -> '{replacement}'")
                    return corrected, True

    print(f"  Could not fully correct base64 ({attempts} attempts)")
    return text, False


def reconstruct_file(
    pdf_path: str,
    page_start: int,
    page_end: int,
    output_dir: str,
    extract_dpi: int = 300,
) -> dict:
    """Full pipeline: extract, OCR, correct, decode, and save."""
    pdf_name = Path(pdf_path).stem
    result = {
        "source_pdf": pdf_path,
        "pages": f"{page_start}-{page_end}",
        "status": "failed",
    }

    # Extract base64 text
    print(f"\nExtracting base64 from {pdf_name} pages {page_start}-{page_end}...")
    raw_text = extract_base64_from_pages(pdf_path, page_start, page_end, extract_dpi)
    result["raw_b64_length"] = len(raw_text)

    if len(raw_text) < 10:
        result["error"] = "No base64 text extracted"
        return result

    # Save raw OCR output for debugging
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, f"{pdf_name}_p{page_start}-{page_end}_raw.txt")
    with open(raw_path, "w") as f:
        f.write(raw_text)
    result["raw_text_file"] = raw_path

    # Attempt error correction and decode
    corrected, success = attempt_error_correction(raw_text)

    if not success:
        # Save what we have even if decode fails
        corrected_path = os.path.join(
            output_dir, f"{pdf_name}_p{page_start}-{page_end}_corrected.txt"
        )
        with open(corrected_path, "w") as f:
            f.write(corrected)
        result["corrected_text_file"] = corrected_path
        result["error"] = "Base64 decode failed after correction attempts"

        # Try partial decode - decode what we can
        partial_decode_result = try_partial_decode(corrected, output_dir, pdf_name, page_start, page_end)
        if partial_decode_result:
            result.update(partial_decode_result)
            result["status"] = "partial"
        return result

    # Decode succeeded
    data, _ = try_decode_base64(corrected)
    ext = detect_file_type(data)
    output_file = os.path.join(
        output_dir, f"{pdf_name}_p{page_start}-{page_end}_recovered{ext}"
    )
    with open(output_file, "wb") as f:
        f.write(data)

    result["status"] = "success"
    result["output_file"] = output_file
    result["file_type"] = ext
    result["file_size"] = len(data)
    print(f"  Recovered {ext} file: {output_file} ({len(data)} bytes)")

    return result


def try_partial_decode(
    text: str, output_dir: str, pdf_name: str, page_start: int, page_end: int
) -> dict | None:
    """Try to decode base64 in chunks, salvaging what we can."""
    # Try decoding larger chunks - some may be valid
    chunk_size = 4 * 256  # 1024 chars = 768 bytes per chunk
    decoded_chunks = []
    error_chunks = 0

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        # Ensure padding
        padding = len(chunk) % 4
        if padding:
            chunk += "=" * (4 - padding)
        try:
            decoded_chunks.append(base64.b64decode(chunk, validate=True))
        except Exception:
            error_chunks += 1
            decoded_chunks.append(b"\x00" * (len(chunk) * 3 // 4))  # placeholder

    if not decoded_chunks:
        return None

    data = b"".join(decoded_chunks)
    ext = detect_file_type(data)
    output_file = os.path.join(
        output_dir, f"{pdf_name}_p{page_start}-{page_end}_partial{ext}"
    )
    with open(output_file, "wb") as f:
        f.write(data)

    total_chunks = len(text) // chunk_size + 1
    print(
        f"  Partial decode: {total_chunks - error_chunks}/{total_chunks} chunks OK, "
        f"saved as {output_file}"
    )

    return {
        "output_file": output_file,
        "file_type": ext,
        "file_size": len(data),
        "chunks_ok": total_chunks - error_chunks,
        "chunks_total": total_chunks,
    }


# ─── Multiprocessing helpers (must be module-level for pickling) ──────────────

_scan_dpi = 100  # set before launching pool


def _scan_one_pdf(pdf_path):
    return str(pdf_path), scan_pdf_for_base64(str(pdf_path), scan_dpi=_scan_dpi)


# ─── CLI Commands ────────────────────────────────────────────────────────────


def cmd_scan(args):
    """Scan a directory of PDFs for base64-encoded pages."""
    global _scan_dpi
    pdf_dir = Path(args.dir).expanduser()
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    print(f"Scanning {len(pdfs)} PDFs in {pdf_dir}...")
    manifest = []

    workers = min(args.workers, len(pdfs))
    _scan_dpi = args.scan_dpi

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_one_pdf, p): p for p in pdfs}
        done = 0
        for future in as_completed(futures):
            done += 1
            pdf_path, pages = future.result()
            pdf_name = Path(pdf_path).name
            if pages:
                groups = group_consecutive_pages(pages)
                entry = {
                    "pdf": pdf_path,
                    "attachments": groups,
                }
                manifest.append(entry)
                for g in groups:
                    print(
                        f"  [{done}/{len(pdfs)}] {pdf_name}: "
                        f"pages {g['pages']} (score: {g['avg_score']})"
                    )
            else:
                if done % 50 == 0 or done == len(pdfs):
                    print(f"  [{done}/{len(pdfs)}] scanned...", end="\r")

    print(f"\nFound {len(manifest)} PDFs with base64 content")

    # Save manifest
    output = args.output or "manifest.json"
    with open(output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {output}")


def cmd_extract(args):
    """Extract and reconstruct attachment from a specific PDF + page range."""
    pdf_path = str(Path(args.pdf).expanduser())
    output_dir = args.output or "./recovered"

    # Parse page range
    if "-" in args.pages:
        start, end = args.pages.split("-")
        page_start, page_end = int(start), int(end)
    else:
        page_start = page_end = int(args.pages)

    result = reconstruct_file(pdf_path, page_start, page_end, output_dir, args.extract_dpi)
    print(f"\nResult: {json.dumps(result, indent=2)}")


def extract_b64_from_text_layer(pdf_path: str, output_dir: str) -> dict:
    """Extract base64 from the PDF's embedded text layer using pdftotext.

    This is faster and often more accurate than re-OCRing, since the text
    layer was typically created by commercial OCR software.
    """
    import subprocess

    pdf_name = Path(pdf_path).stem
    result = {"source_pdf": pdf_path, "method": "textlayer", "status": "failed"}

    # Extract text with pdftotext
    try:
        text = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception as e:
        result["error"] = f"pdftotext failed: {e}"
        return result

    # Find base64 sections (between Content-Transfer-Encoding: base64 and MIME boundary)
    sections = re.split(r"Content-Transfer-Encoding:\s*base64", text)
    if len(sections) < 2:
        result["error"] = "No base64 content found in text layer"
        return result

    os.makedirs(output_dir, exist_ok=True)
    attachments = []

    for idx, section in enumerate(sections[1:], 1):
        # Find end of base64 (MIME boundary or end of document)
        end_match = re.search(r"--[-\w]+--|_\d{3}_", section)
        if end_match:
            b64_text = section[:end_match.start()]
        else:
            b64_text = section

        # Clean up: remove > email quoting, EFTA markers, whitespace
        lines = b64_text.split("\n")
        clean_lines = []
        for line in lines:
            line = line.strip()
            if re.match(r"^EFTA\d+$", line):
                continue
            if not line or line == "•":
                continue
            line = line.replace("\x0c", "")
            line = re.sub(r"^>\s*", "", line)
            if line:
                clean_lines.append(line)

        # Decode line by line for better error handling
        valid_b64 = set(string.ascii_letters + string.digits + "+/=")
        decoded_parts = []
        ok = 0
        fixed = 0
        failed = 0

        for line in clean_lines:
            clean = "".join(c for c in line if c in valid_b64)
            if not clean:
                continue

            # Try strict decode
            try:
                decoded_parts.append(base64.b64decode(clean, validate=True))
                ok += 1
                continue
            except Exception:
                pass

            # Handle internal = signs
            if "=" in clean and not clean.endswith("="):
                stripped = clean.replace("=", "")
                pad = len(stripped) % 4
                if pad:
                    stripped += "=" * (4 - pad)
                try:
                    decoded_parts.append(base64.b64decode(stripped, validate=True))
                    fixed += 1
                    continue
                except Exception:
                    pass

            # Try with padding fix
            pad = len(clean) % 4
            padded = clean + "=" * (4 - pad) if pad else clean
            try:
                decoded_parts.append(base64.b64decode(padded, validate=True))
                ok += 1
            except Exception:
                # Insert placeholder bytes
                decoded_parts.append(b"\x00" * (len(clean) * 3 // 4))
                failed += 1

        if not decoded_parts:
            continue

        data = b"".join(decoded_parts)
        ext = detect_file_type(data)
        suffix = f"_att{idx}" if len(sections) > 2 else ""
        out_file = os.path.join(output_dir, f"{pdf_name}{suffix}_recovered{ext}")
        with open(out_file, "wb") as f:
            f.write(data)

        att_result = {
            "output_file": out_file,
            "file_type": ext,
            "file_size": len(data),
            "lines_ok": ok,
            "lines_fixed": fixed,
            "lines_failed": failed,
            "accuracy": f"{(ok + fixed) / (ok + fixed + failed) * 100:.1f}%"
            if (ok + fixed + failed) > 0 else "N/A",
        }
        attachments.append(att_result)
        print(f"  Attachment {idx}: {ext} ({len(data)} bytes) "
              f"[{ok} OK, {fixed} fixed, {failed} failed lines]")

    if attachments:
        result["status"] = "success" if all(a["lines_failed"] == 0 for a in attachments) else "partial"
        result["attachments"] = attachments
    else:
        result["error"] = "No decodable base64 content found"

    return result


def cmd_textlayer(args):
    """Extract base64 attachment from PDF's embedded text layer."""
    pdf_path = str(Path(args.pdf).expanduser())
    output_dir = args.output or "./recovered"
    result = extract_b64_from_text_layer(pdf_path, output_dir)
    print(f"\nResult: {json.dumps(result, indent=2)}")


def cmd_auto(args):
    """Full pipeline: scan for base64 pages then extract all found attachments."""
    pdf_dir = Path(args.dir).expanduser()
    output_dir = args.output or "./recovered"
    pdfs = sorted(pdf_dir.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    # Phase 1: Scan
    global _scan_dpi
    print(f"=== Phase 1: Scanning {len(pdfs)} PDFs ===")
    manifest = []

    workers = min(args.workers, len(pdfs))
    _scan_dpi = args.scan_dpi

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_one_pdf, p): p for p in pdfs}
        done = 0
        for future in as_completed(futures):
            done += 1
            pdf_path, pages = future.result()
            if pages:
                groups = group_consecutive_pages(pages)
                manifest.append({"pdf": pdf_path, "attachments": groups})
                for g in groups:
                    print(
                        f"  [{done}/{len(pdfs)}] {Path(pdf_path).name}: "
                        f"pages {g['pages']} (score: {g['avg_score']})"
                    )
            elif done % 50 == 0:
                print(f"  [{done}/{len(pdfs)}] scanned...", end="\r")

    total_attachments = sum(len(e["attachments"]) for e in manifest)
    print(f"\nFound {total_attachments} potential attachments in {len(manifest)} PDFs")

    if not manifest:
        print("No base64 content found.")
        return

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Phase 2: Extract
    print(f"\n=== Phase 2: Extracting {total_attachments} attachments ===")
    results = []

    for entry in manifest:
        for att in entry["attachments"]:
            result = reconstruct_file(
                entry["pdf"],
                att["page_start"],
                att["page_end"],
                output_dir,
                args.extract_dpi,
            )
            results.append(result)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    partial = sum(1 for r in results if r["status"] == "partial")
    failed = sum(1 for r in results if r["status"] == "failed")

    print(f"\n=== Results ===")
    print(f"  Success: {success}")
    print(f"  Partial: {partial}")
    print(f"  Failed:  {failed}")

    # Save report
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recover base64-encoded attachments from scanned Epstein PDFs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan PDFs for base64 pages")
    scan_parser.add_argument("--dir", required=True, help="Directory of PDFs to scan")
    scan_parser.add_argument("--output", default="manifest.json", help="Output manifest path")
    scan_parser.add_argument("--scan-dpi", type=int, default=100, help="DPI for scanning (default: 100)")
    scan_parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract attachment from PDF")
    extract_parser.add_argument("--pdf", required=True, help="PDF file path")
    extract_parser.add_argument("--pages", required=True, help="Page range (e.g., 5-12 or 5)")
    extract_parser.add_argument("--output", default="./recovered", help="Output directory")
    extract_parser.add_argument("--extract-dpi", type=int, default=300, help="DPI for extraction (default: 300)")

    # Textlayer command
    tl_parser = subparsers.add_parser("textlayer", help="Extract from PDF text layer (no OCR)")
    tl_parser.add_argument("--pdf", required=True, help="PDF file path")
    tl_parser.add_argument("--output", default="./recovered", help="Output directory")

    # Auto command
    auto_parser = subparsers.add_parser("auto", help="Full pipeline: scan then extract")
    auto_parser.add_argument("--dir", required=True, help="Directory of PDFs to scan")
    auto_parser.add_argument("--output", default="./recovered", help="Output directory")
    auto_parser.add_argument("--scan-dpi", type=int, default=100, help="DPI for scanning (default: 100)")
    auto_parser.add_argument("--extract-dpi", type=int, default=300, help="DPI for extraction (default: 300)")
    auto_parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "textlayer":
        cmd_textlayer(args)
    elif args.command == "auto":
        cmd_auto(args)


if __name__ == "__main__":
    main()
```

## More Usage Examples

```bash
# Scan a directory for PDFs containing base64 pages
python3 recover_attachments.py scan --dir ./pdfs/ --output manifest.json

# Re-OCR specific pages with Tesseract + whitelist
python3 recover_attachments.py extract --pdf file.pdf --pages 2-76

# Extract from embedded text layer (fast, no OCR)
python3 recover_attachments.py textlayer --pdf file.pdf

# Full auto pipeline
python3 recover_attachments.py auto --dir ./pdfs/
```

## Next Steps

The remaining challenge is closing that last 1.5% gap. Potential approaches:

- **Multi-engine consensus**: Run Tesseract, Textract, and the embedded text layer independently, then vote per-character. Where two of three agree, use that character.
- **Claude Vision**: Use a multimodal LLM to read the base64 text from page images. LLMs may handle the ambiguous Courier New characters better than traditional OCR.
- **PDF-aware correction**: Use knowledge of PDF syntax to validate corrections at structural boundaries — dictionary keys, stream markers, and cross-reference entries have predictable patterns.
- **Scan the full dataset**: Run the entropy-based scanner across all 6,067 Dataset 9 PDFs and the extracted volumes to find other base64 attachments. Some may be simpler files (images, small documents) that are more tolerant of byte-level errors.
