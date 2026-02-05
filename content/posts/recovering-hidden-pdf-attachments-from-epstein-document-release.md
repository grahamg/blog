+++
date = '2026-02-05T00:05:51-06:00'
draft = false
title = 'Recovering Hidden PDF Attachments from the Epstein Document Release'
tags = ["AI", "Development", "Claude Code", "Claude Opus"]
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

The recovery pipeline is at `~/recover_attachments.py` with four modes:

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
