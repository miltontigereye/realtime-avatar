# Reference papers

The PDF files are the complete reading copies, including text, equations, tables, and figures. The `text/` directory provides layout-preserving text extraction for searching and text-only research tools; use the PDF for visual content and mathematical notation.

The local collection contains 18 papers (328 pages). Lossless PDF stream recompression and object packing reduced the PDFs from 171,952,018 to 169,414,128 bytes (1.48%). Images were not downsampled or re-encoded. Original PDFs were retained whenever recompression was larger or changed the checked stream inventory.

`MANIFEST.json` records original names and hashes, published PDF and text hashes, page counts, sizes, and validation results. Every published paper preserves the extracted text and page count; retained originals or matching stream inventories preserve content and images.

The original local PDFs are unchanged. `Omni-LiveAvatarpdf` is published as `Omni-LiveAvatar.pdf` for PDF recognition. Full-text extraction may not represent equations, reading order, or diagram labels faithfully; the PDFs preserve that information.

To reproduce into a new directory, install `pikepdf` and Poppler (`pdftotext`), then run:

```bash
python tools/export_papers.py /path/to/local/docs/papers /path/to/new/docs/papers
```
