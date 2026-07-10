import os
import json
from pdf2image import convert_from_path


FIELDS_TO_EXTRACT = {
    "invoice_number", "invoice_date", "vendor_name", "customer_name",
    "amount_before_vat", "vat_or_gst_amount", "amount_after_vat", "wht", "currency"
}

TRAIN_DIR = "/workspace/data/train"
VAL_DIR   = "/workspace/data/val"
TRAIN_OUTPUT = "/workspace/train_comp_data_final.jsonl"
VAL_OUTPUT   = "/workspace/val_comp_data_final.jsonl"
TRAIN_IMAGES_DIR = "/workspace/data/train/PDF"
VAL_IMAGES_DIR   = "/workspace/data/val/PDF"

DPI      = 150
MAX_PAGES = 4

PROMPT = """
Extract the following fields from the invoice. The invoice may span multiple pages.
The invoice may contain text in multiple languages — read and extract all text exactly as it appears using the Roman script present. Do not translate anything.

Fields to extract:
1. invoice_number
2. invoice_date
3. vendor_name
4. customer_name
5. amount_before_vat
6. vat_or_gst_amount
7. amount_after_vat
8. wht
9. currency

Return only a flat JSON dictionary matching this schema. Do not output markdown wrappers (like ```json), thinking process, or any other text. If a field is not found, set its value to null.

Format:
{
  "invoice_number": "553823",
  "invoice_date": "09/18/2015",
  "vendor_name": "ABC Solutions",
  "customer_name": "Zalora South East Asia",
  "amount_before_vat": "900.00",
  "vat_or_gst_amount": "78.12",
  "amount_after_vat": "978.12",
  "wht": null,
  "currency": "SGD"
}
"""

def convert_pdf_to_images(pdf_path, images_dir, stem, dpi=DPI, max_pages=MAX_PAGES):
    """Convert a PDF to PNG images (one per page), return list of image paths."""
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=max_pages)
    page_paths = []
    for i, page in enumerate(pages, 1):
        image_path = os.path.join(images_dir, f"{stem}_page{i}.png")
        if not os.path.exists(image_path):
            page.save(image_path, "PNG")
        page_paths.append(image_path)
    return page_paths


def build_target_json(gt_rows):
    output = {field: None for field in FIELDS_TO_EXTRACT}
    for row in gt_rows:
        field = row.get("field")
        if field in FIELDS_TO_EXTRACT:
            val = row.get("value")
            if val is None or str(val).strip() in ("", "null", "None"):
                val = None
            else:
                if isinstance(val, (int, float)):
                    val = str(val)
                else:
                    val = str(val).strip()
            output[field] = val
    return output

def create_jsonl(dataset_dir, images_dir, output_file):
    pdf_dir = os.path.join(dataset_dir, "PDF")
    gt_dir  = os.path.join(dataset_dir, "GT")

    os.makedirs(images_dir, exist_ok=True)

    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
    print(f"\nProcessing {len(pdf_files)} PDFs for {output_file}...")

    written = 0
    skipped = 0

    with open(output_file, "w", encoding="utf-8") as outfile:
        for i, pdf_file in enumerate(pdf_files, 1):
            stem    = os.path.splitext(pdf_file)[0]
            gt_path = os.path.join(gt_dir, stem + ".json")

            # Check GT exists
            if not os.path.exists(gt_path):
                print(f"  Missing GT: {stem}.json")
                skipped += 1
                continue

            # Load GT
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_raw = json.load(f)
            except Exception as e:
                print(f"  Could not read {stem}.json: {e}")
                skipped += 1
                continue

            # Convert PDF to images
            try:
                pdf_path   = os.path.join(pdf_dir, pdf_file)
                image_paths = convert_pdf_to_images(pdf_path, images_dir, stem)
            except Exception as e:
                print(f"  Failed to convert {pdf_file}: {e}")
                skipped += 1
                continue

            # Build target JSON from GT
            # GT may be a list directly or wrapped — handle both
            if isinstance(gt_raw, dict) and "metadata" in gt_raw:
                gt_rows = gt_raw["metadata"]
            elif isinstance(gt_raw, list):
                gt_rows = gt_raw
            else:
                gt_rows = [gt_raw]

            target_json = build_target_json(gt_rows)

            # ---- THIS IS WHERE THE EXAMPLE BLOCK GOES ----
            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            # One image entry per page
                            *[{"type": "image", "image": img_path} for img_path in image_paths],
                            {"type": "text", "text": PROMPT}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(target_json, ensure_ascii=False)
                            }
                        ]
                    }
                ]
            }
            # -----------------------------------------------

            outfile.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

            if i % 50 == 0:
                print(f"  {i}/{len(pdf_files)} done...")

    print(f"Created: {output_file}")
    print(f"  Written: {written} | Skipped: {skipped}")


if __name__ == "__main__":
    create_jsonl(TRAIN_DIR, TRAIN_IMAGES_DIR, TRAIN_OUTPUT)
    create_jsonl(VAL_DIR,   VAL_IMAGES_DIR,   VAL_OUTPUT)
