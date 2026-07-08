import os
import json
import torch
import tempfile
from pdf2image import convert_from_path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


# ==========================================
PDF_FOLDER  = "/workspace/data/val/PDF"
OUTPUT_FOLDER = "/workspace/present_runs/predictions_epoch2_val"
BASE_MODEL   = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
ADAPTER_PATH = "/workspace/qwen3b_lora/adapter_epoch_2"
BATCH_SIZE  = 20
DPI         = 150
MAX_PAGES   = 3   # cap for very long invoices to stay within context window
# ==========================================

PROMPT = PROMPT = """
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

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading base model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
processor.chat_template = processor.tokenizer.chat_template
processor.tokenizer.padding_side = "left"
print("Model loaded.")

def pdf_to_page_paths(pdf_path, tmpdir, dpi=DPI, max_pages=MAX_PAGES):
    """Convert PDF pages to PNG files in tmpdir, return list of file paths."""
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=max_pages)
    page_paths = []
    for i, page in enumerate(pages, 1):
        p = os.path.join(tmpdir, f"page_{i}.png")
        page.save(p, "PNG")
        page_paths.append(p)
    return page_paths

def build_messages(page_paths):
    """Build the message dict for a single invoice (multiple pages = multiple images)."""
    return [
        {
            "role": "user",
            "content": [
                # One image entry per page — Qwen2.5-VL handles multiple images natively
                *[{"type": "image", "image": p} for p in page_paths],
                {"type": "text", "text": PROMPT}
            ]
        }
    ]

# Collect all PDF files
pdf_files = sorted([f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")])
print(f"Found {len(pdf_files)} PDFs, processing in batches of {BATCH_SIZE}")

# Process in batches
for batch_start in range(0, len(pdf_files), BATCH_SIZE):
    batch_files = pdf_files[batch_start: batch_start + BATCH_SIZE]
    batch_end   = batch_start + len(batch_files)
    print(f"\n[{batch_start + 1}-{batch_end}/{len(pdf_files)}] Processing: {batch_files}")

    # Use a single tempdir for the whole batch, cleaned up after each batch
    with tempfile.TemporaryDirectory() as tmpdir:

        batch_messages    = []
        batch_valid_files = []

        for pdf_file in batch_files:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            try:
                # Each PDF gets its own subfolder inside tmpdir to avoid filename collisions
                pdf_tmpdir = os.path.join(tmpdir, os.path.splitext(pdf_file)[0])
                os.makedirs(pdf_tmpdir, exist_ok=True)

                page_paths = pdf_to_page_paths(pdf_path, pdf_tmpdir)
                msgs = build_messages(page_paths)
                batch_messages.append(msgs)
                batch_valid_files.append(pdf_file)
            except Exception as e:
                print(f"  Failed to process {pdf_file}: {e}")

        if not batch_messages:
            continue

        # Build text inputs for each item in the batch
        batch_texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in batch_messages
        ]

        # Collect all image inputs across all items in the batch
        batch_image_inputs = []
        for msgs in batch_messages:
            img_inputs, _ = process_vision_info(msgs)
            batch_image_inputs.extend(img_inputs)

        inputs = processor(
            text=batch_texts,
            images=batch_image_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        responses = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Save each response
        for pdf_file, response in zip(batch_valid_files, responses):
            output_file = os.path.join(
                OUTPUT_FOLDER,
                os.path.splitext(pdf_file)[0] + ".json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"  Saved -> {output_file}")

print("\nFinished.")

