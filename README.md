# Project Structure

```text
VLM-Finetuning/
├── batch_extract.py               # Main execution script for batch inference
├── create_jsonl.py                # Script to prepare/format training data into JSONL
├── train_lora.py                  # Script to fine-tune the model using LoRA
│
├── data/                          # Dataset directory
│   ├── train/                     # Training data splits
│   ├── test/                      # Testing data splits
│   └── val/                       # Validation data splits
│       ├── GT/                    # Ground Truth files for validation
│       └── PDF/                   # Input directory (Source PDFs)
│           ├── invoice_1.pdf
│           └── invoice_2.pdf
│
├── present_runs/                  # Inference outputs and logs
│   └── predictions_epoch2_val/    # Output directory (Extracted data)
│       ├── invoice_1.json
│       └── invoice_2.json
│
└── qwen3b_lora/                   # Saved model checkpoints
    └── adapter_epoch_2/           # Fine-tuned LoRA adapter weights
