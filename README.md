# VLM-Finetuning

Project structure:

/workspace/
├── batch_extract.py                       <-- Main execution script
├── create_jsonl.py 
├── train_lora.py
│
├── data/
│   ├── train/
│   ├── test/
│   └── val/
│       ├── GT/
│       └── PDF/                           <-- Input directory (Source PDFs)
│           ├── invoice_1.pdf
│           └── invoice_2.pdf
│
├── present_runs/
│   └── predictions_epoch2_val/            <-- Output directory (Extracted data)
│       ├── invoice_1.json
│       └── invoice_2.json
│
└── qwen3b_lora/
    └── adapter_epoch_2/                   <-- Fine-tuned LoRA adapter weights
