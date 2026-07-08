```
workspace/
├── batch_extract.py               # Main execution script for batch inference
├── create_jsonl.py                # Script to prepare/format training data into JSONL
├── evaluate.py                    # Script to evaluate model performance
├── train_lora.py                  # Script to fine-tune the model using LoRA
│
├── data/                          # Dataset directory
│   ├── train/                     # Training data splits
│   ├── test/                      # Testing data splits
│   └── val/                       # Validation data splits
│       ├── GT/                    # Ground Truth files for validation
│       └── PDF/                   # Input directory (Source PDFs)
│
├── present_runs/                  # Inference outputs and logs
│   ├── predictions_epoch1/        # Inference outputs for Epoch 1
│   │   ├── train/                 # Predictions on training set
│   │   ├── test/                  # Predictions on test set
│   │   └── val/                   # Predictions on validation set
│   ├── ...                        # (Epoch 2 & 3 outputs)
│   └── predictions_epoch4/        # Inference outputs for Epoch 4
│       ├── train/
│       ├── test/
│       └── val/
│
└── qwen3b_lora/                   # Saved model checkpoints and training logs
    ├── adapter_epoch_1/           # LoRA adapter weights (Epoch 1)
    ├── ...                        # (Epoch 2 & 3 weights)
    ├── adapter_epoch_4/           # LoRA adapter weights (Epoch 4)
    │
    └── plots/                     # Training metrics and graphs
        ├── epoch_01/
        ├── ...
        └── epoch_04/
            ├── epoch_wall_time.png
            ├── grad_norm.png
            ├── loss_vs_epoch.png
            ├── loss_vs_step.png
            └── lr_schedule.png
```
