import os
os.environ["PYTORCH_JIT"] = "0"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
x = torch.randint(1, 10, (4, 4), device="cuda")
print(x.prod())
orig_prod = torch.prod
def debug_prod(*args, **kwargs):
    import traceback
    print("\n===== TORCH.PROD CALLED =====")
    traceback.print_stack(limit=20)
    return orig_prod(*args, **kwargs)
torch.prod = debug_prod

import json
import csv
import time
import math
from pathlib import Path
from datasets import load_dataset
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import AutoProcessor, TrainerCallback, TrainerState, TrainerControl

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==========================================
MODEL_NAME = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
OUTPUT_DIR = "/workspace/qwen3b_lora"
NUM_EPOCHS = 11

# To resume from a previous epoch adapter, set the path below (e.g. "/workspace/qwen3b_lora/adapter_epoch_4").
# If starting from scratch, set to None.
RESUME_ADAPTER_PATH = "/workspace/qwen3b_lora/adapter_epoch_6"
# ==========================================

PLOTS_ROOT = Path(OUTPUT_DIR) / "plots"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
LOG_CSV    = Path(OUTPUT_DIR) / "training_log.csv"

# Steps per epoch calculation: 960 train size / (1 batch size * 8 grad accum) = 120 steps
STEPS_PER_EPOCH = 120

START_EPOCH = 0
if RESUME_ADAPTER_PATH:
    import re
    match = re.search(r"adapter_epoch_(\d+)", RESUME_ADAPTER_PATH)
    if match:
        START_EPOCH = int(match.group(1))
    print(f"Resuming training from adapter path: {RESUME_ADAPTER_PATH} (starting after epoch {START_EPOCH})")
else:
    print("Starting training from scratch.")


# ------------------------------------------------------------------
# Callback: per-epoch adapter save + live chart refresh
# ------------------------------------------------------------------
class EpochMonitorCallback(TrainerCallback):
    """
    Interval: once per epoch (at epoch end).

    After every epoch:
      1. Save LoRA adapter as  adapter_epoch_N/
      2. Rewrite training_log.csv with every logged entry so far
      3. Save 5 charts to  plots/epoch_N/   (permanent per-epoch snapshot)
                       and plots/latest/    (always the most recent — check this when you log in)
    """

    def __init__(self, start_epoch: int = 0):
        self.start_epoch = start_epoch
        self._epoch_start_time: float = 0.0
        self._epoch_times: list[float] = []

        # running buffers populated from trainer.state.log_history
        self.step_list:    list[int]   = []
        self.train_loss:   list[float] = []
        self.lr_list:      list[float] = []
        self.grad_norm:    list[float] = []
        self.val_steps:    list[int]   = []
        self.val_loss:     list[float] = []

        # one record per epoch for epoch-level overview
        self.epoch_nums:       list[int]   = []
        self.epoch_train_loss: list[float] = []
        self.epoch_val_loss:   list[float] = []
        self.epoch_wall_time:  list[float] = []

        # Load previous history if resuming
        self.old_csv_rows = []
        if self.start_epoch > 0 and LOG_CSV.exists():
            print(f"[EpochMonitor] Loading previous history from {LOG_CSV}...")
            limit_step = self.start_epoch * STEPS_PER_EPOCH
            try:
                with open(LOG_CSV, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for row in reader:
                        if not row:
                            continue
                        step = int(row[0])
                        if step <= limit_step:
                            self.old_csv_rows.append(row)
                            # Populate training curves for plotting
                            row_type = row[1]
                            val = float(row[2])
                            if row_type == "train_loss":
                                if step not in self.step_list:
                                    self.step_list.append(step)
                                    self.train_loss.append(val)
                                if row[4]: # lr
                                    self.lr_list.append(float(row[4]))
                            elif row_type == "grad_norm":
                                self.grad_norm.append(val)
                            elif row_type == "eval_loss":
                                if step not in self.val_steps:
                                    self.val_steps.append(step)
                                    self.val_loss.append(val)

                # Reconstruct per-epoch aggregates
                for ep in range(1, self.start_epoch + 1):
                    ep_start_step = (ep - 1) * STEPS_PER_EPOCH
                    ep_end_step = ep * STEPS_PER_EPOCH
                    # Average training loss in this range
                    losses_in_ep = [l for s, l in zip(self.step_list, self.train_loss) if ep_start_step < s <= ep_end_step]
                    avg_train = sum(losses_in_ep) / len(losses_in_ep) if losses_in_ep else float("nan")
                    # Validation loss at end of epoch
                    val_at_ep = [l for s, l in zip(self.val_steps, self.val_loss) if s == ep_end_step]
                    latest_val = val_at_ep[-1] if val_at_ep else float("nan")
                    
                    self.epoch_nums.append(ep)
                    self.epoch_train_loss.append(avg_train)
                    self.epoch_val_loss.append(latest_val)
                    self.epoch_wall_time.append(0.0) # placeholder for wall time
                print(f"[EpochMonitor] Successfully loaded history for steps <= {limit_step} (Epoch {self.start_epoch})")
            except Exception as e:
                print(f"[EpochMonitor] Warning: Could not load previous history: {e}")

    # ---- helpers ------------------------------------------------

    def _refresh_buffers(self, state: TrainerState):
        # Clear only the elements belonging to the current run
        current_limit = self.start_epoch * STEPS_PER_EPOCH
        self.step_list = [s for s in self.step_list if s <= current_limit]
        self.train_loss = self.train_loss[:len(self.step_list)]
        self.lr_list = self.lr_list[:len(self.step_list)]
        self.grad_norm = self.grad_norm[:len(self.step_list)]
        self.val_steps = [s for s in self.val_steps if s <= current_limit]
        self.val_loss = self.val_loss[:len(self.val_steps)]

        for entry in state.log_history:
            step = entry.get("step")
            if step is None:
                continue
            # Offset steps for continuous plotting
            offset_step = int(step) + current_limit
            if "loss" in entry and "eval_loss" not in entry:
                if offset_step not in self.step_list:
                    self.step_list.append(offset_step)
                    self.train_loss.append(entry["loss"])
                    if "learning_rate" in entry:
                        self.lr_list.append(entry["learning_rate"])
                    if "grad_norm" in entry:
                        self.grad_norm.append(entry["grad_norm"])
            if "eval_loss" in entry:
                if offset_step not in self.val_steps:
                    self.val_steps.append(offset_step)
                    self.val_loss.append(entry["eval_loss"])

    def _write_csv(self, state: TrainerState):
        current_limit = self.start_epoch * STEPS_PER_EPOCH
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "type", "value", "extra_key", "extra_value"])
            # Write old history first
            for row in self.old_csv_rows:
                writer.writerow(row)
            # Write new history
            for entry in state.log_history:
                step = entry.get("step")
                if step is None:
                    continue
                offset_step = int(step) + current_limit
                if "loss" in entry and "eval_loss" not in entry:
                    writer.writerow([offset_step, "train_loss", entry["loss"],
                                     "lr", entry.get("learning_rate", ""),])
                    if "grad_norm" in entry:
                        writer.writerow([offset_step, "grad_norm", entry["grad_norm"], "", ""])
                if "eval_loss" in entry:
                    writer.writerow([offset_step, "eval_loss", entry["eval_loss"], "", ""])

    def _draw_charts(self, epoch_idx: int):
        """Save charts to plots/epoch_N/ (snapshot) and plots/latest/ (current)."""
        import shutil

        epoch_dir  = PLOTS_ROOT / f"epoch_{epoch_idx:02d}"
        latest_dir = PLOTS_ROOT / "latest"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)

        # helper: save fig to both dirs then close
        def _save(fig, name: str):
            fig.savefig(epoch_dir  / name, dpi=130)
            fig.savefig(latest_dir / name, dpi=130)
            plt.close(fig)

        # helper: subtle grid
        def _grid(ax):
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
            ax.set_facecolor("#f8f9fa")

        # 1. Train + Val Loss (step-level, smoothed)
        fig, ax = plt.subplots(figsize=(10, 5))
        if self.train_loss:
            window = max(1, len(self.train_loss) // 40)
            smoothed = [
                sum(self.train_loss[max(0, i - window): i + 1]) /
                len(self.train_loss[max(0, i - window): i + 1])
                for i in range(len(self.train_loss))
            ]
            ax.plot(self.step_list, self.train_loss, alpha=0.25, color="#4C72B0", linewidth=0.8, label="_raw")
            ax.plot(self.step_list, smoothed, color="#4C72B0", linewidth=1.8, label="Train loss (smoothed)")
        if self.val_loss:
            ax.plot(self.val_steps, self.val_loss, color="#DD8452", linewidth=2.0,
                    marker="o", markersize=4, label="Val loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Train vs Validation Loss — step-level (after epoch {epoch_idx})")
        ax.legend()
        _grid(ax)
        fig.tight_layout()
        _save(fig, "loss_vs_step.png")

        # 2. Per-epoch loss overview
        if self.epoch_nums:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(self.epoch_nums, self.epoch_train_loss, color="#4C72B0",
                    marker="o", linewidth=2.0, label="Avg train loss / epoch")
            ax.plot(self.epoch_nums, self.epoch_val_loss, color="#DD8452",
                    marker="s", linewidth=2.0, label="Val loss / epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Train vs Validation Loss (per epoch) — ⚠ val rising = overfit")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend()
            _grid(ax)
            fig.tight_layout()
            _save(fig, "loss_vs_epoch.png")

            # 3. Wall-clock time per epoch
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(self.epoch_nums, self.epoch_wall_time, color="#55A868", width=0.6)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Seconds")
            ax.set_title("Wall-clock time per epoch")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            _grid(ax)
            fig.tight_layout()
            _save(fig, "epoch_wall_time.png")

        # 4. Learning-rate schedule
        if self.lr_list:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(self.step_list[: len(self.lr_list)], self.lr_list,
                    color="#8172B2", linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("LR")
            ax.set_title("Learning-rate schedule")
            _grid(ax)
            fig.tight_layout()
            _save(fig, "lr_schedule.png")

        # 5. Gradient norm
        if self.grad_norm:
            gn_steps = self.step_list[: len(self.grad_norm)]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(gn_steps, self.grad_norm, color="#C44E52", linewidth=0.9, alpha=0.7)
            ax.set_xlabel("Step")
            ax.set_ylabel("Grad norm")
            ax.set_title("Gradient norm (instability detector)")
            _grid(ax)
            fig.tight_layout()
            _save(fig, "grad_norm.png")

    # ---- TrainerCallback hooks ----------------------------------

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        epoch_idx = int(round(state.epoch)) + self.start_epoch
        elapsed   = time.time() - self._epoch_start_time

        # ------ 1. Save adapter ------
        adapter_path = Path(OUTPUT_DIR) / f"adapter_epoch_{epoch_idx}"
        print(f"\n[EpochMonitor] Epoch {epoch_idx} done ({elapsed:.0f}s). Saving adapter → {adapter_path}")
        model.save_pretrained(str(adapter_path))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(adapter_path))

        # ------ 2. Refresh buffers ------
        self._refresh_buffers(state)

        # Compute avg train loss for this epoch from log_history
        epoch_train_entries = [
            e["loss"] for e in state.log_history
            if "loss" in e and "eval_loss" not in e
            and abs(e.get("epoch", -1) - state.epoch) < 1.0
        ]
        avg_train = sum(epoch_train_entries) / len(epoch_train_entries) if epoch_train_entries else float("nan")

        # Latest val loss
        val_entries = [e["eval_loss"] for e in state.log_history if "eval_loss" in e]
        latest_val  = val_entries[-1] if val_entries else float("nan")

        self.epoch_nums.append(epoch_idx)
        self.epoch_train_loss.append(avg_train)
        self.epoch_val_loss.append(latest_val)
        self.epoch_wall_time.append(elapsed)

        # ------ 3. Write CSV ------
        self._write_csv(state)

        # ------ 4. Redraw charts (saved every epoch) ------
        self._draw_charts(epoch_idx)

        print(f"[EpochMonitor] Epoch {epoch_idx}: avg_train_loss={avg_train:.4f}  val_loss={latest_val:.4f}")
        print(f"[EpochMonitor] Snapshot → {PLOTS_ROOT}/epoch_{epoch_idx:02d}/")
        print(f"[EpochMonitor] Latest   → {PLOTS_ROOT}/latest/   (always current)")
        print(f"[EpochMonitor] CSV log  → {LOG_CSV}")


# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
LOAD_PATH = RESUME_ADAPTER_PATH if RESUME_ADAPTER_PATH else MODEL_NAME
print(f"Loading model from: {LOAD_PATH}...")
model, tokenizer = FastVisionModel.from_pretrained(
    LOAD_PATH,
    load_in_4bit=False,
    use_gradient_checkpointing=False,
)
print("loaded")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28
)
processor.chat_template = processor.tokenizer.chat_template

try:    print("model_type =", model.config.model_type)
except Exception as e: print("model_type error:", e)
try:    print("architectures =", model.config.architectures)
except Exception as e: print("architectures error:", e)

print("\n=== CONFIG ===")
print(model.config)

# ------------------------------------------------------------------
# LoRA (Skip get_peft_model if already loaded a PEFT model)
# ------------------------------------------------------------------
if not RESUME_ADAPTER_PATH:
    print("Applying LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("lora applied")
else:
    print("PEFT adapter already applied (loaded from checkpoint). Skipping get_peft_model.")

print("\n=== AFTER LORA ===")
print(type(model))
print(model.__class__.__name__)
for attr in ["model", "base_model"]:
    if hasattr(model, attr):
        print(f"\n{attr}:")
        obj = getattr(model, attr)
        print(type(obj))
        print(obj.__class__.__name__)
        try:    print("model_type =", obj.config.model_type)
        except: pass

# ------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------
print("Loading datasets...")
train_dataset = load_dataset("json", data_files="/workspace/train_comp_data_final.jsonl", split="train")
val_dataset   = load_dataset("json", data_files="/workspace/val_comp_data_final.jsonl",   split="train")
print("Train size:", len(train_dataset))
print("Val size:",   len(val_dataset))

print("\n=== FIRST TRAIN EXAMPLE ===")
print(train_dataset[0])
print("\n=== DATASET FEATURES ===")
print(train_dataset.features)

FastVisionModel.for_training(model)

# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    processing_class=processor,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS - START_EPOCH,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=5,
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported(),
        logging_steps=1,
        eval_strategy="epoch",          # evaluate once per epoch for clean curves
        save_strategy="no",             # we save ourselves in EpochMonitorCallback
        load_best_model_at_end=False,   # you pick the epoch manually on overfit
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
        report_to="none",
        seed=3407,
    ),
    callbacks=[EpochMonitorCallback(start_epoch=START_EPOCH)],
)

print("Starting training...")
print(f"  Epochs:           {NUM_EPOCHS}")
print(f"  Save interval:    every epoch")
print(f"  Adapter saves:    {OUTPUT_DIR}/adapter_epoch_N/")
print(f"  Plot snapshots:   {PLOTS_ROOT}/epoch_NN/   (one folder per epoch)")
print(f"  Plot latest:      {PLOTS_ROOT}/latest/     (always the most recent)")
print(f"  CSV log:          {LOG_CSV}")

try:
    trainer.train()
except KeyboardInterrupt:
    print("\n[Interrupted by user — Ctrl+C detected]")
    print("Saving current state before exit...")

# ------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------
log_history = trainer.state.log_history
train_losses = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]
val_losses   = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
print("Train losses:", train_losses[-5:], "...")
print("Val losses:",   val_losses)

print("Saving final adapter...")
model.save_pretrained(f"{OUTPUT_DIR}/final_adapter_corpus")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter_corpus")
print("Done. All adapters and charts are in", OUTPUT_DIR)
