# train/train_model.py
"""
Fine-tune Flan-T5-base for MCQ generation on your RTX 3050 6GB GPU.

This script:
1. Loads the curated MCQ dataset
2. Fine-tunes google/flan-t5-base using gradient accumulation (fits in 6GB VRAM)
3. Saves the trained model to models/flan-t5-mcq/

Usage:
    python train/train_model.py

Training takes ~15-30 minutes on RTX 3050.
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from train.mcq_dataset import get_training_data


# ---- Configuration ----
MODEL_NAME = "google/flan-t5-base"     # 250M params, ~1GB VRAM for inference
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "flan-t5-mcq")
EPOCHS = 30
BATCH_SIZE = 2           # Small batch for 6GB VRAM
GRAD_ACCUM_STEPS = 4     # Effective batch = 2 × 4 = 8
LEARNING_RATE = 3e-4
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 256
SEED = 42


class MCQDataset(Dataset):
    """PyTorch Dataset for MCQ training examples."""

    def __init__(self, data, tokenizer, max_input_len, max_output_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize input
        input_enc = self.tokenizer(
            item["input"],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize output (labels)
        output_enc = self.tokenizer(
            item["output"],
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token IDs with -100 so they're ignored in loss
        labels = output_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


def train():
    """Main training loop."""
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"🚀 Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU detected. Training on CPU (will be slower).")
        print("   Make sure you have PyTorch CUDA installed:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu126")

    torch.manual_seed(SEED)

    # Load tokenizer and model
    print(f"\n📦 Loading {MODEL_NAME}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    # Print model size
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {params/1e6:.0f}M total, {trainable/1e6:.0f}M trainable")

    # Load dataset
    print("\n📊 Loading training data...")
    data = get_training_data()
    dataset = MCQDataset(data, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"   {len(data)} examples, {len(dataloader)} batches per epoch")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = (len(dataloader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Training loop
    print(f"\n🏋️ Training for {EPOCHS} epochs...")
    print(f"   Batch size: {BATCH_SIZE} × {GRAD_ACCUM_STEPS} gradient accumulation = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Total steps: {total_steps}")
    print("-" * 60)

    model.train()
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            total_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)

        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  LR: {lr:.2e}")

            # Quick generation test every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                test_input = "Generate an easy multiple choice question from this text:\n\nThe sun is a star at the center of the Solar System. It is about 4.6 billion years old."
                test_enc = tokenizer(test_input, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True).to(device)
                with torch.no_grad():
                    gen = model.generate(**test_enc, max_new_tokens=MAX_OUTPUT_LEN, do_sample=False)
                    output = tokenizer.decode(gen[0], skip_special_tokens=True)
                    print(f"  📝 Sample: {output[:120]}...")
                model.train()

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save final model
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training metadata
    meta = {
        "base_model": MODEL_NAME,
        "epochs": EPOCHS,
        "final_loss": avg_loss,
        "best_loss": best_loss,
        "training_examples": len(data),
        "max_input_len": MAX_INPUT_LEN,
        "max_output_len": MAX_OUTPUT_LEN,
    }
    with open(os.path.join(OUTPUT_DIR, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Best loss: {best_loss:.4f}")
    print(f"\n✅ Training complete! Model saved to: {OUTPUT_DIR}")
    print(f"   You can now run the app with: python app.py")

    # GPU memory cleanup
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   Peak GPU memory: {peak_mem:.1f} GB")

    return OUTPUT_DIR


if __name__ == "__main__":
    train()
