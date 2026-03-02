import os
import json
import time
import logging
import argparse
import random
from typing import Dict, List, Optional, Sequence
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

# Import Llama model and tokenizer
from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama
# Hardcoded parameters
#MODEL_PATH = "/home/ec2-user/.llama/checkpoints/Llama3.2-1B"
#TOKENIZER_PATH = os.path.join(MODEL_PATH, "tokenizer.model")
DATA_PATH = r"alpaca\alpaca_data.json"
OUTPUT_DIR = "./llama_finetuned"
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_STEPS = 1000
WARMUP_STEPS = 100
GRADIENT_ACCUMULATION_STEPS = 1
LOGGING_STEPS = 10
SAVE_STEPS = 200
EVAL_STEPS = 200
MAX_GRAD_NORM = 1.0
FP16 = True
SEED = 42
NUM_WORKERS = 4
MAX_SAMPLES = 200  # Use only the first 200 examples

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_length: int = 512, max_samples=200):
        super(SupervisedDataset, self).__init__()
        logging.info(f"Loading data from {data_path}")
        list_data_dict = jload(data_path)
        if max_samples is not None:
            list_data_dict = list_data_dict[:max_samples]
            logging.info(f"Using {max_samples} samples from the dataset")
        logging.info("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = []
        targets = []
        
        for example in list_data_dict:
            if example.get("input", "") != "":
                source = prompt_input.format_map(example)
            else:
                source = prompt_no_input.format_map(example)
            target = example["output"]
            
            sources.append(source)
            targets.append(target)
        
        logging.info("Tokenizing inputs... This may take some time...")
        self.input_ids = []
        self.labels = []
        
        for source, target in zip(sources, targets):
            # Tokenize source
            source_ids = tokenizer.encode(source, bos=True, eos=False)
            
            # Tokenize target with EOS
            target_ids = tokenizer.encode(target, bos=False, eos=True)
            
            # Combine
            combined_ids = source_ids + target_ids
            
            # Truncate if needed
            if len(combined_ids) > max_length:
                combined_ids = combined_ids[:max_length]
                
            # Create labels: ignore source tokens, keep target tokens
            labels = [IGNORE_INDEX] * len(source_ids) + target_ids
            
            # Truncate labels too if needed
            if len(labels) > max_length:
                labels = labels[:max_length]
                
            # Store tokenized sequences
            input_tensor = torch.tensor(combined_ids)
            label_tensor = torch.tensor(labels[:len(input_tensor)]) 
            
            self.input_ids.append(input_tensor)
            self.labels.append(label_tensor)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }

def collate_fn(batch, pad_id):
    """Collate function for data loader."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Get max length in batch
    max_length = max(len(ids) for ids in input_ids)
    
    # Pad sequences
    padded_input_ids = []
    padded_labels = []
    attention_mask = []
    
    for ids, lbl in zip(input_ids, labels):
        padding_length = max_length - len(ids)
        
        # Create attention mask (1 for tokens, 0 for padding)
        mask = torch.ones(len(ids), dtype=torch.long)
        if padding_length > 0:
            mask = torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
        attention_mask.append(mask)
        
        # Pad input_ids
        if padding_length > 0:
            ids = torch.cat([ids, torch.full((padding_length,), pad_id, dtype=ids.dtype)])
        padded_input_ids.append(ids)
        
        # Pad labels
        if padding_length > 0:
            lbl = torch.cat([lbl, torch.full((padding_length,), IGNORE_INDEX, dtype=lbl.dtype)])
        padded_labels.append(lbl)
    
    # Stack tensors
    input_ids = torch.stack(padded_input_ids)
    labels = torch.stack(padded_labels)
    attention_mask = torch.stack(attention_mask)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

class LlamaTrainer:
    """Custom trainer for Llama model."""
    
    def __init__(
        self,
        model: Llama,
        tokenizer: Tokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        gradient_accumulation_steps: int = 1,
        fp16: bool = True,
        output_dir: str = "output",
        save_steps: int = 200,
        eval_steps: int = 200,
        logging_steps: int = 10,
        num_workers: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        
        # Create data loader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer.bos_id),  # Using BOS as pad token
            num_workers=num_workers,
        )
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lambda step: min(1.0, (step + 1) / warmup_steps) if step < warmup_steps else 1.0
        )
        
        # Set up gradient scaler for mixed precision
        self.scaler = GradScaler() if fp16 else None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Keep track of best loss for checkpointing
        self.best_loss = float('inf')
        
    def _create_optimizer(self):
        """Create optimizer with weight decay for non-bias and non-layernorm parameters."""
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "ln", "layernorm", "layer_norm"]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
    
    def train(self):
        """Run training loop."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Training loop
        global_step = 0
        total_loss = 0.0
        logging_loss = 0.0
        step_bar = range(self.max_steps)
        
        self.model.train()
        accumulated_steps = 0
        
        # Calculate total batches per epoch
        epoch_iterator = iter(self.train_dataloader)
        
        for step in step_bar:
            try:
                batch = next(epoch_iterator)
            except StopIteration:
                epoch_iterator = iter(self.train_dataloader)
                batch = next(epoch_iterator)
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            if self.fp16:
                with autocast(device_type='cuda'):
                    logits = self.model(batch["input_ids"], start_pos=0, attention_mask=batch["attention_mask"])

                    
                    # Shift logits and labels for next token prediction
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = batch["labels"][:, 1:].contiguous()
                    
                    # Calculate loss using cross-entropy
                    loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                # Scale loss and backward pass
                self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
            else:
                logits = self.model(batch["input_ids"], start_pos = 0, attention_mask = ["attention_mask"])
                
                # Shift logits and labels for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch["labels"][:, 1:].contiguous()
                
                # Calculate loss using cross-entropy
                loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Normalize loss for gradient accumulation
                (loss / self.gradient_accumulation_steps).backward()
            
            accumulated_steps += 1
            total_loss += loss.item()
            
            # Update model parameters after accumulating gradients
            if accumulated_steps == self.gradient_accumulation_steps:
                # Clip gradients
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update weights
                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                global_step += 1
                accumulated_steps = 0
                
                # Logging
                if global_step % self.logging_steps == 0:
                    avg_loss = (total_loss - logging_loss) / self.logging_steps
                    logging.info(f"Step: {global_step}, Loss: {avg_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.8f}")
                    logging_loss = total_loss
                
                # Save model checkpoint
                if global_step % self.save_steps == 0:
                    self._save_checkpoint(global_step, total_loss / global_step)
                
                # Evaluation
                if global_step % self.eval_steps == 0 and self.eval_dataset is not None:
                    self._evaluate(global_step)
            
            # Check if we've reached max steps
            if global_step >= self.max_steps:
                break
        
        # Save final model
        self._save_checkpoint(global_step, total_loss / global_step, final=True)
        logging.info(f"Training completed. Final loss: {total_loss / global_step:.4f}")
    
    def _save_checkpoint(self, step, loss, final=False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        if final:
            checkpoint_dir = self.output_dir
            
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, "consolidated.00.pth")
        )
        
        # Save tokenizer model if available
        if hasattr(self.tokenizer, 'model_path'):
            import shutil
            shutil.copy(self.tokenizer.model_path, os.path.join(checkpoint_dir, "tokenizer.model"))
            
        # Save training info
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as f:
            json.dump({
                "step": step,
                "loss": loss,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "timestamp": time.time(),
            }, f)
            
        logging.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")
        
        # Keep track of best model
        if loss < self.best_loss:
            self.best_loss = loss
            best_dir = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(
                self.model.state_dict(),
                os.path.join(best_dir, "consolidated.00.pth")
            )
            logging.info(f"Saved best model with loss {loss:.4f} to {best_dir}")
    
    def _evaluate(self, step):
        """Run evaluation."""
        logging.info(f"Running evaluation at step {step}")
        
        # TODO: Implement evaluation logic if needed
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to pretrained Llama model directory")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer model file (default: model_path/tokenizer.model)")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to fine-tuning dataset JSON file")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./llama_ft_output",
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Total number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log training metrics every X steps")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="Evaluate model every X steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    return parser.parse_args()

def main():
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    # Parse arguments
    # args = parse_args()
    
    # # Set random seed
    # set_seed(args.seed)
    
    # # Set tokenizer path if not provided
    # if args.tokenizer_path is None:
    #     args.tokenizer_path = os.path.join(args.model_path, "tokenizer.model")
    
    # # Load tokenizer
    # logging.info(f"Loading tokenizer from {args.tokenizer_path}")
    # tokenizer = Tokenizer(args.tokenizer_path)
    
    # # Load model
    # logging.info(f"Loading model from {args.model_path}")
    # model_path = os.path.join(args.model_path, "consolidated.00.pth")
    # checkpoint = torch.load(model_path, map_location="cpu")
    
    # # Set up model
    # model_args = ModelArgs()
    # model = Llama(model_args)
    # model.load_state_dict(checkpoint, strict=False)
    # logging.info(f"Model loaded successfully")
    
    # # Create dataset
    # train_dataset = SupervisedDataset(
    #     data_path=args.data_path,
    #     tokenizer=tokenizer,
    #     max_length=args.max_length
    # )
    # logging.info(f"Loaded {len(train_dataset)} training examples")
    
    # # Create trainer
    # trainer = LlamaTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     batch_size=args.batch_size,
    #     learning_rate=args.learning_rate,
    #     weight_decay=args.weight_decay,
    #     max_steps=args.max_steps,
    #     warmup_steps=args.warmup_steps,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     fp16=args.fp16,
    #     output_dir=args.output_dir,
    #     save_steps=args.save_steps,
    #     eval_steps=args.eval_steps,
    #     logging_steps=args.logging_steps,
    #     max_grad_norm=args.max_grad_norm,
    #     num_workers=args.num_workers
    # )

    #logging.info(f"Loading tokenizer from {TOKENIZER_PATH}")
    #tokenizer = Tokenizer(TOKENIZER_PATH)

    checkppoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
    tokenizer_path = os.path.join(checkppoint_dir, "tokenizer.model")
    logging.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer(tokenizer_path)
    
    # Load model
    #logging.info(f"Loading model from {MODEL_PATH}")
    #model_path = os.path.join(MODEL_PATH, "consolidated.00.pth")
    model_path = os.path.join(checkppoint_dir, "consolidated.00.pth")
    logging.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Set up model
    model_args = ModelArgs()
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=False)
    logging.info(f"Model loaded successfully")
    
    # Create dataset
    train_dataset = SupervisedDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        max_samples=MAX_SAMPLES
    )
    logging.info(f"Loaded {len(train_dataset)} training examples")
    
    # Create trainer
    trainer = LlamaTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        fp16=FP16,
        output_dir=OUTPUT_DIR,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        num_workers=NUM_WORKERS
    )
    
    # Start training
    logging.info("Starting training")
    trainer.train()
    logging.info(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()