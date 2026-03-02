import datetime
import os
import sys
import math
import json
from typing import Dict, Optional, Sequence
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, List, Dict

from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_PAD_TOKEN_ID = -1
DATA_SIZE = 200

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

def _tokenize_fn(strings: Sequence[str], tokenizer: Tokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        # tokenizer(
        #     text,
        #     return_tensors="pt",
        #     padding="longest",
        #     max_length=tokenizer.model_max_length,
        #     truncation=True,
        # )
        tokenizer.encode(text, bos=True, eos=False)
        for text in strings
    ]
    input_ids = labels = tokenized_list
    input_ids_lens = labels_lens = [
        len(tokenized) for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: Tokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]

    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    example_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(example_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = [IGNORE_INDEX] * source_len
    return dict(input_ids=example_ids, labels=labels, prompt_lens=sources_tokenized["input_ids_lens"])




class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: Tokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        list_data_dict = raw_data[:DATA_SIZE]

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{DEFAULT_EOS_TOKEN}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.prompt_lens = data_dict["prompt_lens"]
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_id

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], prompt_lens=self.prompt_lens[i])
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in batch]

        prompt_lens = [torch.tensor(example["prompt_lens"], dtype=torch.long) for example in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(input_ids=input_ids, labels=labels, prompt_lens=prompt_lens)

@dataclass
class TrainingConfig:
    model_dir: str
    data_path: str
    output_dir: str
    batch_size: int = 1 # need to deal with pad_id if batch_size > 1
    lr: float = 1e-5
    epochs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_every: int = 1

def train():
    cfg = TrainingConfig(
        model_dir=os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B"),
        data_path="alpaca/alpaca_data.json",
        output_dir="llama3_alpaca_ckpt/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        device="cuda",
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(cfg.output_dir, "train.log"))
    data_path = "alpaca/alpaca_data.json"
    checkppoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
    tokenizer_path = os.path.join(checkppoint_dir, "tokenizer.model")
    model_path = os.path.join(checkppoint_dir, "consolidated.00.pth")
    # load model from the saved checkpoint from output_dir
    #model_path = os.path.join(cfg.output_dir, "checkpoint_epoch3.pt")

    tokenizer = Tokenizer(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = ModelArgs()
    model_args.kv_caching = False
    #torch.set_default_tensor_type(torch.cuda.HalfTensor) # load model in fp16
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    device = cfg.device
    model.to(device)

    dataset = SupervisedDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    model.train()

    for epoch in range(cfg.epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids,start_pos = 0)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()

            if step % 2 == 0:
                torch.cuda.synchronize()
                print(f"\n[Memory] Epoch {epoch+1} Step {step}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"  Reserved : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                print(f"  Peak Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

                # Optimizer state memory
                opt_bytes = 0
                for group in optimizer.state.values():
                    for k, v in group.items():
                        if isinstance(v, torch.Tensor):
                            opt_bytes += v.numel() * v.element_size()
                print(f"  Optimizer State: {opt_bytes / 1024**2:.2f} MB")

                # Gradient memory
                grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
                print(f"  Gradients: {grad_bytes / 1024**2:.2f} MB")

                # Parameter memory
                param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                print(f"  Parameters: {param_bytes / 1024**2:.2f} MB")

            optimizer.step()
            total_loss += loss.item()

            if step % 2 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")

        print(f"[Epoch {epoch+1}] Average Loss: {total_loss / len(dataloader):.4f}")

        if (epoch + 1) % cfg.save_every == 0:
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch+1}.pt"))

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()



if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # tokenizer = Tokenizer(os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/tokenizer.model"))
    # print(tokenizer.pad_id,tokenizer.eos_id,tokenizer.bos_id)
    # x = torch.ones(4, dtype=torch.complex64).cuda()
    # print("Complex CUDA OK:", x * x)    
    #inference()
    
    
    # for i in range(5):
    #     print(f"Hello {i}")
    train()
