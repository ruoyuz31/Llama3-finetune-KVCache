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




class AlpacaDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_length: int):
        self.tokenizer = tokenizer
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        self.samples = self.preprocess(raw_data[:200], max_length)

    def preprocess(self, raw_data, max_length):
        samples = []
        for ex in raw_data:
            prompt = PROMPT_DICT["prompt_input"].format_map(ex) if ex.get("input") \
                else PROMPT_DICT["prompt_no_input"].format_map(ex)
            full_text = prompt + " " + ex["output"] + DEFAULT_EOS_TOKEN
            encoded = self.tokenizer.encode(full_text, bos=True, eos=False)
            print(len(encoded), max_length)
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            labels = encoded.copy()
            prompt_ids = self.tokenizer.encode(prompt, bos=True, eos=False)
            # print(self.tokenizer.decode(encoded))
            # exit(0)  # RZ: remove this line to process all data
            labels[:len(prompt_ids)] = [IGNORE_INDEX] * len(prompt_ids)
            samples.append((torch.tensor(encoded), torch.tensor(labels)))
            # print(labels)
            # print(self.tokenizer.decode(labels[len(prompt_ids):]))
            # exit(0)  # RZ: remove this line to process all data
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn_w_mask(batch):
    input_ids, labels = zip(*batch)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids != 0
    return input_ids, attention_mask, labels


class LlamaForCausalLM(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        model_args = ModelArgs()
        self.model = Llama(model_args)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "consolidated.00.pth"), map_location="cpu"), strict=True)
        self.model = self.model.to("cuda")

    def forward(self, input_ids, attention_mask):
        B, T = input_ids.shape
        x = self.model.tok_embeddings(input_ids)
        freqs = self.freqs_cis[:T].to(input_ids.device)

        mask = torch.full((T, T), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        h = x
        for layer in self.model.layers:
            h = layer(h, 0, freqs, mask)

        h = self.model.norm(h)
        return self.model.output(h)

def inference():
    torch.manual_seed(1)
    kv_caching = True
    data_path = "alpaca/alpaca_data.json"
    batch_size = 2
    checkppoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
    tokenizer_path = os.path.join(checkppoint_dir, "tokenizer.model")
    model_path = os.path.join(checkppoint_dir, "consolidated.00.pth")

    tokenizer = Tokenizer(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = ModelArgs()
    #torch.set_default_tensor_type(torch.cuda.HalfTensor) # load model in fp16
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    device = "cuda"
    model.to(device)

    dataset = SupervisedDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    

    model.eval()
    full_ids, output_ids = model.generate_tokens(tokenizer, input_ids, labels, temperature=0.6, top_p=0.9, kv_caching=kv_caching, device=device)

    for input_id, out_id, label in zip(input_ids, output_ids, labels):
        print(f"input >")
        print(tokenizer.decode(input_id.tolist()))
        print(f"output >")
        print(tokenizer.decode(out_id))
        print("\n==================================\n")
        print(f"labels >")
        label = [t for t in label if t != IGNORE_INDEX]
        print(tokenizer.decode(label))
        print("\n==================================\n")


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
    max_seq_len: int = 256
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
    device = "cuda"
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

                ### Optimizer state memory
                opt_bytes = 0
                for group in optimizer.state.values():
                    for k, v in group.items():
                        if isinstance(v, torch.Tensor):
                            opt_bytes += v.numel() * v.element_size()
                print(f"  Optimizer State: {opt_bytes / 1024**2:.2f} MB")
                print(f"  Gradients: {sum(p.numel() for p in model.parameters() if p.grad is not None) / 1024**2:.2f} MB")
                print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1024**2:.2f} MB")
                print(f"  Total: {torch.cuda.memory_allocated() / 1024**2 + opt_bytes / 1024**2:.2f} MB")
                print(f"  Peak Total: {torch.cuda.max_memory_allocated() / 1024**2 + opt_bytes / 1024**2:.2f} MB")


            optimizer.step()
            total_loss += loss.item()

            if step % 2 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")

        print(f"[Epoch {epoch+1}] Average Loss: {total_loss / len(dataloader):.4f}")

        if (epoch + 1) % cfg.save_every == 0:
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch+1}.pt"))


def train_old():
    cfg = TrainingConfig(
        model_dir=os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B"),
        data_path="alpaca/alpaca_data.json",
        output_dir="llama3_alpaca_ckpt",
        device="cuda",
    )

    checkppoint_dir = cfg.model_dir
    tokenizer_path = os.path.join(checkppoint_dir, "tokenizer.model")
    model_path = os.path.join(checkppoint_dir, "consolidated.00.pth")
    tokenizer = Tokenizer(tokenizer_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    # # RZ: map to GPU
    # if torch.cuda.is_available():
    #     checkpoint = torch.load(model_path, map_location="cuda:0")
    model_args = ModelArgs()
    
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    device = "cuda"
    model.to(device)

    model.train()

    #dataset = AlpacaDataset(cfg.data_path, tokenizer, cfg.max_seq_len)
    dataset = SupervisedDataset(cfg.data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        total_loss = 0
        for step, (input_ids, labels) in enumerate(dataloader):
            max_len = max(len(t) for t in labels)
            input_ids = input_ids.to(cfg.device)
            labels = labels.to(cfg.device)
            logits = model.generate_tokens(tokenizer, input_ids, max_gen_len=max_len, temperature=0.6, top_p=0.9, kv_caching=True, device=device)
            
            #logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")

        print(f"[Epoch {epoch+1}] Average Loss: {total_loss / len(dataloader):.4f}")

        if (epoch + 1) % cfg.save_every == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
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
