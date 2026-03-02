import datetime
import os
import time
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
    examples = [s+ " " + t for s, t in zip(sources, targets)] 
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    example_ids = examples_tokenized["input_ids"]
    # targets_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (targets, sources)]
    # example_ids = [target+source for target, source in zip(targets_tokenized["input_ids"], sources_tokenized["input_ids"])]
    
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
    model_dir: str = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
    data_path: str = "alpaca/alpaca_data.json"
    output_dir: str = "llama3_alpaca_ckpt/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_size: int = 1 # need to deal with pad_id if batch_size > 1
    lr: float = 1e-5
    epochs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print_every: int = 1 # print every n steps
    save_every: int = 5 # save every n epochs
    
    grad_accumulation: bool = False
    grad_accumulation_steps: int = 8 # Gradient accumulation steps
    
    mixed_precision: bool = False # use mixed precision training
    use_amp: bool = True # use automatic mixed precision (AMP) for training
    
    grad_checkpointing: bool = True # use gradient checkpointing to save memory
    # a list of checkpointing layers, starting from embedding layer
    grad_checkpointing_list: List[str] = None
    
    lora: bool = False # use LoRA for training
    lora_r: int = 16 # LoRA rank
    lora_alpha: int = 32 # LoRA alpha
    lora_dropout: float = 0.05 # LoRA dropout

    # print cfg
    def print(self):
        print(f"#########################################################")
        print(f"#################     TrainingConfig    #################")
        print(f"#########################################################")
        print(f"  model_dir: {self.model_dir}")
        print(f"  data_path: {self.data_path}")
        print(f"  output_dir: {self.output_dir}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  lr: {self.lr}")
        print(f"  epochs: {self.epochs}")
        print(f"  device: {self.device}")
        print(f"  grad_accumulation: {self.grad_accumulation}")
        print(f"  grad_accumulation_steps: {self.grad_accumulation_steps}")
        print(f"  mixed_precision: {self.mixed_precision}")
        print(f"  use_amp: {self.use_amp}")
        print(f"  grad_checkpointing: {self.grad_checkpointing}")
        print(f"  grad_checkpointing_list: {self.grad_checkpointing_list}")
        print(f"  lora: {self.lora}")
        print(f"  lora_r: {self.lora_r}")
        print(f"  lora_alpha: {self.lora_alpha}")
        print(f"  lora_dropout: {self.lora_dropout}")
        print(f"  print_every: {self.print_every}")
        print(f"  save_every: {self.save_every}")
        print(f"#########################################################")
        

def train():
    torch.manual_seed(1)
    max_step = 10
    cfg = TrainingConfig()
    if cfg.grad_accumulation:
        cfg.output_dir += "_grad_accumulation"
    if cfg.mixed_precision:
        cfg.output_dir += "_mixed_precision"
    if cfg.grad_checkpointing:
        cfg.output_dir += "_grad_checkpointing"
        cfg.grad_checkpointing_list = [
        # "embedding",  # embedding layer cannot be checkpointed
        "attention.wq",
        "attention.wk",
        "attention.wv",
        "attention.wo",
        "ffc.w1",
        "ffc.w2",
        "ffc.w3",
        "attention_norm",
        "ffc_norm",
        "norm",
        "output",
        ]
    if cfg.lora:
        cfg.output_dir += "_lora"
    os.makedirs(cfg.output_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(cfg.output_dir, "train.log"))
    cfg.print()
    # save the cfg to the output_dir which can be used for inference
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=4)
    # save grad_checkpointing_list to the output_dir
    if cfg.grad_checkpointing:
        with open(os.path.join(cfg.output_dir, "grad_checkpointing_list.json"), "w") as f:
            json.dump(cfg.grad_checkpointing_list, f, indent=4)
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
    
    if cfg.grad_checkpointing:
        model_args.grad_checkpointing = True
        model_args.grad_checkpointing_list = cfg.grad_checkpointing_list
    model = Llama(model_args)
    
    model.load_state_dict(checkpoint, strict=True)
    device = cfg.device
    model.to(device)
    print(model)
    dataset = SupervisedDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    if cfg.mixed_precision:
        scaler = torch.amp.GradScaler("cuda" ,enabled=cfg.use_amp)

    model.train()
    # add a timer to measure the training time
    total_start_time = time.time()
    for epoch in range(cfg.epochs):
        total_loss = 0
        epoch_start_time = time.time()
        for step, batch in enumerate(dataloader):
            step_start_time = time.time()
            if cfg.mixed_precision:
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.use_amp):
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    logits = model(input_ids,start_pos = 0)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids,start_pos = 0)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            
            if cfg.grad_accumulation:
                loss = loss / cfg.grad_accumulation_steps
            if cfg.mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if cfg.grad_accumulation:
                # Gradient accumulation
                if (step+1) % cfg.grad_accumulation_steps == 0:
                    if cfg.mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    print(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")
                    print("Gradient accumulation optimizer step")
            else:
                if cfg.mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            if step >= max_step:
                break
            
            
            print(f"Epoch {epoch+1} Step {step}")
            print(f"    Loss: {loss.item():.4f}")
            step_end_time = time.time()
            print(f"    Time: {step_end_time - step_start_time:.10f} seconds")

            # print the resource usage every cfg.print_every steps
            if (step) % cfg.print_every == 0:
                torch.cuda.synchronize()
                print(f"    Allocated: {torch.cuda.memory_allocated() / 1024**2:.10f} MB")
                print(f"    Reserved : {torch.cuda.memory_reserved() / 1024**2:.10f} MB")
                print(f"    Peak Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.10f} MB")

                # Optimizer state memory
                opt_bytes = 0
                for group in optimizer.state.values():
                    for k, v in group.items():
                        if isinstance(v, torch.Tensor):
                            opt_bytes += v.numel() * v.element_size()
                print(f"    Optimizer State: {opt_bytes / 1024**2:.10f} MB")

                # Gradient memory
                grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
                print(f"    Gradients: {grad_bytes / 1024**2:.10f} MB")

                # Parameter memory
                param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                print(f"    Parameters: {param_bytes / 1024**2:.10f} MB")

                
            
                
        
        epoch_end_time = time.time()
        print(f"[Epoch {epoch+1}]")
        print(f"    Average Loss: {total_loss / len(dataloader):.4f}")
        print(f"    Total Time: {epoch_end_time - epoch_start_time:.2f} seconds")
        if (epoch + 1) % cfg.save_every == 0:
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch+1}.pt"))
    print("Training finished")
    total_end_time = time.time()
    print(f"    Total Time: {total_end_time - total_start_time:.2f} seconds")

def inference():
    torch.manual_seed(1)
    
    kv_caching = True
    ##### future version: load cfg from the saved config.json file
    # get cfg from the saved config.json file
    # with open(os.path.join(folder_path, "config.json"), "r") as f:
    #     cfg = TrainingConfig()
    #     cfg.__dict__ = json.load(f)
    # if cfg.grad_checkpointing:
    #     with open(os.path.join(folder_path, "grad_checkpointing_list.json"), "r") as f:
    #         cfg.grad_checkpointing_list = json.load(f)
    
    ##### current version: hard code the cfg
    
    folder_path = "llama3_alpaca_ckpt/2025-05-11_00-24-46_grad_accumulation"
    cfg = TrainingConfig()
    cfg.output_dir = folder_path
    if "grad_accumulation" in cfg.output_dir:
        cfg.grad_accumulation = True
    if "mixed_precision" in cfg.output_dir:
        cfg.mixed_precision = True
    if "lora" in cfg.output_dir:
        cfg.lora = True
    cfg.grad_checkpointing = False
    sys.stdout = Tee(os.path.join(cfg.output_dir, "inference_"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".log"))
    cfg.print()
    data_path = "alpaca/alpaca_data.json"
    checkppoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
    tokenizer_path = os.path.join(checkppoint_dir, "tokenizer.model")
    # load model from the saved checkpoint from output_dir
    model_path = os.path.join(cfg.output_dir, "checkpoint_epoch5.pt")

    tokenizer = Tokenizer(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = ModelArgs()
    model_args.kv_caching = kv_caching
    #torch.set_default_tensor_type(torch.cuda.HalfTensor) # load model in fp16
    
    model = Llama(model_args)
    
    model.load_state_dict(checkpoint, strict=True)
    device = cfg.device
    model.to(device)
    print(model)
    dataset = SupervisedDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # get inference output for all the data in the dataset with generation.py
    max_step = 10
    for step, batch in enumerate(dataloader):
        
        prompts_lens = batch["prompt_lens"]
        
       


        prompts = batch["input_ids"].to(device)
        # cut the each prompt in prompts to its length and convert to a list
        prompts = [prompt[:prompt_len] for prompt, prompt_len in zip(prompts, prompts_lens)]
        # convert to string
        prompts = [tokenizer.decode(prompt.tolist()) for prompt in prompts]

        # get the targets from the labels
        targets = batch["labels"].to(device)
        # remove the prompt part from the targets
        targets = [target[prompt_len:] for target, prompt_len in zip(targets, prompts_lens)]
        max_gen_len = max([len(target) for target in targets])
        # convert to string
        targets = [tokenizer.decode(target.tolist()) for target in targets]

        # set the max_gen_len to the max length of the targets
        

        # generate the output
        model.eval()
        results = model.generate(
            tokenizer,
            prompts,
            max_gen_len=max_gen_len,
            temperature=0.6,
            top_p=0.9,
            kv_caching=kv_caching,
            device=device,
        )
        if step >= max_step:
            break

        for prompt, result, target in zip(prompts, results, targets):
            # print step
            print(f"Step {step}")
            print(prompt)
            print(f">>>>>Result: {result['generation']}")
            # print the target
            print(f">>>>>Target: {target}")
            print("\n==================================\n")

def inference_original():
    torch.manual_seed(2)
    
    kv_caching = True
    ##### future version: load cfg from the saved config.json file
    # get cfg from the saved config.json file
    # with open(os.path.join(folder_path, "config.json"), "r") as f:
    #     cfg = TrainingConfig()
    #     cfg.__dict__ = json.load(f)
    # if cfg.grad_checkpointing:
    #     with open(os.path.join(folder_path, "grad_checkpointing_list.json"), "r") as f:
    #         cfg.grad_checkpointing_list = json.load(f)
    
    ##### current version: hard code the cfg
    
    folder_path = "llama3_alpaca_ckpt/original"
    cfg = TrainingConfig()
    cfg.output_dir = folder_path
    if "grad_accumulation" in cfg.output_dir:
        cfg.grad_accumulation = True
    if "mixed_precision" in cfg.output_dir:
        cfg.mixed_precision = True
    if "lora" in cfg.output_dir:
        cfg.lora = True
    cfg.grad_checkpointing = False
    sys.stdout = Tee(os.path.join(cfg.output_dir, "inference_"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".log"))
    cfg.print()
    data_path = "alpaca/alpaca_data.json"
    checkppoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
    tokenizer_path = os.path.join(checkppoint_dir, "tokenizer.model")
    # load model from the saved checkpoint from output_dir
    model_path = os.path.join(checkppoint_dir, "consolidated.00.pth")

    tokenizer = Tokenizer(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = ModelArgs()
    model_args.kv_caching = kv_caching
    #torch.set_default_tensor_type(torch.cuda.HalfTensor) # load model in fp16
    
    model = Llama(model_args)
    
    model.load_state_dict(checkpoint, strict=True)
    device = cfg.device
    model.to(device)
    print(model)
    dataset = SupervisedDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # get inference output for all the data in the dataset with generation.py
    max_step = 10
    for step, batch in enumerate(dataloader):
        
        prompts_lens = batch["prompt_lens"]
        
       


        prompts = batch["input_ids"].to(device)
        # cut the each prompt in prompts to its length and convert to a list
        prompts = [prompt[:prompt_len] for prompt, prompt_len in zip(prompts, prompts_lens)]
        # convert to string
        prompts = [tokenizer.decode(prompt.tolist()) for prompt in prompts]

        # get the targets from the labels
        targets = batch["labels"].to(device)
        # remove the prompt part from the targets
        targets = [target[prompt_len:] for target, prompt_len in zip(targets, prompts_lens)]
        max_gen_len = max([len(target) for target in targets])
        # convert to string
        targets = [tokenizer.decode(target.tolist()) for target in targets]

        # set the max_gen_len to the max length of the targets
        

        # generate the output
        model.eval()
        results = model.generate(
            tokenizer,
            prompts,
            max_gen_len=max_gen_len,
            temperature=0.6,
            top_p=0.9,
            kv_caching=kv_caching,
            device=device,
        )
        if step >= max_step:
            break

        for prompt, result, target in zip(prompts, results, targets):
            # print step
            print(f"Step {step}")
            print(prompt)
            print(f">>>>>Result: {result['generation']}")
            # print the target
            print(f">>>>>Target: {target}")
            print("\n==================================\n")    

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
    inference()
    
    
    # for i in range(5):
    #     print(f"Hello {i}")
    # train()
    # inference_original()
