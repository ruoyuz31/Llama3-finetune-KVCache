from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama
import torch
import os

def inference(kv_caching=True):
    torch.manual_seed(1)

    checkppoint_dir = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B")
    tokenizer_path = os.path.join(checkppoint_dir, "tokenizer.model")
    model_path = os.path.join(checkppoint_dir, "consolidated.00.pth")

    tokenizer = Tokenizer(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    # # RZ: map to GPU
    # if torch.cuda.is_available():
    #     checkpoint = torch.load(model_path, map_location="cuda:0")
    model_args = ModelArgs()
    torch.set_default_tensor_type(torch.cuda.HalfTensor) # load model in fp16
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    device = "cuda"
    model.to(device)
    
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    model.eval()
    results = model.generate(tokenizer, prompts, max_gen_len=64, temperature=0.6, top_p=0.9, kv_caching=kv_caching, device=device)
    
    # RZ: check where the model is running
    device = next(model.parameters()).device
    print(f"Model is running on {device}")

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

    record = {
        "kv_caching": kv_caching,
        "prompts": prompts,
        "results": results,
    }

    return record

    
if __name__ == "__main__":
    #inference(kv_caching=False)
    
    records = []
    for kv_caching in [True, False]:
        for i in range(5):
            record = inference(kv_caching)
            print(f"KV Caching: {kv_caching}")
            records.append(record)
    # save the results to a file
    import pickle
    with open("inference.pkl", "wb") as f:
        pickle.dump(records, f)
    # print the results in a txt file
    # with open("inference.txt", "w") as f:
    #     for record in records:
    #         f.write(f"KV Caching: {record['kv_caching']}\n")
    #         for prompt, result in zip(record['prompts'], record['results']):
    #             f.write(prompt + "\n")
    #             f.write(f"> {result['generation']}\n")
    #             f.write("\n==================================\n")