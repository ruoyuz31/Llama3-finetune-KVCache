#print the results in a txt file
import pickle
import numpy as np
# Load the inference.pkl from the pickle file
with open("inference.pkl", "rb") as f:
    records = pickle.load(f)
with open("inference_results.txt", "w",encoding="utf-8") as f:
    for record in records:
        # add a header for each record
        f.write("***********************************\n")
        f.write(f"************* record {records.index(record)} ************\n")
        f.write("***********************************\n")


        f.write(f"KV Caching: {record['kv_caching']}\n")
        for prompt, result in zip(record['prompts'], record['results']):
            f.write(prompt + "\n")
            f.write(f"> {result['generation']}\n")
            f.write("\n==================================\n")



# load the benchmark_results.pkl from the pickle file
filename = "benchmark_phase2_results_20250427-001740"
with open(filename+".pkl", "rb") as f:
    records = pickle.load(f)

# print records in a txt file
with open(filename+".txt", "w") as f:
    for record in records:
        f.write(f"Batch size: {record['batch_size']}\n")
        f.write(f"Input length: {record['input_len']} tokens\n")
        f.write(f"Output length: {record['output_len']} tokens\n")
        f.write(f"Inference time: {record['inference_time']:.2f} seconds\n")
        f.write(f"Tokens per second: {record['tokens_per_second']:.2f}\n")
        f.write(f"Model weights memory usage: {record['model_memory_mb']:.2f} MB\n")
        f.write(f"Peak memory usage: {record['peak_memory_mb']:.2f} MB\n")
        f.write(f"KV Caching: {record['kv_caching']}\n")
        f.write("\n=== Sample Output ===\n")
        f.write(record["generation"] + "\n\n")
# print the inference time and peak memory usage for the same batch size and kv_caching in a table of txt file
# calculate the average inference time and peak memory usage for each batch size and kv_caching

results = []
for batch_size in [1, 8, 16]:
    for kv_caching in [False, True]:
        batch_records = [record for record in records if record['batch_size'] == batch_size and record['kv_caching'] == kv_caching]
        if batch_records:
            avg_inference_time = np.mean([record['inference_time'] for record in batch_records])
            avg_peak_memory = np.mean([record['peak_memory_mb'] for record in batch_records])
            # calculate the standard deviation of the inference time and peak memory usage
            std_inference_time = np.std([record['inference_time'] for record in batch_records])
            std_peak_memory = np.std([record['peak_memory_mb'] for record in batch_records])
            results.append({
                "batch_size": batch_size,
                "kv_caching": kv_caching,
                "inference_time": avg_inference_time,
                "peak_memory_mb": avg_peak_memory,
                "std_inference_time": std_inference_time,
                "std_peak_memory": std_peak_memory,
            })

with open(filename+"_table.txt", "w") as f:
    f.write(f"{'Batch Size':<10} {'KV Caching':<15} {'Inference Time (s)':<20} {'Peak Memory (MB)':<20}\n")
    f.write("="*70 + "\n")
    for record in records:
        f.write(f"{record['batch_size']:<10} {str(record['kv_caching']):<15} {record['inference_time']:<20.2f} {record['peak_memory_mb']:<20.2f}\n")
    
    # write the average and std of inference time and peak memory usage for each batch size and kv_caching in a table
    f.write("\n\nAverage and Standard Deviation of Inference Time and Peak Memory Usage:\n")
    f.write(f"{'Batch Size':<10} {'KV Caching':<15} {'Avg Inference Time (s)':<20} {'Std Inference Time (s)':<20} {'Avg Peak Memory (MB)':<20} {'Std Peak Memory (MB)':<20}\n")
    f.write("="*140 + "\n")
    for result in results:
        f.write(f"{result['batch_size']:<10} {str(result['kv_caching']):<15} {result['inference_time']:<20.2f} {result['std_inference_time']:<20.2f} {result['peak_memory_mb']:<20.2f} {result['std_peak_memory']:<20.2f}\n")


