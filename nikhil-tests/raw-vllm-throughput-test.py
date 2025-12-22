import os
import subprocess
import time
import json
import requests
import math
import csv
import signal

models = [
    "Qwen/Qwen1.5-MoE-A2.7B", 
    "microsoft/Phi-tiny-MoE-instruct", 
    "microsoft/Phi-3.5-MoE-instruct", 
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
]

# Benchmark parameters
num_runs = 5           # Number of benchmark runs per model
max_new_tokens = 32    # 32 default
default_port = 30000
output_csv_file_path = "/home/azureuser/vllm/nikhil-tests/base-vllm-with-ep.csv"
host = "127.0.0.1"

# Results storage
all_results = []

# CSV field names
FIELDNAMES = ["model", "run", "runtime_s", "throughput_tokens_s", "num_prompts", "max_new_tokens", "num_finished", "success"]


def load_questions(filename):
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            questions.append(obj)
    return questions


def check_status(host: str = host, port: int = default_port, timeout: int = 300, delay=5):
    """check the /health endpoint to see if the server is running"""
    url = f"http://{host}:{port}/health"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return "OK"
            else:
                return "ERROR"
        except:
            time.sleep(delay)
    return None


def save_result_to_csv(result_entry, filepath, write_header=False):
    """Append a single result to CSV (creates file with header if needed)"""
    mode = "w" if write_header else "a"
    with open(filepath, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(result_entry)


def kill_server(server_proc, max_wait=30):
    """Kill the server and all its child processes, then verify GPU is freed"""
    print("Killing server process tree...")
    
    # Kill the entire process group (server + all children)
    try:
        pgid = os.getpgid(server_proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(2)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass  # Process already dead
    
    # Kill any remaining vllm processes aggressively
    subprocess.run("pkill -9 -f 'vllm serve' 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f 'EngineCore' 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f 'from multiprocessing' 2>/dev/null", shell=True)
    time.sleep(2)
    
    # Wait for GPU memory to be freed
    print("Waiting for GPU memory to be freed...")
    for i in range(max_wait):
        result = subprocess.run(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            # Get memory used on first GPU (in MiB)
            mem_used = int(result.stdout.strip().split('\n')[0])
            # Consider memory freed if less than 1GB used
            if mem_used < 1000:
                print(f"GPU memory freed ({mem_used} MiB used)")
                return True
        time.sleep(1)
    
    print(f"WARNING: GPU memory may not be fully freed after {max_wait}s")
    return False


def run_benchmark(model, url, prompts, max_new_tokens, run_num):
    """Run a single benchmark iteration and return results dict"""
    print(f"\n--- Run {run_num} ---")
    
    start = time.time()
    try:
        response = requests.post(
            url + "/v1/completions",
            json={
                "model": model,
                "prompt": prompts,
                "temperature": 0,
                "max_tokens": max_new_tokens,
            },
            timeout=600,
        )
        end = time.time()
        
        num_finished = 0
        success = False
        
        # Parse OpenAI-format response
        result = response.json()
        if response.status_code == 200 and 'choices' in result:
            for choice in result['choices']:
                if choice.get('finish_reason') == 'length':
                    num_finished += 1
            success = True
        else:
            print(f"Error response: {result}")
            
    except Exception as e:
        end = time.time()
        print(f"Error during benchmark: {e}")
        num_finished = 0
        success = False
    
    runtime = end - start
    throughput = len(prompts) * max_new_tokens / runtime if runtime > 0 else 0
    
    print(f"Time: {runtime:.3f}s | Throughput: {throughput:.3f} tokens/s")
    
    return {
        "model": model,
        "run": run_num,
        "runtime_s": round(runtime, 3),
        "throughput_tokens_s": round(throughput, 3),
        "num_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "num_finished": num_finished,
        "success": success,
    }


# Load questions
os.chdir("/home/azureuser/artifacts/benchmarks/mtbench/")
questions = load_questions("./question.jsonl")

# Prepare prompts
prompts = []
for i in range(math.ceil(324 * 14 // 80)):
    for question in questions:
        prompts.append(question["turns"][0])

print(f"Loaded {len(prompts)} prompts")
print(f"Running {num_runs} benchmark runs per model")
print(f"Results will be saved to: {output_csv_file_path}")

# Initialize CSV with header
first_result = True

for model in models:
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")
    
    # Clean up any leftover processes before starting
    subprocess.run("pkill -9 -f 'vllm serve' 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f 'EngineCore' 2>/dev/null", shell=True)
    time.sleep(3)
    
    # Start server with new process group so we can kill all children
    print(f"Starting server...")
    command = f"vllm serve {model} --port {default_port} --host {host} --enable-expert-parallel --all2all-backend allgather_reducescatter"
    server_proc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    
    print(f"Waiting for server to be ready...")
    if check_status(host, default_port) != "OK":
        # Kill the entire process group
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
        except:
            pass
        print(f"ERROR: Server failed to start for model {model}")
        time.sleep(5)
        continue
    
    url = f"http://{host}:{default_port}"
    
    # Warmup
    print(f"Running warmup...")
    try:
        warmup = requests.post(
            url + "/v1/completions",
            json={
                "model": model,
                "prompt": prompts[:50],
                "temperature": 0,
                "max_tokens": 2,
            },
            timeout=60,
        )
        print(f"Warmup status: {warmup.status_code}")
    except Exception as e:
        print(f"Warmup failed: {e}")
    
    # Run benchmark n times
    for run_num in range(1, num_runs + 1):
        result_entry = run_benchmark(model, url, prompts, max_new_tokens, run_num)
        all_results.append(result_entry)
        
        # Save to CSV immediately (write header only on first result)
        save_result_to_csv(result_entry, output_csv_file_path, write_header=first_result)
        first_result = False
    
    # Cleanup: kill the server and all child processes
    print(f"\nStopping server for {model}...")
    kill_server(server_proc)

# Print summary
print(f"\n{'='*60}")
print("=== Results Summary ===")
print(f"{'='*60}")

# Group results by model and calculate averages
from collections import defaultdict
model_results = defaultdict(list)
for r in all_results:
    if r['success']:
        model_results[r['model']].append(r['throughput_tokens_s'])

for model, throughputs in model_results.items():
    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
    print(f"{model}:")
    print(f"  Runs: {len(throughputs)}")
    print(f"  Avg Throughput: {avg_throughput:.3f} tokens/s")
    print(f"  Min: {min(throughputs):.3f} | Max: {max(throughputs):.3f}")

print(f"\nResults saved to: {output_csv_file_path}")
