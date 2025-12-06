# @title Resource requirements for LLM inference
import os
import requests
import json
from huggingface_hub import snapshot_download, model_info

class ModelConfigMemUsage:
    def __init__(self, huggingface_model_name: str, model_params_billion: float, avg_input_length: int, 
                 avg_output_length: int, max_num_sequences: int):
        
        self.huggingface_model_name = huggingface_model_name
        self.model_params_billion = model_params_billion
        self.avg_input_length = avg_input_length
        self.avg_output_length = avg_output_length
        self.max_num_sequences = max_num_sequences
        self.model_weight_memory = 0.0
        self.total_memory_usage = 0.0

def get_model_size(model_name: str, model_params_billion: float, avg_input_length: int, 
                   avg_output_length: int, max_num_sequences: int, huggingface_api_key: str = None) -> ModelConfigMemUsage:
  ConfigMemUsage = ModelConfigMemUsage(
      model_name, model_params_billion, avg_input_length, avg_output_length, max_num_sequences
  )
  # --- Step 1: Fetch Model Configuration ---
  print(f"--- Fetching configuration for {model_name} ---")

  # Create a directory to store the config files
  os.makedirs("config_files", exist_ok=True)
  # Sanitize model name for use in filename (replace '/' with '--')
  safe_model_name = model_name.replace("/", "--")
  config_path = os.path.join("config_files", f"{safe_model_name}_config.json")

  # Download the model's config.json file
  config_url = f"https://huggingface.co/{model_name}/resolve/main/config.json?download=true"
  headers = {"Authorization": f"Bearer {huggingface_api_key}"} if huggingface_api_key else {}
  response = requests.get(config_url, headers=headers)
  response.raise_for_status()

  with open(config_path, "w") as f:
      f.write(response.text)

  with open(config_path, "r") as f:
    data = json.load(f)

  # Check for a nested text_config, common in multimodal models.
  if "text_config" in data and isinstance(data["text_config"], dict):
      config_source = data["text_config"]
      print("Using nested 'text_config' for model parameters.")
  else:
      config_source = data
      print("Using top-level config for model parameters.")

  # Use .get() for safe dictionary access to extract model architecture details
  hidden_size = config_source.get('hidden_size')
  num_hidden_layers = config_source.get('num_hidden_layers')
  num_attention_heads = config_source.get('num_attention_heads')
  # Some models use 'ffn_dim' (e.g., OPT) instead of 'intermediate_size'
  intermediate_size = config_source.get('intermediate_size') or config_source.get('ffn_dim')
  num_kv_heads = config_source.get('num_key_value_heads', num_attention_heads)

  # Ensure all required parameters were found
  required_params = [hidden_size, num_hidden_layers, num_attention_heads, intermediate_size]
  if not all(required_params):
      raise ValueError("One or more required model parameters (e.g., hidden_size, num_hidden_layers) could not be found in the config.")

  head_dims = hidden_size // num_attention_heads
  dtype = data.get('torch_dtype', 'bfloat16')

  # Determine data type size in bytes based on model's precision (dtype)
  match dtype:
    case 'float16' | 'bfloat16':
      parameter_data_type_size = 2
      kv_data_type_size = 2
    case 'float32':
      parameter_data_type_size = 4
      kv_data_type_size = 4
    case _: # Default to bfloat16 if not specified
      parameter_data_type_size = 2
      kv_data_type_size = 2

  # --- Step 2: Calculate Required xPU Memory ---
  print("\n--- Calculating Required GPU Memory ---")
  print(f"0. Model Parameters (in billions): {model_params_billion}")
  # Component 1: Model Weight Memory (Static)
  # Memory needed to load the model's parameters onto the xPU.
  number_of_model_parameters = model_params_billion * 1e9
  model_weight_bytes = number_of_model_parameters * parameter_data_type_size
  model_weight_gb = model_weight_bytes / (1000**3)
  print(f"1. Model Weight Memory: {model_weight_gb:.2f} GB")

  # Component 2: Overhead Memory (Static)
  # Fixed memory for non-PyTorch components like CUDA kernels, etc.
  overhead_memory_gb = 1.0
  print(f"2. System Overhead Memory: {overhead_memory_gb:.2f} GB")

  # Component 3: PyTorch Activation Peak Memory (Dynamic)
  # Memory for storing intermediate calculations (activations) during the forward pass.
  # This scales with max_num_sequences and sequence length.
  sequence_length = avg_input_length + avg_output_length
  pytorch_activation_peak_memory_bytes = max_num_sequences * sequence_length * (18 * hidden_size + 4 * intermediate_size)
  pytorch_activation_peak_memory_gb = pytorch_activation_peak_memory_bytes / (1000**3)
  print(f"3. PyTorch Activation Memory (for max_num_sequences size {max_num_sequences}): {pytorch_activation_peak_memory_gb:.2f} GB")

  # Component 4: KV Cache Memory (Dynamic)
  # Memory for the Key-Value cache, which stores attention context to speed up token generation.
  # This scales with max_num_sequences and sequence length.
  kv_vectors = 2 # One for Key, one for Value
  kv_cache_memory_per_batch_bytes = kv_vectors * max_num_sequences * sequence_length * num_kv_heads * head_dims * num_hidden_layers * kv_data_type_size
  kv_cache_memory_per_batch_gb = kv_cache_memory_per_batch_bytes / (1000**3)
  print(f"4. KV Cache Memory (for max_num_sequences {max_num_sequences}): {kv_cache_memory_per_batch_gb:.2f} GB")

  # --- Final Calculation ---
  print("\n--- Total Memory Calculation ---")
  # Sum of static and dynamic memory components.

  ConfigMemUsage.total_memory_usage = (
      model_weight_gb +
      overhead_memory_gb +
      pytorch_activation_peak_memory_gb +
      kv_cache_memory_per_batch_gb
  )

  ConfigMemUsage.model_weight_memory = model_weight_gb

  print("\n-------------------------------------")
  print(f"✅ Required GPU Memory: {ConfigMemUsage.total_memory_usage:.2f} GB")
  print("-------------------------------------")
  print(f"\nThis is the estimated total GPU VRAM needed to run inference for a max_num_sequences of {ConfigMemUsage.max_num_sequences} with the specified model and sequence lengths.")
  print("\n-------------------------------------")
  print("NOTE: IF YOU DIDN'T SPECIFY OR CHANGE MODEL PARAM SIZE, THE RESULTS WILL BE WRONG!!!!!")
  print("-------------------------------------")
  return ConfigMemUsage

if __name__ == "__main__":
  huggingface_api_key = os.getenv("hf_token")# @param {type:"string"}
  huggingface_model_name = "facebook/opt-125m" # @param {type:"string"}
  model_parameters_in_billions = 0.125 # @param {type:"number"}
  avg_input_length = 1500 # @param {type:"integer"}
  avg_output_length = 200 # @param {type:"integer"}
  max_num_sequences = 10000 # @param {type:"integer"}

  get_model_size(
      huggingface_model_name,
      model_parameters_in_billions,
      avg_input_length,
      avg_output_length,
      max_num_sequences
  )