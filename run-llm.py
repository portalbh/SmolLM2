# Import the necessary libraries
import torch  # PyTorch, used for tensor operations and model computations
from transformers import AutoModelForCausalLM, AutoTokenizer  # Hugging Face libraries for model and tokenizer management

# Specify the local directory where your model checkpoint is saved.
# This directory should contain the model's configuration, weights, and tokenizer files.
checkpoint = "C:\\Repo\\Local-LLMs\\SmolLm2\\SmolLM2-1.7B-Instruct"

# Set the device to "cpu" or "cuda" depending on whether you're using a CPU or a GPU.
# Using a GPU (with "cuda") can significantly speed up inference for large models.
device = "cpu"  # Change to "cuda" if you have a compatible GPU

# Load the tokenizer from the checkpoint.
# The tokenizer converts raw text into token IDs that the model can understand.
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the pre-trained causal language model from the checkpoint.
# The .to(device) method moves the model to the chosen device (CPU or GPU).
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Define the input message in a chat format.
# The model expects a list of messages with roles (e.g., "user", "assistant").
messages = [{"role": "user", "content": "What is the capital of Bahrain."}]

# Format the conversation using the tokenizer's chat template.
# The method apply_chat_template structures the input in a way that the model was trained to understand.
# Setting tokenize=False returns a raw string instead of token IDs.
input_text = tokenizer.apply_chat_template(messages, tokenize=False)

# Tokenize the input text to convert it into a format suitable for the model.
# Using return_tensors="pt" returns the output as PyTorch tensors, which are required for model inference.
# The returned dictionary includes:
#   - "input_ids": the numerical representation of the tokens
#   - "attention_mask": indicates which tokens should be attended to (1) or ignored (0, e.g., padding)
inputs = tokenizer(input_text, return_tensors="pt")

# Move the token IDs to the chosen device (CPU or GPU)
input_ids = inputs["input_ids"].to(device)

# Move the attention mask to the device.
# The attention mask helps the model distinguish between real tokens and padding.
attention_mask = inputs["attention_mask"].to(device)

# Generate a response using the model.
# Explanation of parameters:
#   - input_ids: The encoded input text
#   - attention_mask: Guides the model on which tokens to focus on
#   - max_new_tokens=50: Limits the response to 50 new tokens
#   - temperature=0.2: Controls randomness; lower values produce more deterministic outputs
#   - top_p=0.9: Nucleus sampling parameter, limiting the token selection to the top 90% cumulative probability
#   - do_sample=True: Enables sampling, which introduces variability in the generated response
outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
    temperature=0.2,
    top_p=0.9,
    do_sample=True
)

# Decode the generated tokens back into a human-readable string.
# skip_special_tokens=True removes special tokens (like end-of-sequence markers) from the output.
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
