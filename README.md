Running a Local LLM with Hugging Face Transformers
This repository is part of a YouTube tutorial demonstrating how to run a local large language model (LLM) using the Hugging Face Transformers library. The provided Python script shows how to load a pre-trained causal language model from a local checkpoint, format a chat-style input, generate a response, and decode the output.

Overview
The example script (run-llm.py) demonstrates:

Model & Tokenizer Loading: How to load a local model checkpoint.
Input Formatting: How to format a chat-based message.
Tokenization: Converting text into tokens along with an attention mask.
Text Generation: Using the model to generate a response with sampling parameters.
Output Decoding: Decoding the generated tokens back into human-readable text.
This tutorial is intended for educational purposes, showing the step-by-step process to run a local LLM.

Prerequisites
Python 3.7+
PyTorch: Install the CPU or CUDA-enabled version based on your hardware.
Hugging Face Transformers: For loading and interacting with the model.
Installation
Clone the repository:
bash
Copy
git clone https://github.com/portalbh/SmolLM2.git
Navigate into the repository:
bash
Copy
cd yourrepository
(Optional) Create and activate a virtual environment:
On Windows:
bash
Copy
python -m venv venv
venv\Scripts\activate
On macOS/Linux:
bash
Copy
python3 -m venv venv
source venv/bin/activate
Install required dependencies:
bash
Copy
pip install torch transformers
Setup
Model Checkpoint:
Ensure that your local model checkpoint is stored at the path specified in the script (by default: C:\Repo\Local-LLMs\SmolLm2\SmolLM2-1.7B-Instruct). If your checkpoint is located elsewhere, update the checkpoint variable in run-llm.py.
Usage
Run the script using Python:

bash
Copy
python run-llm.py
The script will:

Load the model and tokenizer.
Format a chat input asking: "What is the capital of Bahrain."
Tokenize the input and generate a response using the specified generation parameters.
Decode and print the response to the console.
Code Walkthrough
Here's an overview of the key sections in the code:

python
Copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the path to your local model checkpoint
checkpoint = "C:\\Repo\\Local-LLMs\\SmolLm2\\SmolLM2-1.7B-Instruct"
device = "cpu"  # Change to "cuda" if you have a compatible GPU

# Load the tokenizer and model from the checkpoint directory
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Prepare a chat message in the required format
messages = [{"role": "user", "content": "What is the capital of Bahrain."}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)

# Tokenize the input and generate both input_ids and attention_mask
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate a response from the model with defined sampling parameters
outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
    temperature=0.2,
    top_p=0.9,
    do_sample=True
)

# Decode the generated tokens into a readable string and print the output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Key Points:
Device Setup: Use "cuda" if a GPU is available to accelerate inference.
Attention Mask: Ensures that the model knows which tokens are actual input versus padding.
Sampling Parameters: Adjust max_new_tokens, temperature, and top_p to control the generated text's length and randomness.
Contributing
Contributions, suggestions, and bug reports are welcome! Feel free to fork the repository and open a pull request if you have improvements or additional examples.

License
This project is licensed under the MIT License. See the LICENSE file for details.

YouTube
https://www.youtube.com/@BahrainAI
