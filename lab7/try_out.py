import sys
# do not remove, otherwise needed libraries will be missing
sys.path.append('/opt/python3.10/site-packages')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from argparse import ArgumentParser

# Parse argument
parser = ArgumentParser()

parser.add_argument("-m", "--model-path", help="model name or path", dest="model_name_or_path", default="gpt2")
parser.add_argument("-p", "--prompt", help="prompt to the model", dest="prompt", default="Who is Jerry Chou in NTHU?")

args = parser.parse_args()

def main(model_name_or_path: str, prompt: str):
  # use GPU if possible
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # Set seed before initializing model.
  set_seed(42)

  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
  model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
  inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
  outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
  print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

if __name__ == "__main__":
  model_name_or_path = args.model_name_or_path
  prompt = args.prompt

  main(model_name_or_path, prompt)
