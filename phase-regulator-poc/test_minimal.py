from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def test():
    print("Loading GPT2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
    print("Generating...")
    inputs = tokenizer("Hello", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10)
    print("Done:", tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    test()
