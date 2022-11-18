from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("./model")
model = GPT2LMHeadModel.from_pretrained("./model")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)

print(generation_output[:2])
print(tokenizer.decode(generation_output[0], skip_special_tokens=True))