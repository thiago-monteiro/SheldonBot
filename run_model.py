from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import queue
import sys
import os

device = "xpu"

tokenizer = GPT2Tokenizer.from_pretrained("./model")
model = GPT2LMHeadModel.from_pretrained("./model").to(device)

step = 0
user_input = ""

while user_input.lower() != "done":
    user_input = str(input("User: "))
    
    if user_input.lower() != "done":

        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 and step < 6 else new_user_input_ids
        
        if step > 5:
            step = 5
    
        chat_history_ids = model.generate(
        bot_input_ids, max_length=1000,
        pad_token_id=tokenizer.eos_token_id
        )

        print("SheldonBot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        step += 1