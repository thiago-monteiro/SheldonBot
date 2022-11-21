from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("./model")
model = GPT2LMHeadModel.from_pretrained("./model").to(device)

step = 0
user_input = ""

while user_input != "done":
    user_input = input('>> User (type "done" to exit): ')
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 and step < 6 else new_user_input_ids
    
    if step > 5:
        step = 0

    # generated a response while limiting the total chat history to 1000 tokens    
    chat_history_ids = model.generate(
    bot_input_ids, max_length=1000,
    pad_token_id=tokenizer.eos_token_id
    )

    # pretty print last ouput tokens from bot
    print("SheldonBot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

    step += 1
