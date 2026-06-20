from flask import Flask, request, Response
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pywhisper

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("./model")
model = GPT2LMHeadModel.from_pretrained("./model").to(device)


user_input = ""
app = Flask(__name__)

def run_model():
    step = 0
    whisper_model = pywhisper.load_model("base")
    result = whisper_model.transcribe("web-audio.wav")

    user_input = str(result["text"])
    
    print("USER INPUT: " + user_input)

    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 and step < 6 else new_user_input_ids
    
    if step > 5:
        step = 5

    chat_history_ids = model.generate(
    bot_input_ids, max_length=1000,
    pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    print("SheldonBot: {}".format(result))
    step += 1

    return result

@app.route("/uploadAudio", methods=['POST', 'GET'])
def recordAudio():
    if request.method == "POST":
        f = request.files['audio-data']
        with open('web-audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

    result = run_model()
    
    response = Response(str(result))
    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

if __name__ == "__main__":
    app.run(debug=True)
    