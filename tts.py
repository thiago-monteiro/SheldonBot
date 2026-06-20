import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"

print(TTS().list_models())

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts.tts_to_file(text="Hello world!", speaker_wav="sheldon-voice-sample-1min-modified.mp3", language="en", file_path="output.wav")