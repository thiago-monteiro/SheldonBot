import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

prompt = "Sheldon Cooper eating rice"
image = pipe(prompt).images[0]  
    
image.save("sheldonbot-pose.png")