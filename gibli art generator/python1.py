from diffusers import StableDiffusionImg2ImgPipeline
import torch

# Load the model
model_id = "nitrosocke/Ghibli-Diffusion"  # Use the correct model ID
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")