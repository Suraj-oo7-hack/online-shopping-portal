from PIL import Image
import matplotlib.pyplot as plt
import time

def generate_ghibli_image(image, pipe, strength):
    image = image.convert("RGB").resize((512, 512))  # Resize image
    prompt = "Ghibli-style anime painting, soft pastel colors, highly detailed, masterpiece"
    
    start_time = time.time()
    result = pipe(prompt=prompt, image=image, strength=strength).images[0]
    print(f"Image generated in {time.time() - start_time:.2f} seconds!")
    return result