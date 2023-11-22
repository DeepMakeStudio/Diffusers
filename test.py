from diffusers import DiffusionPipeline
import torch
from safetensors.torch import load_file

# state_dict = load_file("Ana_de_Armas.safetensors")
# print(state_dict.keys())
# exit()
print("Loading model...")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda")
print("Loading weights...")
pipe.load_textual_inversion("sd-concepts-library/cat-toy")
# pipe.load_lora_weights(".", weight_name="Victo_Ngai_Style.safetensors")
print("Generating image...")

prompt = "a <cat toy> backpack"
generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
image.save("cat backpack.png")

