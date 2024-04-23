import sys

plugin = {
    "Name": "Stable Diffusion (diffusers)",
    "Version": "0.1.0", 
    "Author": "DeepMake",
    "Description": "Stable Diffusion generation using the Diffusers library",
    "env": "sd",
    "memory": 4000,
    "model_memory": 3000
}
config = {
    "model_name": "runwayml/stable-diffusion-v1-5", # "options": ["runwayml/stable-diffusion-v1-5", "stabilityai/sdxl-turbo"]},
    "model_dtype": "fp32" if sys.platform == "darwin" else "fp16",
    "loras": [],
    "inverters": [],
    "scheduler": "pndm",
    "controlnet": "lllyasviel/sd-controlnet-canny"
}
endpoints = {
    "generate_image": {
        "call": "execute",
        "inputs": {
            "prompt": "Text",
            "seed": "Int(default=None, optional=true)",
            "iterations": "Int(default=20, min=1, optional=true)",
            "height": "Int(default=512, min=16, optional=true)",
            "width": "Int(default=512, min=16, optional=true)"
        },
        "outputs": {"output_img": "Image"}
    },
    "refine_image": {
        "call": "execute2",
        "inputs": {
            "prompt": "Text",
            "seed": "Int(default=None, optional=true)",
            "img": "Image",
            "iterations": "Int(default=20, min=1, optional=true)",
            "height": "Int(default=512, min=16, optional=true)",
            "width": "Int(default=512, min=16, optional=true)",
            "strength": "Float(default=0.75, min=0.0, max=1.0, optional=true)"
        },
        "outputs": {"output_img": "Image"}
    }
}
