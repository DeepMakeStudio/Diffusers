import sys

plugin = {
    "Name": "Stable Diffusion (diffusers)",
    "Version": "0.1.0", 
    "Author": "DeepMake",
    "Description": "Stable Diffusion generation using the Diffusers library",
    "env": "sd"
}
config = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    "model_dtype": "fp32" if sys.platform == "darwin" else "fp16",
    "loras_path": "loras",
    "textual_embedding_path": "ti",
    "negative_prompt_prefix": "NSFW, nudity, low quality, ",
    "prompt_prefix": "",
    "scheduler": "pndm",
    "controlnet": "lllyasviel/sd-controlnet-canny",
    "schedulers": [
        "pndm", "dpm", "euler", "heun", "lms",
        "dpm_2", "dpm_2_ancestral", "dpm_fast", "dpm_adaptive",
        "dpmpp_2s_ancestral", "dpmpp_2m"
    ]
}
endpoints = {
    "generate_image": {
        "call": "execute",
        "method": "PUT",
        "inputs": {
            "prompt": "Text",
            "seed": "Int(default=None, optional=true)",
            "iterations": "Int(default=20, min=1, optional=true)",
            "height": "Int(default=512, min=16, optional=true)",
            "width": "Int(default=512, min=16, optional=true)",
            "negative_prompt": "Text(default=None, optional=true)",
            "scheduler": "Text(default='pndm', optional=true)"
        },
        "outputs": {"output_img": "Image"}
    },
    "refine_image": {
        "call": "execute2",
        "method": "PUT",
        "inputs": {
            "prompt": "Text",
            "seed": "Int(default=None, optional=true)",
            "img": "Image",
            "iterations": "Int(default=20, min=1, optional=true)",
            "height": "Int(default=512, min=16, optional=true)",
            "width": "Int(default=512, min=16, optional=true)",
            "strength": "Float(default=0.75, min=0.0, max=1.0, optional=true)",
            "negative_prompt": "Text(default=None, optional=true)",
            "scheduler": "Text(default='pndm', optional=true)"
        },
        "outputs": {"output_img": "Image"}
    }
}
