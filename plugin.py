import os
from argparse import Namespace
from fastapi import FastAPI
from PIL import Image

from io import BytesIO
import torch
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline,AutoPipelineForText2Image,AutoPipelineForImage2Image, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, PNDMScheduler, StableDiffusionInpaintPipeline, ControlNetModel
import threading
import time
import psutil
import sys
from safetensors.torch import load_file
from collections import defaultdict
from compel import Compel, ReturnedEmbeddingsType
from plugin import Plugin, fetch_image, store_image
from .config import plugin, config, endpoints
import numpy as np
from .download_gui import LoadingBar
from huggingface_hub import scan_cache_dir
app = FastAPI()

def check_model():
    if 'sd_plugin' not in globals():
        set_model()

@app.get("/get_info/")
def plugin_info():
    check_model()
    return sd_plugin.plugin_info()

@app.get("/get_config/")
def get_config():
    check_model()
    return sd_plugin.get_config()

@app.put("/set_config/")
def set_config(update: dict):
    sd_plugin.set_config(update) # TODO: Validate config dict are all valid keys
    return sd_plugin.get_config()

@app.on_event("startup")
async def startup_event():
    print("Starting up")
    # A slight delay to ensure the app has started up.
    try:
        set_model()
        print("Successfully started up")
        sd_plugin.notify_main_system_of_startup("True")
    except:
        sd_plugin.notify_main_system_of_startup("False")

@app.get("/set_model/")
def set_model():
    global sd_plugin
    args = {"plugin": plugin, "config": config, "endpoints": endpoints}
    sd_plugin = SD(Namespace(**args))
    sd_plugin.set_model()
    # try:
    # sd_plugin.set_model(args["model_name"], dtype=args["model_dtype"])
    model_name = sd_plugin.config["model_name"]
    return {"status": "Success", "detail": f"Model set successfully to {model_name}"}

@app.get("/execute/{prompt}")
def execute(prompt: str, seed: int = None, iterations: int = 20, height: int = 512, width: int = 512, guidance_scale: float = 7.0, control_image: str = None):
    # check_model()
    if control_image is None:
        im = sd_plugin._predict(prompt, seed=seed, iterations=iterations, height=height, width=width, guidance_scale=guidance_scale)
    else:
        imagebytes = fetch_image(control_image)
        control_image = Image.open(BytesIO(imagebytes))
        im = sd_plugin.controlnet_predict(prompt, control_image, seed=seed)

    output = BytesIO()
    im.save(output, format="PNG")
    image_id = store_image(output.getvalue())

    return {"status": "Success", "output_img": image_id}

@app.get("/execute2/{text}/{img}")
def execute2(text: str, img: str, seed = None, iterations: int = 20, height: int = 512, width: int = 512, guidance_scale: float = 7.0, strength: float = 0.75):
    # check_model()

    imagebytes = fetch_image(img)
    image = Image.open(BytesIO(imagebytes))
    image = image.convert("RGB")
    im = sd_plugin.img_to_img_predict(text, image, seed=seed, iterations=iterations, height=height, width=width, guidance_scale=guidance_scale, strength=strength)
    output = BytesIO()
    im.save(output, format="PNG")
    image_id = store_image(output.getvalue())

    return {"status": "Success", "output_img": image_id}

def self_terminate():
    time.sleep(3)
    parent = psutil.Process(psutil.Process(os.getpid()).ppid())
    print(f"Killing parent process {parent.pid}")
    # os.kill(parent.pid, 1)
    # parent.kill()

@app.get("/shutdown/")  #Shutdown the plugin
def shutdown():
    threading.Thread(target=self_terminate, daemon=True).start()
    return {"success": True}

class SD(Plugin):
    """
    Prediction inference.
    """
    def __init__(self, arguments: "Namespace") -> None:
        super().__init__(arguments, plugin_name = "Diffusers")

    def load_lora_weights(self, pipeline, checkpoint_path, multiplier=1):
        if self.type == "xl":
            print("Warning: LoRA weights are not yet working for XL models")
            return pipeline
        original_dtype = pipeline.unet.dtype
        pipeline = pipeline.to("cpu", torch.float32)
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        # load LoRA weight from .safetensors
        state_dict = load_file(checkpoint_path)

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        # directly update weight in diffusers model
        for layer, elems in updates.items():

            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipeline.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight'].to(torch.float32)
            weight_down = elems['lora_down.weight'].to(torch.float32)
            alpha = elems['alpha']
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                # curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                # if weight_down.shape[3] == 3:StableDiffusi),weight_down)
                updated_weights = multiplier * alpha * weights_mm
                #torch.mm(weight_up.squeeze(3).squeeze(2),weight_down[:,:,0,0]).unsqueeze(2).unsqueeze(3) # I know this hack is wrong, but it gets close enough
            else:
                # curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
                updated_weights = multiplier * alpha * torch.mm(weight_up, weight_down)

            # print(f"Updating {layer} with {updated_weights.shape}")
            curr_layer.weight.data += updated_weights

        return pipeline.to("cpu", original_dtype)

    def load_loras(self, pipeline):
        for lora, multiplier in config['loras']:
            self.load_lora_weights(self.tti, lora, multiplier=multiplier)
        return pipeline
    
    def on_install(self, progress_callback=None):
        set_model()

    def set_model(self) -> None:
        """
        Load given weights into model.
        """
        model_path = self.config["model_name"].strip()
        dtype = self.config["model_dtype"]
        cached_models = scan_cache_dir()
        if not any(model_path in repo_info.repo_id for repo_info in cached_models.repos) and not os.path.exists(model_path):
            load_bar = LoadingBar()
            load_bar.start(model_path, torch.float32 if dtype == "fp32" else torch.float16, dtype)

        if os.path.exists(model_path):
            if "xl" in model_path.lower():
                self.type = "xl"
                self.tti = DiffusionPipeline.from_single_file(model_path,
                                                              torch_dtype=torch.float32 if dtype == "fp32" else torch.float16,
                                                              variant=dtype)
            else:
                self.type = "sd"
                self.tti = StableDiffusionPipeline.from_single_file(model_path,
                                                                    torch_dtype=torch.float32 if dtype == "fp32" else torch.float16,
                                                                    variant=dtype)
        else:
            if "xl" in model_path.lower():
                self.type = "xl"
                # self.tti = StableDiffusionXLPipeline.from_pretrained(model_path,
                #                                                      torch_dtype=torch.float32 if dtype == "fp32" else torch.float16,
                #                                                      variant=dtype)
            else:
                self.type = "sd"
                # self.tti = StableDiffusionPipeline.from_pretrained(model_path,
                #                                                    torch_dtype=torch.float32 if dtype == "fp32" else torch.float16,
                #                                                    variant=dtype)
            self.tti = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=torch.float32 if dtype == "fp32" else torch.float16, variant=dtype)
        if self.config["scheduler"] == "pndm":
            pass
        elif self.config["scheduler"] == "dpm":
            self.tti.scheduler = DPMSolverMultistepScheduler.from_config(self.tti.scheduler.config)
        else:
            print("Warning: Unknown scheduler. Using PNDM")

        self.iti = AutoPipelineForImage2Image.from_pipe(self.tti)
        controlnetpath = self.config["controlnet"]
        if controlnetpath is not None:
            controlnetmodel = ControlNetModel.from_pretrained(controlnetpath, torch_dtype=torch.float32 if dtype == "fp32" else torch.float16)
            self.controlpipe = AutoPipelineForText2Image.from_pipe(self.tti, controlnet=controlnetmodel)
            if sys.platform == "darwin":
                self.controlpipe.to("mps")
            else:
                self.controlpipe.enable_model_cpu_offload()
        else:
            self.controlpipe = None

        self.lora()
        self.load_textual_inversion()
        self.tti.to("cpu", torch.float32 if dtype == "fp32" else torch.float16)
        if sys.platform == "darwin":
            self.tti.to("mps", torch.float32 if dtype == "fp32" else torch.float16)
            self.tti.enable_attention_slicing()
        elif torch.cuda.is_available():
            self.tti.to("cuda", torch.float32 if dtype == "fp32" else torch.float16)
            self.iti.to("cuda", torch.float32 if dtype == "fp32" else torch.float16)

    def prep_inputs(self, seed, text):
        if self.type == "xl":
            compel_proc = Compel(
                tokenizer=[self.tti.tokenizer, self.tti.tokenizer_2] ,
                text_encoder=[self.tti.text_encoder, self.tti.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
                )
        else:
            compel_proc = Compel(tokenizer=self.tti.tokenizer, text_encoder=self.tti.text_encoder)
        embed_prompt = compel_proc(text)
        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)
        return embed_prompt, generator

    def _predict(self, text, seed = None, iterations=20, height=512, width=512, guidance_scale=7.0) -> None:
        """ Predict from the loaded frames.

        With a threading lock (to prevent stacking), run the selected faces through the Faceswap
        model predict function and add the output to :attr:`predicted`
        """
        embed_prompt, generator = self.prep_inputs(seed, text) 
        if self.type == "xl":
            conditioning, pooled = embed_prompt
            image =  self.tti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale).images[0]
        else:
            image = self.tti(prompt_embeds=embed_prompt, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale).images[0]
        return image

    def img_to_img_predict(self, text, image, seed=None, iterations=25, height=512, width=512, guidance_scale=7.0, strength=0.75):
        embed_prompt, generator = self.prep_inputs(seed, text)
        if self.type == "xl":
            conditioning, pooled = embed_prompt
            output_img = self.iti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, image=image, generator=generator, num_inference_steps=iterations, guidance_scale=guidance_scale).images[0]
        else:
            output_img = self.iti(prompt_embeds=embed_prompt, image=image, generator=generator, num_inference_steps=iterations, guidance_scale=guidance_scale,strength=strength).images[0]
        output_img = output_img.resize((height, width))
        return output_img

    def lora(self):
        for lora, multiplier in self.config['loras']:
            self.tti.load_lora_weights(".", weight_name=lora)
            self.tti.fuse_lora(lora_scale=multiplier)

    def load_textual_inversion(self):
        for inverter in self.config['inverters']:
            self.tti.load_textual_inversion(inverter)

    def controlnet_predict(self, prompt: str, image, seed):
        embed_prompt, generator = self.prep_inputs(seed, prompt)
        output_img = self.controlpipe(prompt_embeds=embed_prompt, generator=generator, image = image, num_inference_steps=25).images[0]
        return output_img

    def on_install(self, model_urls=None):
        dtype = self.config["model_dtype"]

        self.notify_main_system_of_installation(0, "Starting download of runwayml stable-diffusion v1 5")
        AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32 if dtype == "fp32" else torch.float16, variant=dtype)
        self.notify_main_system_of_installation(100, "Download of runwayml stable diffusion v1 5 complete")