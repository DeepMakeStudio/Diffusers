import os
from argparse import Namespace
from fastapi import FastAPI, HTTPException
from PIL import Image

from io import BytesIO
import torch
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline,AutoPipelineForText2Image,AutoPipelineForImage2Image, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, PNDMScheduler, StableDiffusionInpaintPipeline, ControlNetModel
from .pipelines import StableDiffusionPTPipeline, retrieve_timesteps, StableDiffusionXLPTPipeline, StableDiffusionImg2ImgPTPipeline, StableDiffusionXLImg2ImgPTPipeline
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
import re

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
    except Exception as e:
        # raise e
        sd_plugin.notify_main_system_of_startup("False")

@app.get("/set_model/")
def set_model():
    global sd_plugin
    args = {"plugin": plugin, "config": config, "endpoints": endpoints}
    pp = PromptParser(Namespace(**args))
    sd_plugin = SD(Namespace(**args), pp)
    sd_plugin.set_model()
    # try:
    # sd_plugin.set_model(args["model_name"], dtype=args["model_dtype"])
    model_name = sd_plugin.config["model_name"]
    return {"status": "Success", "detail": f"Model set successfully to {model_name}"}

@app.put("/execute/")
def execute(json_data: dict, seed: int = None, iterations: int = 20, height: int = 512, width: int = 512, guidance_scale: float = 7.0, control_image: str = None, negative_prompt: str = None):
    # check_model()
    prompt = json_data["prompt"]
    if control_image is None:
        im = sd_plugin._predict(prompt, seed=seed, iterations=iterations, height=height, width=width, guidance_scale=guidance_scale, negative_prompt=negative_prompt)
    else:
        imagebytes = fetch_image(control_image)
        control_image = Image.open(BytesIO(imagebytes))
        im = sd_plugin.controlnet_predict(prompt, control_image, seed=seed)

    output = BytesIO()
    im.save(output, format="PNG")
    image_id = store_image(output.getvalue())

    return {"status": "Success", "output_img": image_id}

@app.put("/execute2/")
def execute2(json_data: dict, seed = None, iterations: int = 20, height: int = 512, width: int = 512, guidance_scale: float = 7.0, strength: float = 0.75, negative_prompt=None):
    # check_model()
    text = json_data["prompt"]
    img = json_data["img"]
    imagebytes = fetch_image(img)
    image = Image.open(BytesIO(imagebytes))
    image = image.convert("RGB")
    im = sd_plugin.img_to_img_predict(text, image, seed=seed, iterations=iterations, height=height, width=width, guidance_scale=guidance_scale, strength=strength, negative_prompt=negative_prompt)
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
    def __init__(self, arguments: "Namespace", pp) -> None:
        super().__init__(arguments)
        self.plugin_name = "Diffusers"
        self.pp = pp

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

    def set_model(self) -> None:
        """
        Load given weights into model.
        """
        model_path = self.config["model_name"]
        dtype = self.config["model_dtype"]
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
                self.tti = StableDiffusionXLPTPipeline.from_pretrained(model_path,
                                                                     torch_dtype=torch.float32 if dtype == "fp32" else torch.float16,
                                                                     variant=dtype)
            else:
                self.type = "sd"
                self.tti = StableDiffusionPTPipeline.from_pretrained(model_path,
                                                                   torch_dtype=torch.float16,
                                                                   variant=dtype)
            # self.tti = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=torch.float32 if dtype == "fp32" else torch.float16, variant=dtype)
        # self.tti.scheduler = PNDMScheduler.from_config(self.tti.scheduler.config)
        if self.config["scheduler"] == "pndm":
            pass
        elif self.config["scheduler"] == "dpm":
            self.tti.scheduler = DPMSolverMultistepScheduler.from_config(self.tti.scheduler.config)
        else:
            print("Warning: Unknown scheduler. Using PNDM")

        if self.type == "xl":
            self.iti = StableDiffusionXLImg2ImgPTPipeline(**self.tti.components)
        else: 
            self.iti = StableDiffusionImg2ImgPTPipeline(**self.tti.components)
        # controlnetpath = self.config["controlnet"]
        # if controlnetpath is not None:
        #     controlnetmodel = ControlNetModel.from_pretrained(controlnetpath, torch_dtype=torch.float32 if dtype == "fp32" else torch.float16)
        #     self.controlpipe = AutoPipelineForText2Image.from_pipe(self.tti, controlnet=controlnetmodel)
        #     if sys.platform == "darwin":
        #         self.controlpipe.to("mps")
        #     else:
        #         self.controlpipe.enable_model_cpu_offload()
        # else:
        #     self.controlpipe = None

        # self.lora()
        # self.load_textual_inversion()
        self.tti.to("cpu", torch.float32 if dtype == "fp32" else torch.float16)
        self.iti.to("cpu", torch.float32 if dtype == "fp32" else torch.float16)

        if sys.platform == "darwin":
            self.tti.to("mps", torch.float32 if dtype == "fp32" else torch.float16)
            self.tti.enable_attention_slicing()
            self.iti.to("mps", torch.float32 if dtype == "fp32" else torch.float16)
            self.iti.enable_attention_slicing()
        elif torch.cuda.is_available():
            self.tti.to("cuda", torch.float32 if dtype == "fp32" else torch.float16)
            self.iti.to("cuda", torch.float32 if dtype == "fp32" else torch.float16)

    def prep_inputs(self, seed, text, compel_proc):
        embed_prompt = compel_proc(text)

        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)

        return embed_prompt, generator
    
    def apply_weights(self, embeddings, tokenizer, weights, text):
        input_ids = tokenizer(text, return_tensors='pt').input_ids[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        original_embeddings = embeddings.clone()

    def _predict(self, text, seed=None, iterations=20, height=512, width=512, guidance_scale=7.0, negative_prompt=None) -> None:

        parsed_result = self.pp.parse_prompt(self.tti, text)
        if isinstance(parsed_result, tuple):
            text, timestep_table = parsed_result
        else:
            text = [parsed_result]
            timestep_table = None

        print(f"parse_prompt: new_prompt={text}")

        embed_prompts = []

        if self.type == "xl":
            compel_proc = Compel(
                tokenizer=[self.tti.tokenizer, self.tti.tokenizer_2],
                text_encoder=[self.tti.text_encoder, self.tti.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            pooled_prompts = []
            for prompt in text:
                encoded, generator = self.prep_inputs(seed, prompt, compel_proc)
                embed_prompt, pooled_prompt = encoded
                embed_prompts.append(embed_prompt)
                pooled_prompts.append(pooled_prompt)
        else:
            compel_proc = Compel(tokenizer=self.tti.tokenizer, text_encoder=self.tti.text_encoder)
            for prompt in text:
                embed_prompt, generator = self.prep_inputs(seed, prompt, compel_proc)
                embed_prompts.append(embed_prompt)
        image = None
        timesteps, num_inference_steps = retrieve_timesteps(self.tti.scheduler, iterations, self.tti._execution_device, None)
        timesteps = timesteps.cpu()

        if self.type == "xl":
            for i in range(len(text)):
                conditioning = embed_prompts[i]
                pooled = pooled_prompts[i]
                start_step = timestep_table[i] if timestep_table is not None else None

                if i < len(text) - 1:
                    end_step = timestep_table[i + 1] if timestep_table is not None else None
                else:
                    end_step = None

                if i == len(text) - 1:
                    if image is not None:
                        image = image[None, :, :, :]
                    image =  self.tti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, timesteps=timesteps[start_step:iterations], latents=image, negative_prompt=negative_prompt).images[0]
                elif i == 0:
                    image = self.tti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type="latent", timesteps=timesteps[:end_step], negative_prompt=negative_prompt).images[0]
                elif i < len(text) - 1:
                    image = image[None, :, :, :]
                    image = self.tti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type= "latent", timesteps=timesteps[start_step:end_step], latents=image, negative_prompt=negative_prompt).images[0]
        else:
            for i in range(len(text)):
                embed_prompt = embed_prompts[i]
                start_step = timestep_table[i] if timestep_table is not None else None
                if i < len(text) - 1:
                    end_step = timestep_table[i + 1] if timestep_table is not None else None
                else:
                    end_step = None

                if i == len(text) - 1:
                    if image is not None:
                        image = image[None, :, :, :]
                    image = self.tti(prompt_embeds=embed_prompt, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, timesteps=timesteps[start_step:iterations], latents=image, negative_prompt=negative_prompt).images[0]
                elif i == 0:
                    image = self.tti(prompt_embeds=embed_prompt, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type="latent", timesteps=timesteps[:end_step], negative_prompt=negative_prompt).images[0]
                elif i < len(text) - 1:
                    image = image[None, :, :, :]
                    image = self.tti(prompt_embeds=embed_prompt, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type= "latent", timesteps=timesteps[start_step:end_step], latents=image, negative_prompt=negative_prompt).images[0]
                
        return image

    def img_to_img_predict(self, text, image, seed=None, iterations=25, height=512, width=512, guidance_scale=7.0, strength=0.75, negative_prompt=None):
        text = self.parse_prompt(self.tti, text)
        if isinstance(text, tuple):
            text, timestep_table = text
        elif isinstance(text, str):
            text = [text]
            timestep_table = None
        embed_prompts = []

        if self.type == "xl":
            compel_proc = Compel(
                tokenizer=[self.tti.tokenizer, self.tti.tokenizer_2],
                text_encoder=[self.tti.text_encoder, self.tti.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            pooled_prompts = []
            for prompt in text:
                encoded, generator = self.prep_inputs(seed, prompt, compel_proc)
                embed_prompt, pooled_prompt = encoded
                embed_prompts.append(embed_prompt)
                pooled_prompts.append(pooled_prompt)
        else:
            compel_proc = Compel(tokenizer=self.tti.tokenizer, text_encoder=self.tti.text_encoder)

            for prompt in text:
                embed_prompt, generator = self.prep_inputs(seed, prompt, compel_proc) 
                embed_prompts.append(embed_prompt)

        output_img = None
        timesteps, num_inference_steps = retrieve_timesteps(self.iti.scheduler, iterations, self.iti._execution_device, None)
        print(timesteps, len(timesteps), num_inference_steps)
        timesteps, num_inference_steps = self.iti.get_timesteps(num_inference_steps, strength, self.iti._execution_device)
        timesteps = timesteps.cpu()
        print(timesteps, len(timesteps), num_inference_steps)


        if self.type == "xl":
            for i in range(len(text)):
                conditioning = embed_prompts[i]
                pooled = pooled_prompts[i]
                start_step = timestep_table[i] if timestep_table is not None else 0
                print(len(timesteps))

                if i < len(text) - 1:
                    end_step = timestep_table[i+1] if timestep_table is not None else None
                else:
                    end_step = None

                print(len(text), i, start_step, timestep_table)

                if i == len(text) - 1:
                    if output_img is not None:
                        output_img = output_img[None, :, :, :]
                    output_img =  self.iti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, image = image,generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, timesteps=timesteps[start_step:num_inference_steps], latents=output_img, negative_prompt=negative_prompt).images[0]
                elif i == 0:
                    output_img = self.iti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, image=image,generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type="latent", timesteps=timesteps[:end_step], negative_prompt=negative_prompt).images[0]
                elif i < len(text) - 1:
                    output_img = output_img[None, :, :, :]
                    output_img = self.iti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, image=image,generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type= "latent", timesteps=timesteps[start_step:end_step], latents=output_img, negative_prompt=negative_prompt).images[0]
        
            # conditioning, pooled = embed_prompt
            # output_img = self.iti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, image=image, generator=generator, num_inference_steps=iterations, guidance_scale=guidance_scale).images[0]
        else:
            for i in range(len(text)):
                embed_prompt = embed_prompts[i]
                start_step = timestep_table[i] if timestep_table is not None else None
                print(len(text), i, start_step, timestep_table)
                if i < len(text) - 1:
                    end_step = timestep_table[i+1] if timestep_table is not None else None
                else:
                    end_step = None
                if i == len(text) - 1:
                    if output_img is not None:
                        output_img = output_img[None, :, :, :]
                    output_img = self.iti(prompt_embeds=embed_prompt, image=image, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, timesteps=timesteps[start_step:num_inference_steps], latents=output_img, negative_prompt=negative_prompt).images[0]
                elif i == 0:
                    output_img = self.iti(prompt_embeds=embed_prompt, image=image, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type="latent", timesteps=timesteps[:end_step], negative_prompt=negative_prompt).images[0]
                elif i < len(text) - 1:
                    output_img = output_img[None, :, :, :]
                    output_img = self.iti(prompt_embeds=embed_prompt, image=image, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type= "latent", timesteps=timesteps[start_step:end_step], latents=output_img, negative_prompt=negative_prompt).images[0]
            # output_img = self.iti(prompt_embeds=embed_prompt, image=image, generator=generator, num_inference_steps=iterations, guidance_scale=guidance_scale,strength=strength).images[0]
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

class PromptParser():
    def __init__(self, args):
        self.loras = {}
        self.loras_path = args.config["loras_path"]
        self.textual_embedding_path = args.config["textual_embedding_path"]

    def parse_prompt(self, pipeline, prompt):
        print(f"parse_prompt: initial prompt={prompt}")

        # Replace %2F with / in prompt from HTTP TODO: Find a better way to handle this
        new_prompt = prompt

        # TODO Possible error?
        # if re.search("<", prompt) and not re.search("<lora:", prompt) and not re.search("<ti:", prompt):
            

        print("Parsing for loras")
        split = re.split("<lora:", prompt, 1)
        if len(split) == 1:
            if len(self.loras) == 1:
                pipeline.unfuse_lora()
            elif len(self.loras) > 1:
                sd_plugin.set_model()
        else:
            new_prompt = self.parse_loras(pipeline, prompt)
        print("Parsing for textual inversion")

        split = re.split("<ti:", new_prompt, 1)
        if len(split) != 1:
            new_prompt = self.parse_ti(pipeline, new_prompt)
        print(new_prompt)
        new_prompt = self.parse_weighted_prompt(new_prompt)
        print(f"parse_prompt: new_prompt={new_prompt}")

        print("Parsing for prompt travel")
        split = re.split("\[", new_prompt, 1)
        timestep_table = None
        if len(split) != 1:
            new_prompt, timestep_table = self.parse_prompt_travel(new_prompt)
            return new_prompt, timestep_table
        return new_prompt
    
    def parse_prompt_travel(self, prompt):
        prompt_dict = {0: prompt}
        # while re.search("\[", prompt):
        #     self.get_table(prompt)
        pattern = re.compile(r'(\[(\w*?\d*?):(\w*?\d*?):(\d*)(:(\w*?\d*?):\d*)*?\])')
        matches = pattern.findall(prompt)
        for match in matches:
            timestep_table = [0]
            replace_phrase = match[0]
            info = replace_phrase.replace("[", "").replace("]", "").split(":")
            text_list = []
            for i in range(len(info)):
                if i <= 1:
                    text_list.append(info[i])
                elif i % 2 == 0:
                    timestep_table.append(int(info[i]))
                else:
                    text_list.append(info[i])
            print("text_list: ", text_list)
            print(timestep_table)
            temp_prompt_dict = prompt_dict.copy()
            for i in range(len(timestep_table)):
                print("temp_prompt_dict: ", temp_prompt_dict)
                timestep = timestep_table[i]
                phrase = text_list[i]
                print("phrase: ", phrase, "timestep: ", timestep)
                if timestep in prompt_dict.keys():
                    prompt_dict[timestep] = temp_prompt_dict[timestep].replace(replace_phrase, phrase)
                else:
                    timestep_list = list(sorted(prompt_dict.keys()))
                    prev_idx = 0
                    for idx in range(len(timestep_list)):
                        if timestep_list[idx] < timestep:
                            prev_idx = timestep_list[idx]
                        else:
                            break
                    print("prev_idx: ", prev_idx)
                    print("temp_prompt_dict: ", temp_prompt_dict)
                    prompt_dict[timestep] = temp_prompt_dict[prev_idx].replace(replace_phrase, phrase)
                    temp_prompt_dict[timestep] = temp_prompt_dict[prev_idx]
            print(prompt_dict)
            for key in prompt_dict.keys():
                if key not in timestep_table:

                    timestep_list = list(sorted(prompt_dict.keys()))
                    prev_idx = 0
                    for idx in range(len(timestep_table)):
                        if timestep_list[idx] < key:
                            phrase = text_list[idx]
                        else:
                            break
                    prompt_dict[key] = prompt_dict[key].replace(replace_phrase, phrase)


        prompt_dict = dict(sorted(prompt_dict.items()))
        prompt_list = list(prompt_dict.values())
        timestep_table = list(prompt_dict.keys())
        print(prompt_list, timestep_table)
        return prompt_list, timestep_table
    
    def parse_ti(self, pipeline, prompt):
        while re.search("<ti:", prompt):
            prompt_start, temp = re.split("<ti:", prompt, 1)
            ti_info, prompt_end = re.split(">", temp, 1)
            ti_name, token = ti_info.split(":")
            if sd_plugin.type == "xl":
                if "/" in ti_name:
                    hf_repo, weight_name = self.hf_split(ti_name)
                    state_dict = load_file(os.path.join(hf_repo, weight_name))
                else:
                    state_dict = load_file(os.path.join(self.textual_embedding_path, ti_name))
                token0 = "<" + token + "0>"
                token1 = "<" + token + "1>"
                pipeline.load_textual_inversion(state_dict["clip_l"], token=[token0, token1], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
                pipeline.load_textual_inversion(state_dict["clip_g"], token=[token0, token1], text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2)
                prompt = prompt_start + token0 + token1 + prompt_end
            else:
                if "/" in ti_name:
                    hf_repo, weight_name = self.hf_split(ti_name)
                    pipeline.load_textual_inversion(hf_repo, weight_name=weight_name, token=token)
                else:
                    pipeline.load_textual_inversion(self.textual_embedding_path, weight_name=ti_name, token=token)
                prompt = prompt_start + token + prompt_end
        return prompt
    
    def parse_loras(self, pipeline, prompt):
        # Parsing for multiple loras
        new_prompt = prompt
        lora_dict = {}
        print("Parsing prompt")
        while re.search("<lora:", new_prompt):
            prompt_start, temp = re.split("<lora:", new_prompt, 1)
            lora_info, prompt_end = re.split(">", temp, 1)
            lora_name, lora_weight = lora_info.split(":")
            lora_dict[lora_name] = float(lora_weight)
            new_prompt = prompt_start + prompt_end

        # If the lora structure is the same, return immediately
        same = True
        for lora_name in lora_dict.keys():
            if lora_name not in self.loras or self.loras[lora_name] != lora_dict[lora_name]:
                same = False
                break
            
        if same:    
            return new_prompt
    
        # Unload the current lora weights
        if len(self.loras) == 1:
            pipeline.unfuse_lora()
        elif len(self.loras) > 1:
            sd_plugin.set_model()
        
        # Cache lora info and load the lora weights
        self.loras = lora_dict
        adapter_name = 0
        adapter_name_list = []
        adapter_weight_list = []

        print("Loading LoRA weights")
        # Load the lora weights based on the lora_dict information and prepare adapter info
        for lora_name, lora_weight in lora_dict.items():
            if "/" in lora_name:
                hf_repo, weight_name = self.hf_split(lora_name)
                self.check_lora_weights(pipeline, hf_repo, weight_name=weight_name, adapter_name=str(adapter_name), hf_repo=True)
            else:
                self.check_lora_weights(pipeline, self.loras_path, weight_name=lora_name, adapter_name=str(adapter_name))
            adapter_name_list.append(str(adapter_name))
            adapter_weight_list.append(lora_weight)
            adapter_name += 1
        print("Merging LoRA weights")
        # Set the adapters
        pipeline.set_adapters(adapter_name_list, adapter_weight_list)
        print("Fusing Loras")
        # Fuse Loras
        pipeline.fuse_lora(adapter_names=adapter_name_list, lora_scale=1.0)
        pipeline.unload_lora_weights()

        return new_prompt
    
    def hf_split(self, hf_path):
        try:
            author, repo, weight_name = hf_path.split("/")
        except:
            raise HTTPException(status_code=400, detail="Please make sure weights are selected from Hugging Face model hub in the format 'author/repo/weight_name'")
        hf_repo = "/".join([author, repo])
        return hf_repo, weight_name

    def check_lora_weights(self, pipeline, path, weight_name, adapter_name=None, hf_repo=False):
        try:
            pipeline.load_lora_weights(path, weight_name=weight_name, adapter_name=adapter_name)
        except Exception as e:
            if isinstance(e, ValueError):
                raise HTTPException(status_code=400, detail="LoRA weights are incompatible with loaded model")
            elif isinstance(e, OSError):
                if hf_repo:
                    raise HTTPException(status_code=400, detail="LoRA weights not found in Hugging Face model hub")
                else:
                    raise HTTPException(status_code=400, detail="LoRA file not found in the specified path")
    def parse_weighted_prompt(self, prompt):
        cleaned_prompt = self.extract_weights(prompt)
        return cleaned_prompt

    def extract_weights(self, prompt):
        # Match patterns like (prompt:weight) and extract them
        pattern = re.compile(r'\(([^:]+):(\d*\.?\d+)\)')
        matches = pattern.findall(prompt)

        # Convert (prompt:weight) to (prompt)weight
        def replace_match(match):
            word, weight = match.groups()
            return f"({word}){weight}"

        cleaned_prompt = pattern.sub(replace_match, prompt)
        return cleaned_prompt
    

    """
    def parse_weighted_prompt(self, pipeline, prompt):
        weights = self.extract_weights(prompt)
        cleaned_prompt = re.sub(r"\+\+|--", "", prompt)
        return cleaned_prompt, weights

    def extract_weights(self, prompt):
        # Finds patterns like "word++" or "word--" and assigns weight modifications
        matches = re.findall(r"(\w+)(\+\+|--)", prompt)
        weights = {match[0]: 1.1 ** match[1].count('+') if '+' in match[1] else 0.9 ** match[1].count('-') for match in matches}
        return weights
    """
