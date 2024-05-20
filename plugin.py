import os
from argparse import Namespace
from fastapi import FastAPI, HTTPException
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
def execute(json_data: dict, seed: int = None, iterations: int = 20, height: int = 512, width: int = 512, guidance_scale: float = 7.0, control_image: str = None):
    # check_model()
    prompt = json_data["prompt"]
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

        # self.lora()
        # self.load_textual_inversion()
        self.tti.to("cpu", torch.float32 if dtype == "fp32" else torch.float16)
        if sys.platform == "darwin":
            self.tti.to("mps", torch.float32 if dtype == "fp32" else torch.float16)
            self.tti.enable_attention_slicing()
        elif torch.cuda.is_available():
            self.tti.to("cuda", torch.float32 if dtype == "fp32" else torch.float16)
            self.iti.to("cuda", torch.float32 if dtype == "fp32" else torch.float16)
    
    def prep_inputs(self, seed, text, compel_proc):
        print(f"prep_inputs: seed={seed}, text={text}")

        embed_prompt = compel_proc(text)

        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)

        return embed_prompt, generator
    
    def apply_weights(self, embeddings, tokenizer, weights, text):
        input_ids = tokenizer(text, return_tensors='pt').input_ids[0]
        original_embeddings = embeddings.clone()
        print(f"Original Embeddings:\n {original_embeddings}")

        for word, weight in weights.items():
            token_ids = self.find_word_indices(word, tokenizer, input_ids)
            print(f"Word: {word}, Weight: {weight}, Token IDs: {token_ids}")
            for index in token_ids:
                if index < embeddings.size(1):
                    print(f"Applying weight: {weight} to word: {word} at index: {index}")
                    print(f"Value before applying weight: {embeddings[:, index]}")
                    embeddings[:, index] *= weight
                    print(f"Value after applying weight: {embeddings[:, index]}")

        if not torch.equal(original_embeddings, embeddings):
            print("The embeddings tensor was modified.")
        else:
            print("The embeddings tensor was NOT modified.")
        print("Modified Embeddings:\n", embeddings)

        comparison_result = torch.equal(original_embeddings, embeddings)
        print(f"Comparison Result: {comparison_result}")

        return embeddings

    def find_word_indices(self, word, tokenizer, input_ids):
        word_indices = []
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        for idx, token in enumerate(tokens):
            if word in token:
                word_indices.append(idx)
        return word_indices

    def _predict(self, text, seed=None, iterations=20, height=512, width=512, guidance_scale=7.0) -> None:
        print(f"_predict: text={text}, seed={seed}, iterations={iterations}, height=512, width=512, guidance_scale={guidance_scale}")

        parsed_result, weights = self.pp.parse_prompt(self.tti, text)
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
                print(f"Embeddings before applying weights: {embed_prompt}")
                if weights:
                    embed_prompt = self.apply_weights(embed_prompt, self.tti.tokenizer, weights, prompt)
                print(f"Embeddings after applying weights: {embed_prompt}")
                embed_prompts.append(embed_prompt)

        print(f"_predict: embed_prompts={embed_prompts}")

        image = None
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
                    image = self.tti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, start_step=start_step, latents=image).images[0]
                elif i == 0:
                    image = self.tti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type="latent", end_step=end_step)
                elif i < len(text) - 1:
                    image = self.tti(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type="latent", end_step=end_step, start_step=start_step, latents=image)
        else:
            for i in range(len(text)):
                embed_prompt = embed_prompts[i]
                start_step = timestep_table[i] if timestep_table is not None else None
                if i < len(text) - 1:
                    end_step = timestep_table[i + 1] if timestep_table is not None else None
                else:
                    end_step = None

                if i == len(text) - 1:
                    image = self.tti(prompt_embeds=embed_prompt, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, start_step=start_step, latents=image).images[0]
                elif i == 0:
                    image = self.tti(prompt_embeds=embed_prompt, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type="latent", end_step=end_step)
                elif i < len(text) - 1:
                    image = self.tti(prompt_embeds=embed_prompt, generator=generator, num_inference_steps=iterations, height=height, width=width, guidance_scale=guidance_scale, output_type="latent", end_step=end_step, start_step=start_step, latents=image)

        print(f"_predict: generated image={image}")
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

class PromptParser():
    def __init__(self, args):
        self.loras = {}
        self.loras_path = args.config["loras_path"]
        self.textual_embedding_path = args.config["textual_embedding_path"]

    def parse_prompt(self, pipeline, prompt):
        print(f"parse_prompt: initial prompt={prompt}")

        # Replace %2F with / in prompt from HTTP TODO: Find a better way to handle this
        prompt = re.sub("%2F", "/", prompt)
        new_prompt = prompt

        print("Parsing for loras")
        split = re.split("<lora:", prompt, 1)
        if len(split) == 1:
            if len(self.loras) == 1:
                pipeline.unfuse_lora()
            elif len(self.loras) > 1:
                self.set_model()
        else:
            new_prompt = self.parse_loras(pipeline, prompt)
        print("Parsing for textual inversion")

        split = re.split("<ti:", new_prompt, 1)
        if len(split) != 1:
            new_prompt = self.parse_ti(pipeline, new_prompt)

        print("Parsing for prompt travel")
        split = re.split("\[", new_prompt, 1)
        if len(split) != 1:
            new_prompt, timestep_table = self.parse_prompt_travel(new_prompt)
            return new_prompt, timestep_table
        
        new_prompt, weights = self.parse_weighted_prompt(pipeline, new_prompt)
        print(f"parse_prompt: new_prompt={new_prompt}")
        print(f"Weights: {weights}")
        return new_prompt, weights
    
    def parse_prompt_travel(self, prompt):
        prompt_list = []
        timestep_table = [0]
        prompt_start, temp = re.split("\[", prompt, 1)
        # [prompt1:prompt2:step:prompt3:step]
        phrase1, phrase2, timestep = re.split(":", temp, 2)
        timestep, prompt_end = re.split("\]", timestep, 1)
        timestep_table.append(int(timestep.split(":")[0]))
        prompt1 = prompt_start + phrase1 + prompt_end
        prompt2 = prompt_start + phrase2 + prompt_end
        prompt_list.append(prompt1)
        prompt_list.append(prompt2)
        temp = timestep
        if re.search(":", temp):
            timestep, temp = re.split(":", temp, 1)
        print(temp)
        while re.search(":", temp):
            phrase, timestep = re.split(":", temp, 1)
            print(phrase, timestep)
            if re.search(":", timestep):
                timestep, temp = re.split(":", timestep, 1)
            else:
                temp = ""
            timestep_table.append(int(timestep.split(":")[0]))
            prompt_list.append(prompt_start + phrase + prompt_end)
        print(prompt_list, timestep_table)
        return prompt_list, timestep_table
    
    def parse_ti(self, pipeline, prompt):
        while re.search("<ti:", prompt):
            prompt_start, temp = re.split("<ti:", prompt, 1)
            ti_info, prompt_end = re.split(">", temp, 1)
            ti_name, token = ti_info.split(":")
            if self.type == "xl":
                if "/" in ti_name:
                    author, repo, weight_name = ti_name.split("/")
                    hf_repo = "/".join([author, repo])
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
                    author, repo, weight_name = ti_name.split("/")
                    hf_repo = "/".join([author, repo])
                    pipeline.load_textual_inversion(hf_repo, weight_name=weight_name, token=f"<{token}>")
                else:
                    pipeline.load_textual_inversion(self.textual_embedding_path, weight_name=ti_name, token=f"<{token}>")
                prompt = prompt_start + f"<{token}>" + prompt_end
    
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
        if lora_dict == self.loras:
            return new_prompt
    
        # Unload the current lora weights
        if len(self.loras) == 1:
            pipeline.unfuse_lora()
        elif len(self.loras) > 1:
            self.set_model()
        
        # Cache lora info and load the lora weights
        self.loras = lora_dict
        adapter_name = 0
        adapter_name_list = []
        adapter_weight_list = []

        print("Loading LoRA weights")
        # Load the lora weights based on the lora_dict information and prepare adapter info
        for lora_name, lora_weight in lora_dict.items():
            if "/" in lora_name:
                author, repo, weight_name = lora_name.split("/")
                hf_repo = "/".join([author, repo])
                pipeline.load_lora_weights(hf_repo, weight_name=weight_name, adapter_name=str(adapter_name))
                # pipeline.load_lora_weights(hf_repo, weight_name=weight_name)
            else:
                pipeline.load_lora_weights(self.loras_path, weight_name=lora_name,  adapter_name=str(adapter_name))
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
    
    def parse_weighted_prompt(self, pipeline, prompt):
        weights = self.extract_weights(prompt)
        cleaned_prompt = re.sub(r"\+\+|--", "", prompt)
        return cleaned_prompt, weights

    def extract_weights(self, prompt):
        # Finds patterns like "word++" or "word--" and assigns weight modifications
        matches = re.findall(r"(\w+)(\+\+|--)", prompt)
        weights = {match[0]: 1.1 ** match[1].count('+') if '+' in match[1] else 0.9 ** match[1].count('-') for match in matches}
        return weights