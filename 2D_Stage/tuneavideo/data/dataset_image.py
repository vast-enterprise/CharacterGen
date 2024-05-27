import decord
import os
import numpy as np
import torch
import PIL
import json
import random
from PIL import Image
from glob import glob
from random import choice

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms

class TuneMVImageDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            prompt_path: str,
            laion_data_dir: str,
            width: int = 256,
            height: int = 256,
            use_render_group_2 = False,
            # n_sample_frames: int = 8,
            # sample_start_idx: int = 0,
            # sample_frame_rate: int = 1,
            num_views = 4,
            unidream = False,
            pretrained_model_path = '',
            bg_color: str = 'gray',
    ):
        # data from render
        self.images = []
        self.prompts = []
        self.prompt_ids = []
        self.num_views = num_views
        self.unidream = unidream
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.use_render_group_2 = use_render_group_2
        # self.n_sample_frames = n_sample_frames
        # self.sample_start_idx = sample_start_idx
        # self.sample_frame_rate = sample_frame_rate
        self.totensor = transforms.ToTensor()
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        with open(prompt_path, 'r') as f:
            prompt_pairs = f.readlines()
        i = 0
        for prompt_pair in prompt_pairs:
            i += 1
            if i%10000 == 0:
                print('----------------------------------------',i)
            image_path, prompt = prompt_pair.strip().split('\t')
            if os.path.exists(os.path.join(image_dir, image_path)):
                self.images.append(os.path.join(image_dir, image_path))
                self.prompts.append(prompt)
                self.prompt_ids.append(tokenizer(
                    prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                    return_tensors="pt"
                ).input_ids[0])
        

    def __len__(self):
        return len(self.images)

    def get_bg_color(self, bg_color):
        if bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == 'random':
            bg_color = np.random.rand(3)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color


    def __getitem__(self, index):
        image_dir = self.images[index]
        # load meta info
        meta_path = os.path.join(image_dir, "meta.json")
        meta_infos= json.load(open(meta_path))['locations']
        # load image
        if self.unidream:
            select_id = np.random.randint(9,size=1)
        else:
            group_len = 36 // self.num_views
            select_id = np.random.randint(group_len,size=1)
        imgs = []
        camera_matrixs = []
        for i in range(self.num_views):
            if self.use_render_group_2:
                if self.unidream:
                    use_id_1 = int(select_id*4+i)
                    use_id_2 = int(select_id*4+i+36)
                    use_id = choice([use_id_1, use_id_2])
                else:
                    use_id_1 = int(select_id+group_len*i)
                    use_id_2 = int(select_id+group_len*i+36)
                    use_id = choice([use_id_1, use_id_2])
            else:
                if self.unidream:
                    use_id = int(select_id*4+i)
                else:
                    use_id = int(select_id+group_len*i)

            # camera_matrix
            camera_matrix = meta_infos[use_id]['transform_matrix'][:3] #3x4
            camera_matrixs.append(self.totensor(np.array(camera_matrix))) 
            # image
            if self.unidream:
                image_path = os.path.join(image_dir, "render_%04d.webp"%(use_id))
            else:
                image_path = os.path.join(image_dir, "render_%04d.png"%(use_id))
            image = Image.open(image_path)
            if not image.mode == "RGBA":
                image = image.convert("RGBA")
            image = image.resize((int(self.width), int(self.height)), resample=PIL.Image.BICUBIC)
            image = np.array(image)
            bg_color = self.get_bg_color(self.bg_color)
            image = image.astype(np.float32) / 255.
            assert image.shape[-1] == 4  # RGBA
            alpha = image[..., 3:4]
            image = image[..., :3] * alpha + bg_color * (1 - alpha)
            # image = (image / 127.5 - 1.0).astype(np.float32)
            image = image.astype(np.float32)*2.0-1.0
            image = self.totensor(image)
            imgs.append(image.unsqueeze(0))
        # tile images
        # h, w, = imgs[0].size[0], imgs[0].size[1]
        # ROW = 2
        # COL = 2
        # image_all = Image.new('RGB', (w * ROW, h * COL))
        # for row in range(ROW):
        #     for col in range(COL):
        #         image_all.paste(imgs[COL * row + col], (0 + self.width * col, 0 + self.height * row))
        imgs = torch.cat(imgs, dim=0)
        camera_matrixs = torch.cat(camera_matrixs, dim=0)
        example = {
            "pixel_values": imgs,
            "prompt_ids": self.prompt_ids[index],
            "camera_matrixs": camera_matrixs
         }

        return example
    

class TuneLaionImageDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            prompt_path: str,
            laion_data_dir: str,
            width: int = 256,
            height: int = 256,
            use_render_group_2 = False,
            # n_sample_frames: int = 8,
            # sample_start_idx: int = 0,
            # sample_frame_rate: int = 1,
            num_views = 4,
            unidream = False,
            pretrained_model_path = '',
            bg_color = ''
    ):
        # data from laion-5b
        self.laion_images = []
        self.laion_prompts = []
        self.laion_prompt_ids = []
        self.width = width
        self.height = height
        # self.n_sample_frames = n_sample_frames
        # self.sample_start_idx = sample_start_idx
        # self.sample_frame_rate = sample_frame_rate
        self.totensor = transforms.ToTensor()
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        # for laion_image_path in glob(f'{laion_data_dir}/*/*.webp'):
        #     laion_text_path = laion_image_path.replace('webp', 'txt')
        #     laion_prompt = open(laion_text_path).readlines()[0]
        i = 0
        for line in open(laion_data_dir).readlines():
            i += 1
            if i%10000 == 0:
                print('----------------------------------------',i)
            line = line.strip('\n')
            laion_image_path, laion_prompt = line.split('\t')[0], line.split('\t')[1]

            self.laion_images.append(laion_image_path)
            self.laion_prompts.append(laion_prompt)
            self.laion_prompt_ids.append(tokenizer(
                laion_prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                return_tensors="pt"
            ).input_ids[0])

    def __len__(self):
        return len(self.laion_images)

    def __getitem__(self, index):
        image_path = self.laion_images[index]
        imgs = []
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((self.width, self.height), resample=PIL.Image.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = self.totensor(image)
        imgs.append(image.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)
        example = {
            "pixel_values": imgs,
            "prompt_ids": self.laion_prompt_ids[index],
        }

        return example