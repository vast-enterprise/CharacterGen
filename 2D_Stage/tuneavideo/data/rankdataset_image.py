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
import torch.distributed as dist
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
            rank_dataset = False,
            fseek = False,
            use_modal = '',
            instant3d_rt = False,
            instant3d_ele = False,
    ):
        # data from render
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.images = []
        self.prompts = []
        self.prompt_ids = []
        self.num_views = num_views
        self.unidream = unidream
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.use_render_group_2 = use_render_group_2
        self.rank_dataset = rank_dataset
        self.image_dir = image_dir
        self.fseek = fseek
        self.use_modal = use_modal
        self.instant3d_rt = instant3d_rt
        self.instant3d_ele = instant3d_ele
        # self.n_sample_frames = n_sample_frames
        # self.sample_start_idx = sample_start_idx
        # self.sample_frame_rate = sample_frame_rate
        self.totensor = transforms.ToTensor()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")

        if self.rank_dataset:
            self.read_meta_files(prompt_path, image_dir)
        else:
            with open(prompt_path, 'r') as f:
                prompt_pairs = f.readlines()
            i = 0
            for prompt_pair in prompt_pairs:
                i += 1
                if i%10000 == 0:
                    if self.rank == 0:
                        print('----------------------------------------',i)
                image_path, prompt = prompt_pair.strip().split('\t')
                if os.path.exists(os.path.join(image_dir, image_path)):
                    self.images.append(os.path.join(image_dir, image_path))
                    self.prompts.append(prompt)
                    self.prompt_ids.append(self.tokenizer(
                        prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
                        return_tensors="pt"
                    ).input_ids[0])
        

    def __len__(self):
        if self.rank_dataset:
            return self.num
        else:
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

    def _load_meta(self, idx):
        if self.fseek:
            source_id = 0
            while idx >= len(self.line_offsets[source_id]):
                idx -= len(self.line_offsets[source_id])
                source_id += 1 #fixed
            with open(self.meta_file[source_id]) as f:
                f.seek(self.line_offsets[source_id][idx])
                line = f.readline()
                line = line.strip('\n')

                try:
                    image_path, prompt = line.split('\t')[0], line.split('\t')[1]
                except:
                    print(line, flush=True)
                image_path = os.path.join(self.image_dir, image_path)
                
                prompt_id = self.tokenizer(
                            prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
                            return_tensors="pt"
                        ).input_ids[0]
        else:
            image_path = self.images[idx]
            prompt_id = self.prompt_ids[idx]
        return image_path, prompt_id

    def worldNormal2camNormal(self, rot_w2c, normal_map_world):
        H,W,_ = normal_map_world.shape
        # normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

        # faster version
        # Reshape the normal map into a 2D array where each row represents a normal vector
        normal_map_flat = normal_map_world.reshape(-1, 3)

        # Transform the normal vectors using the transformation matrix
        normal_map_camera_flat = np.dot(normal_map_flat, rot_w2c.T)

        # Reshape the transformed normal map back to its original shape
        normal_map_camera = normal_map_camera_flat.reshape(normal_map_world.shape)

        return normal_map_camera

    def process_image(self, image_dir, use_id):
        # image
        image_path = os.path.join(image_dir, "render_%04d.webp"%(use_id))
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
        return image
        
    def process_normal(self, image_dir, use_id, camera_matrix):
        # image
        image_path = os.path.join(image_dir, "render_%04d.webp"%(use_id))
        image = Image.open(image_path)
        if not image.mode == "RGBA":
            image = image.convert("RGBA")
        image = image.resize((int(self.width), int(self.height)), resample=PIL.Image.BICUBIC)
        image = np.array(image)
        bg_color = self.get_bg_color(self.bg_color)
        image = image.astype(np.float32) / 255.
        assert image.shape[-1] == 4  # RGBA
        alpha = image[..., 3:4]
        # normal
        normal_path = os.path.join(image_dir, "normal_%04d.webp"%(use_id))
        normal = Image.open(normal_path)
        normal = normal.resize((int(self.width), int(self.height)), resample=PIL.Image.BICUBIC)
        normal = np.array(normal)
        normal = normal.astype(np.float32) / 255.
        normal = normal*2.0-1.0
        normal = self.worldNormal2camNormal(np.array(camera_matrix)[:3,:3], normal)
        normal = ((normal+1.0)/2.0).astype(np.float32)
        bg_color = self.get_bg_color(self.bg_color)
        normal = normal* alpha + bg_color * (1 - alpha)
        normal = normal.astype(np.float32)*2.0-1.0
        normal = self.totensor(normal)
        return normal

    def process_basecolor(self, image_dir, use_id):
        # basecolor
        basecolor_path = os.path.join(image_dir, "base-color_%04d.webp"%(use_id))
        basecolor = Image.open(basecolor_path)
        if not basecolor.mode == "RGBA":
            basecolor = basecolor.convert("RGBA")
        basecolor = basecolor.resize((int(self.width), int(self.height)), resample=PIL.Image.BICUBIC)
        basecolor = np.array(basecolor)
        bg_color = self.get_bg_color(self.bg_color)
        basecolor = basecolor.astype(np.float32) / 255.
        assert basecolor.shape[-1] == 4  # RGBA
        alpha = basecolor[..., 3:4]
        basecolor = basecolor[..., :3] * alpha + bg_color * (1 - alpha)
        # basecolor = (basecolor / 127.5 - 1.0).astype(np.float32)
        basecolor = basecolor.astype(np.float32)*2.0-1.0
        basecolor = self.totensor(basecolor)
        return basecolor

    def __getitem__(self, index):
        # image_dir = self.images[index]
        if self.rank_dataset:
            image_dir, prompt_id = self._load_meta(index)
        else:
            image_dir = self.images[index]
            prompt_id = self.prompt_ids[index]
            prompt = self.prompts[index]
        # load meta info
        # import pdb; pdb.set_trace()
        meta_path = os.path.join(image_dir, "meta.json")
        meta_infos= json.load(open(meta_path))['locations']
        # load image
        if self.instant3d_rt or self.instant3d_ele:
            data_list = [[
            "render_k-0_azi-0_index-0000.webp",
            "render_k-0_azi-1_index-0001.webp",
            "render_k-0_azi-2_index-0002.webp",
            "render_k-0_azi-3_index-0003.webp",
            "render_k-0_azi-4_index-0004.webp"],

            ["render_k-1_azi-0_index-0005.webp",
            "render_k-1_azi-1_index-0006.webp",
            "render_k-1_azi-2_index-0007.webp",
            "render_k-1_azi-3_index-0008.webp",
            "render_k-1_azi-4_index-0009.webp"],

            ["render_k-2_azi-0_index-0010.webp",
            "render_k-2_azi-1_index-0011.webp",
            "render_k-2_azi-2_index-0012.webp",
            "render_k-2_azi-3_index-0013.webp",
            "render_k-2_azi-4_index-0014.webp"],

            ["render_k-3_azi-0_index-0015.webp",
            "render_k-3_azi-1_index-0016.webp",
            "render_k-3_azi-2_index-0017.webp",
            "render_k-3_azi-3_index-0018.webp",
            "render_k-3_azi-4_index-0019.webp"]]
            img_inp = []
            imgs = []
            camera_matrixs = []
            select_id = np.random.randint(4,size=1)
            for image_lst in data_list[int(select_id)]:
                image_path = os.path.join(image_dir, image_lst)
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
                # image = image.astype(np.float32)*2.0-1.0
                image = self.totensor(image)
                if 'azi-0' in image_path:
                    img_inp.append(image.unsqueeze(0))
                else:
                    imgs.append(image.unsqueeze(0))
                # camera_matrix
                camera_matrix = meta_infos[int(image_lst.split('-')[-1].strip('.webp'))]['transform_matrix'][:3] #3x4
                if 'azi-0' not in image_path:
                    camera_matrixs.append(self.totensor(np.array(camera_matrix))) 

            img_inp = torch.cat(img_inp, dim=0)
            imgs = torch.cat(imgs, dim=0)
            camera_matrixs = torch.cat(camera_matrixs, dim=0)
            # camera normalize
            translation = camera_matrixs[:,:3,3]
            translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
            camera_matrixs[:,:3,3] = translation

            example = {
                "input_pixel_values": img_inp,
                "pixel_values": imgs,
                "prompt_ids": prompt_id,
                "camera_matrixs": camera_matrixs,
                "prompts": prompt
            }
            return example


        else:
            if self.unidream:
                select_id = np.random.randint(9,size=1)
            else:
                group_len = 36 // self.num_views
                select_id = np.random.randint(group_len,size=1)
            imgs = []
            normals = []
            basecolors = []
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
                
                if self.use_modal == 'rgb':
                    image = self.process_image(image_dir, use_id)
                    imgs.append(image.unsqueeze(0))
                elif self.use_modal == 'normal':
                    normal = self.process_normal(image_dir, use_id, camera_matrix)
                    normals.append(normal.unsqueeze(0))
                elif self.use_modal == 'basecolor':
                    basecolor = self.process_basecolor(image_dir, use_id)
                    basecolors.append(basecolor.unsqueeze(0))

            if self.use_modal == 'rgb':
                imgs = torch.cat(imgs, dim=0)
            elif self.use_modal == 'normal':
                imgs = torch.cat(normals, dim=0)
            elif self.use_modal == 'basecolor':
                imgs = torch.cat(basecolors, dim=0)

            camera_matrixs = torch.cat(camera_matrixs, dim=0)
            # camera normalize
            translation = camera_matrixs[:,:3,3]
            translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
            camera_matrixs[:,:3,3] = translation

            example = {
                "pixel_values": imgs,
                "prompt_ids": prompt_id,
                "camera_matrixs": camera_matrixs,
            }
            return example