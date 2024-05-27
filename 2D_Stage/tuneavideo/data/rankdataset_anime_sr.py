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

class TuneAnimeMVImageDataset(Dataset):
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
            num_views = 1,
            unidream = False,
            pretrained_model_path = '',
            bg_color: str = 'gray',
            rank_dataset = False,
            fseek = False,
            use_modal = '',
            instant3d_rt = False,
            instant3d_ele = False,
            valid = False,
    ):
        # data from render
        if not valid:
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
        self.valid = valid
        # self.n_sample_frames = n_sample_frames
        # self.sample_start_idx = sample_start_idx
        # self.sample_frame_rate = sample_frame_rate
        self.totensor = transforms.ToTensor()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.image_root_path = "/mnt/pfs/users/zhangjiapeng/workspace_for_IP_Adapter/new_render"
        self.image_index = [0, 8, 16, 24]
        if valid:
            prompt_path = os.path.join(self.image_root_path, "valid_meta.jsonl")
        if self.rank_dataset:
            self.read_meta_files(prompt_path, image_dir)
        else:
            with open(prompt_path, 'r') as f:
                prompt_pairs = f.readlines()
            i = 0
            for prompt_pair in prompt_pairs:
                i += 1
                # if i != 5:
                #     continue
                # else:
                if i%10000 == 0:
                    if self.rank == 0:
                        print('----------------------------------------',i)
                data = json.loads(prompt_pair)
                if os.path.exists(os.path.join(self.image_root_path, data['image_dir'])):
                    self.images.append(data['image_dir'])
                    prompt = "high quality, best quality"
                    self.prompts.append(prompt)
                    self.prompt_ids.append(self.tokenizer(
                        prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
                        return_tensors="pt"
                    ).input_ids[0])
                    # break

    
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
        image_path = os.path.join(self.image_root_path, image_dir  + "_a_pose", image_dir + "_" + use_id + "_rgb.png")
        pose_path = os.path.join(self.image_root_path, image_dir + "_a_pose", image_dir+"_"+use_id+"_pose.png")
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
        pose_image = self.totensor(np.array(Image.open(pose_path).resize((int(self.width), int(self.height)), resample=PIL.Image.BICUBIC)).astype(np.float32)) / 255.
        return image, pose_image
        
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
        if random.random() < 0.7: # use frontview first
            image_inp = "007"
        else:
            image_inp = str(np.random.randint(0, 20) * 2 + 1).zfill(3)
        # load meta info
        # import pdb; pdb.set_trace()
        # meta_path = os.path.join(image_dir, "meta.json")
        # meta_infos= json.load(open(meta_path))['locations']
        # load image
        if self.instant3d_rt or self.instant3d_ele:
            pass
        else:
            # if self.unidream:
            #     select_id = np.random.randint(9,size=1)
            # else:
            #     group_len = 36 // self.num_views
            #     select_id = np.random.randint(group_len,size=1)
            imgs = []
            pose_imgs = []
            normals = []
            basecolors = []
            camera_matrixs = []
            if random.random() < 0.6:
                image_inp = np.asarray(Image.open(os.path.join(self.image_root_path, image_dir + "_a_pose", image_dir + "_512_gen.png"))) / 255.
            else:
                image_inp = np.asarray(Image.open(os.path.join(self.image_root_path, image_dir + "_a_pose", image_dir + "_512.png"))) / 255.
                bg_color = self.get_bg_color(self.bg_color)
                alpha = image_inp[..., 3:4]
                image_inp = image_inp[..., :3] * alpha + bg_color * (1 - alpha)
            image_inp = self.totensor(image_inp)
            
            assert self.num_views == 1, "sr only front view!"
                # select_id = 100
            select_id = 6
            for i in range(self.num_views):
                use_id = str(select_id + i * 2).zfill(3)

                # camera_matrix
                # camera_vector = torch.tensor(json.load(open(os.path.join(self.image_root_path, image_dir + "_a_pose", image_dir+"_" +f"{use_id}.json")))["extrinsicMatrix"]["elements"]).float()
                # camera_matrix = camera_vector.reshape(4,4).transpose(1,0)[:3,:4] # 3 * 4
                # camera_matrixs.append(camera_matrix)
                
                if self.use_modal == 'rgb':
                    image, pose_image = self.process_image(image_dir, use_id)
                    imgs.append(image.unsqueeze(0))
                    pose_imgs.append(pose_image.unsqueeze(0))
                elif self.use_modal == 'normal':
                    normal = self.process_normal(image_dir, use_id, camera_matrix)
                    normals.append(normal.unsqueeze(0))
                elif self.use_modal == 'basecolor':
                    basecolor = self.process_basecolor(image_dir, use_id)
                    basecolors.append(basecolor.unsqueeze(0))

            if self.use_modal == 'rgb':
                imgs = torch.cat(imgs, dim=0)
                pose_imgs = torch.cat(pose_imgs, dim=0)
            elif self.use_modal == 'normal':
                imgs = torch.cat(normals, dim=0)
            elif self.use_modal == 'basecolor':
                imgs = torch.cat(basecolors, dim=0)

            # camera_matrixs = torch.stack(camera_matrixs, dim=0)
            # camera normalize
            # translation = camera_matrixs[:,:3,3]
            # translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
            # camera_matrixs[:,:3,3] = translation
            example = {
                "input_pixel_values": image_inp.unsqueeze(0),
                "pixel_values": imgs,
                "prompt_ids": prompt_id,
                "camera_matrixs": [],
                "prompts": prompt,
                "pose_pixel_values": pose_imgs,
            }
            return example