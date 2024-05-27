import json
import math
import os
import random
from dataclasses import dataclass, field
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transform
from PIL import Image
import imageio
import cv2
from torch.utils.data import DataLoader, Dataset

import lrm
from ..utils.config import parse_structured
from ..utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
    get_intrinsic_from_fov,
    c2w_to_polar,
    polar_to_c2w,
)
from ..utils.typing import *

def generate_gt_probs(grid_vertices, joints, std=0.5):
    distance = (grid_vertices[:,None,:] - joints[None,...])**2 # 15 * g * 3
    return 1 / math.sqrt((2 * np.pi * std**2)**3) * torch.exp(-1/(2*std**2)*distance)

@dataclass
class ObjaverseDataModuleConfig:
    root_dir: Any = ""
    scene_list: Any = ""
    version: str = "v1"
    image_suffix: str = "webp"
    background_color: Tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0)
    )
    train_random_background: Optional[str] = None

    num_views_per_scene: int = 20
    num_views_input: int = 1
    num_views_output: int = 3

    relative_pose: bool = False
    relative_pose_prob: float = 1.0
    fix_input_pose: bool = False  # legacy, not used
    train_input_views: Optional[List[int]] = None
    train_sup_views: str = "random"
    cond_c2w_perturb_type: Optional[str] = None
    cond_c2w_perturb_scale: Tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )
    cond_rgb_perturb_type: Optional[str] = None
    cond_rgb_perturb_scale: Dict[str, Any] = field(default_factory=lambda: {})
    cond_rgb_swap_patch_type: Optional[str] = None
    cond_rgb_swap_patches: int = 4
    cond_rgb_swap_patch_size: int = 14

    return_first_n_cases: int = -1  # for debugging purpose
    repeat: int = 1  # for debugging purpose

    train_indices: Optional[Tuple[int, int]] = None
    val_indices: Optional[Tuple[int, int]] = None
    test_indices: Optional[Tuple[int, int]] = None

    height: int = 128
    width: int = 128
    rand_min_height: Optional[int] = None
    rand_max_height: int = 128
    rand_max_width: int = 128
    cond_height: int = 512
    cond_width: int = 512
    batch_size: int = 1
    num_workers: int = 16

    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 2.0
    eval_fovy_deg: float = 45.0
    n_test_views: int = 120
    eval_input_views: Optional[List[int]] = None


class ObjaverseDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val"]
        self.cfg: ObjaverseDataModuleConfig = cfg
        self.root_dir = self.cfg.root_dir[0]

        if split == "train":
            json_path = os.path.join(self.root_dir, "imagedream.jsonl")
            all_data = open(json_path, "r").readlines()
            self.all_dirs = []
            for data in all_data:
                data = json.loads(data)
                if os.path.exists(os.path.join(self.root_dir, data["image_dir"])):
                    self.all_dirs.append(data["image_dir"] + "_a_pose")
        else:
            json_path = os.path.join(self.root_dir, "valid_meta.jsonl")
            all_data = open(json_path, "r").readlines()
            self.all_dirs = []
            for data in all_data:
                data = json.loads(data)
                if os.path.exists(os.path.join(self.root_dir, data["image_dir"])):
                    self.all_dirs.append(data["image_dir"] + "_a_pose")

        self.background_color = torch.as_tensor(self.cfg.background_color)
        self.split = split
        # if self.split == "train" and self.cfg.train_indices is not None:
        #     self.all_scenes = self.all_scenes[
        #         self.cfg.train_indices[0] : self.cfg.train_indices[1]
        #     ]
        #     self.all_scenes = self.all_scenes * self.cfg.repeat
        # elif self.split == "val" and self.cfg.val_indices is not None:
        #     self.all_scenes = self.all_scenes[
        #         self.cfg.val_indices[0] : self.cfg.val_indices[1]
        #     ]
        # elif self.split == "test" and self.cfg.test_indices is not None:
        #     self.all_scenes = self.all_scenes[
        #         self.cfg.test_indices[0] : self.cfg.test_indices[1]
        #     ]

        self.ids = np.array([0, 4, 8, 12, 16])

    def __len__(self):
        return len(self.all_dirs)

    def __getitem__(self, index):
        scene_index = index
        if self.cfg.return_first_n_cases > 0:
            scene_index %= self.cfg.return_first_n_cases

        if random.random() < 0.6:
            init_ids = self.ids[0]
        else:
            init_ids = self.ids[np.random.choice([1,2,3,4])]
        view_ids_1 = [str(init_ids + i*2).zfill(3) for i in range(4)]
        view_ids_2 = [str(i*2).zfill(3) for i in np.random.randint(20, size=(4,))]
        view_ids = view_ids_1 + view_ids_2
        if self.split != "train" and self.cfg.eval_input_views is not None:
            assert len(self.cfg.eval_input_views) == self.cfg.num_views_input
            view_ids[: self.cfg.num_views_input] = self.cfg.eval_input_views

        data_cond, data_sup = [], []
        scene_dir = self.all_dirs[scene_index]
        scene_id = scene_dir.replace("_a_pose", "")
        # with open(os.path.join(scene_dir, scene_id+)) as f:   
        #     transforms = json.loads(f.read())

        if self.split != "train" or self.cfg.train_random_background is None:
            background_color = self.background_color
        elif self.cfg.train_random_background == "random_gray":
            background_color = torch.as_tensor(
                [random.random()] * 3, dtype=torch.float32
            )
        elif self.cfg.train_random_background == "random":
            background_color = torch.from_numpy(np.random.random(3)).float()

        ref_c2w, ref_w2c, canonical_c2w = None, None, None
        for i, view_index in enumerate(view_ids):
            if i < self.cfg.num_views_input:
                data_cur = data_cond
                crop_height, crop_width = self.cfg.cond_height, self.cfg.cond_width
                resize_height, resize_width = self.cfg.cond_height, self.cfg.cond_width
            else:
                data_cur = data_sup
                if self.split == "train":
                    crop_height, crop_width = self.cfg.height, self.cfg.width
                    resize_height = np.random.randint(
                        self.cfg.rand_min_height or self.cfg.height,
                        self.cfg.rand_max_height + 1,
                    )
                    resize_width = int(
                        np.round(resize_height * self.cfg.width / self.cfg.height)
                    )
                else:
                    crop_height, crop_width = self.cfg.eval_height, self.cfg.eval_width
                    resize_height, resize_width = (
                        self.cfg.eval_height,
                        self.cfg.eval_width,
                    )
            view_index_str = view_index
            
            metas = json.load(open(os.path.join(self.root_dir, scene_dir, scene_id+ "_" + view_index_str + ".json"), "r"))
            fovy = 40 / 180 * np.pi

            img_path = os.path.join(
                self.root_dir, scene_dir, f"{scene_id}_{view_index_str}_rgb.{self.cfg.image_suffix}"
            )
            img = torch.from_numpy(
                np.asarray(
                    Image.fromarray(imageio.v2.imread(img_path))
                    .convert("RGBA")
                    .resize((resize_width, resize_height))
                )
                / 255.0
            ).float()
            mask: Float[Tensor, "H W 1"] = img[:, :, -1:]
            rgb: Float[Tensor, "H W 3"] = img[:, :, :3] * mask + background_color[
                None, None, :
            ] * (1 - mask)

            depth_path = os.path.join(scene_dir, f"depth_{view_index_str}.exr")
            if os.path.exists(depth_path):
                depth: Float[Tensor, "H W 1"] = torch.from_numpy(
                    cv2.resize(
                        np.asarray(imageio.v2.imread(depth_path)),
                        (resize_width, resize_height),
                    )[..., 0:1]
                )
                mask[depth > 100.0] = 0.0
                depth[~(mask > 0.5)] = 0.0  # set invalid depth to 0
            else:
                depth = torch.zeros_like(mask)

            c2w = torch.as_tensor(
                np.array(metas["extrinsicMatrix"]["elements"]).astype(np.float32).reshape(4, 4).transpose(1, 0)[[0,2,1,3]],
                dtype=torch.float32,
            )
            c2w[:3,3] = c2w[:3,3]

            if (
                self.cfg.cond_c2w_perturb_type is not None
                and i < self.cfg.num_views_input
            ):
                elevation, azimuth, distance = c2w_to_polar(c2w)
                # print(f'before perturb {elevation * 180. / math.pi:.1f} {azimuth * 180. / math.pi:.1f} {distance:.2f}')
                scales = torch.randn(3) * torch.as_tensor(
                    self.cfg.cond_c2w_perturb_scale
                )
                elevation, azimuth, distance = (
                    torch.as_tensor([elevation, azimuth, distance])
                    + torch.as_tensor([math.pi / 180.0, math.pi / 180.0, 1.0]) * scales
                ).tolist()
                # print(f'after perturb {elevation * 180. / math.pi:.1f} {azimuth * 180. / math.pi:.1f} {distance:.2f}')
                c2w_perturb = polar_to_c2w(elevation, azimuth, distance)
                if self.cfg.cond_c2w_perturb_type == "all":
                    c2w = c2w_perturb
                elif self.cfg.cond_c2w_perturb_type == "except_first":
                    # use the ground truth c2w for the first conditional frame
                    if i > 0:
                        c2w = c2w_perturb
                elif self.cfg.cond_c2w_perturb_type == "zero_except_first":
                    if i > 0:
                        c2w = torch.zeros_like(c2w)
                else:
                    raise NotImplementedError

            if self.cfg.relative_pose and random.random() < self.cfg.relative_pose_prob:
                if ref_c2w is None:
                    # use the first frame as the reference frame
                    ref_c2w = c2w.clone()
                    ref_w2c = torch.linalg.inv(ref_c2w)
                    ref_distance = ref_c2w[0:3, 3].norm()
                    canonical_c2w = torch.as_tensor(
                        [
                            [0, 0, 1, ref_distance],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                        ]
                    ).float()
                c2w = canonical_c2w @ ref_w2c @ c2w

            normal_path = os.path.join(scene_dir, f"normal_{view_index_str}.webp")
            if os.path.exists(normal_path):
                normal: Float[Tensor, "H W 3"] = torch.from_numpy(
                    np.asarray(
                        Image.fromarray(imageio.v2.imread(normal_path))
                        .convert("RGB")
                        .resize((resize_width, resize_height)),
                    )
                    / 255.0
                    * 2.0
                    - 1.0
                ).float()
                normal = F.normalize(normal, dim=-1)
                normal = (normal[:, :, None, :] * c2w[:3, :3]).sum(-1)
                normal = F.normalize(normal, dim=-1)
                normal = normal * 0.5 + 0.5  # output normal in [0, 1]
                normal[~(mask.squeeze(-1) > 0.5)] = 0.0
            else:
                normal = torch.zeros_like(rgb)

            if (
                self.cfg.cond_rgb_perturb_type is not None
                and i < self.cfg.num_views_input
            ):
                trans = transform.Compose(
                    [
                        transform.RandomAffine(
                            degrees=self.cfg.cond_rgb_perturb_scale["rotate"],
                            translate=self.cfg.cond_rgb_perturb_scale["translate"],
                            scale=self.cfg.cond_rgb_perturb_scale["scale"],
                            fill=background_color.tolist(),
                        ),
                    ]
                )
                rgb_perturb = trans(rgb.permute(2, 0, 1)).permute(1, 2, 0)
                if self.cfg.cond_rgb_perturb_type == "all":
                    rgb = rgb_perturb
                elif self.cfg.cond_rgb_perturb_type == "except_first":
                    # use the ground truth rgb for the first conditional frame
                    if i > 0:
                        rgb = rgb_perturb
                else:
                    raise NotImplementedError

            if (
                self.cfg.cond_rgb_swap_patch_type is not None
                and self.cfg.cond_rgb_swap_patches > 0
                and i < self.cfg.num_views_input
            ):
                assert self.cfg.cond_rgb_swap_patch_type in ["all", "except_first"]
                if self.cfg.cond_rgb_swap_patch_type == "all" or (
                    self.cfg.cond_rgb_swap_patch_type == "except_first" and i > 0
                ):
                    for _ in range(self.cfg.cond_rgb_swap_patches):
                        patch_size = self.cfg.cond_rgb_swap_patch_size
                        H, W, _ = rgb.shape
                        # Select two random patch locations
                        x1 = random.randint(0, W - patch_size)
                        y1 = random.randint(0, H - patch_size)
                        x2 = random.randint(0, W - patch_size)
                        y2 = random.randint(0, H - patch_size)

                        # Extract patches
                        patch1 = rgb[y1 : y1 + patch_size, x1 : x1 + patch_size]
                        patch2 = rgb[y2 : y2 + patch_size, x2 : x2 + patch_size]

                        # Swap patches
                        rgb[y1 : y1 + patch_size, x1 : x1 + patch_size] = patch2
                        rgb[y2 : y2 + patch_size, x2 : x2 + patch_size] = patch1

            intrinsic = get_intrinsic_from_fov(fovy, H=resize_height, W=resize_width)
            focal_length = 0.5 * resize_height / math.tan(0.5 * fovy)
            directions_unnormed: Float[Tensor, "H W 3"] = get_ray_directions(
                H=resize_height,
                W=resize_width,
                focal=focal_length,
                normalize=False,
            )
            directions: Float[Tensor, "H W 3"] = F.normalize(
                directions_unnormed, dim=-1
            )
            rays_o, rays_d_unnormed = get_rays(directions_unnormed, c2w, keepdim=True)
            _, rays_d = get_rays(directions, c2w, keepdim=True)

            camera_positions: Float[Tensor, "3"] = c2w[0:3, 3]

            proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
                fovy, resize_width / resize_height, 0.1, 1000.0
            )
            mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(c2w, proj_mtx)

            # Random crop image patches for rendering supervison
            if crop_height < resize_height:
                # if random.random() < 0.6: # upper body crop to sample more head image.
                #     a0 = np.random.randint(0, max(1, resize_height // 2 - crop_height // 2))
                #     a1 = a0 + crop_height
                #     b0 = max(0, resize_width // 2 - crop_width // 2)
                #     b1 = b0 + crop_width
                # else:
                a0 = np.random.randint(0, resize_height - crop_height + 1)
                a1 = a0 + crop_height
                b0 = np.random.randint(0, resize_width - crop_width + 1)
                b1 = b0 + crop_width
                rgb = rgb[a0:a1, b0:b1]
                mask = mask[a0:a1, b0:b1]
                depth = depth[a0:a1, b0:b1]
                normal = normal[a0:a1, b0:b1]
                rays_o = rays_o[a0:a1, b0:b1]
                rays_d = rays_d[a0:a1, b0:b1]
                rays_d_unnormed = rays_d_unnormed[a0:a1, b0:b1]
                intrinsic[..., 0, 2] -= b0
                intrinsic[..., 1, 2] -= a0

            # Calc normalized intrinsic by crop size (by image height)
            intrinsic_normed = intrinsic.clone()
            intrinsic_normed[..., 0, 2] /= crop_width
            intrinsic_normed[..., 1, 2] /= crop_height
            intrinsic_normed[..., 0, 0] /= crop_width
            intrinsic_normed[..., 1, 1] /= crop_height

            data_cur.append(
                {
                    "view_index": torch.as_tensor(int(view_index)),
                    "rgb": rgb,
                    "mask": mask,
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rays_d_unnormed": rays_d_unnormed,
                    "depth": depth,
                    "normal": normal,
                    "camera_positions": camera_positions,
                    "mvp_mtx": mvp_mtx,
                    "c2w": c2w,
                    "intrinsic": intrinsic,
                    "intrinsic_normed": intrinsic_normed,
                }
            )
        data_out = {}
        for k in data_cond[0].keys():
            data_out[k + "_cond"] = torch.stack([d[k] for d in data_cond], dim=0)
        for k in data_sup[0].keys():
            data_out[k] = torch.stack([d[k] for d in data_sup], dim=0)

        out = {
            **data_out,
            "index": torch.as_tensor(scene_index),
            "scene_id": scene_id,
            "background_color": background_color,
        }

        return out

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        if self.split == "train":
            batch.update({"height": self.cfg.height, "width": self.cfg.width})
        else:
            batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


class ObjaverseOrbitDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: ObjaverseDataModuleConfig = cfg

        self.split = split
        self.n_views = self.cfg.n_test_views

        self.root_dir = self.cfg.root_dir[0]

        if split == "train":
            json_path = os.path.join(self.root_dir, "imagedream.jsonl")
            all_data = open(json_path, "r").readlines()
            self.all_dirs = []
            for data in all_data:
                data = json.loads(data)
                self.all_dirs.append(data["image_dir"] + "_a_pose")
        else:
            json_path = os.path.join(self.root_dir, "valid_meta.jsonl")
            all_data = open(json_path, "r").readlines()
            self.all_dirs = []
            for data in all_data:
                data = json.loads(data)
                self.all_dirs.append(data["image_dir"] + "_a_pose")

        # if self.split == "test" and self.cfg.test_indices is not None:
        #     self.all_scenes = self.all_scenes[
        #         self.cfg.test_indices[0] : self.cfg.test_indices[1]
        #     ]
        # else:
        #     raise ValueError("Only support split in ['test'].")

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
        else:
            azimuth_deg = torch.linspace(0, 360.0, self.n_views)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.n_views, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height,
            W=self.cfg.eval_width,
            focal=1.0,
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )
        # must use normalize=True to normalize directions here
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        intrinsic: Float[Tensor, "B 3 3"] = get_intrinsic_from_fov(
            self.cfg.eval_fovy_deg * math.pi / 180,
            H=self.cfg.eval_height,
            W=self.cfg.eval_width,
            bs=self.n_views,
        )
        intrinsic_normed: Float[Tensor, "B 3 3"] = intrinsic.clone()
        intrinsic_normed[..., 0, 2] /= self.cfg.eval_width
        intrinsic_normed[..., 1, 2] /= self.cfg.eval_height
        intrinsic_normed[..., 0, 0] /= self.cfg.eval_width
        intrinsic_normed[..., 1, 1] /= self.cfg.eval_height

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.intrinsic = intrinsic
        self.intrinsic_normed = intrinsic_normed
        self.c2w = c2w
        self.camera_positions = camera_positions

        self.background_color = torch.as_tensor(self.cfg.background_color)
        # self.annotation_file = (
        #     "transforms.json" if self.cfg.version == "v1" else "meta.json"
        # )
        # self.frame_namespace = "frames" if self.cfg.version == "v1" else "locations"

        if self.cfg.eval_input_views is None:
            self.view_ids = random.sample(
                range(self.cfg.num_views_per_scene),
                k=self.cfg.num_views_input,
            )
        else:
            self.view_ids = self.cfg.eval_input_views
            assert len(self.view_ids) == self.cfg.num_views_input

    def __len__(self):
        return len(self.all_dirs) * self.n_views

    def __getitem__(self, index):
        scene_index = index // self.n_views
        view_index = index % self.n_views

        scene_dir = self.all_dirs[scene_index]
        scene_id = scene_dir.replace("_a_pose", "")


        # view_index_str = str(view_index).zfill(3)
        data = []
        for input_view_index in self.view_ids:
            view_index_str = str(input_view_index * 2).zfill(3)
            fovy = 40
            img_path = os.path.join(
                self.root_dir, scene_dir, f"{scene_id}_{view_index_str}_rgb.{self.cfg.image_suffix}"
            )
            json_path = os.path.join(
                self.root_dir, scene_dir, f"{scene_id}_{view_index_str}.json"
            )
            metas = json.load(open(json_path, "r"))
            img_cond = torch.from_numpy(
                np.asarray(
                    Image.fromarray(imageio.v2.imread(img_path))
                    .convert("RGBA")
                    .resize((self.cfg.cond_width, self.cfg.cond_height))
                )
                / 255.0
            ).float()
            mask_cond: Float[Tensor, "Hc Wc 1"] = img_cond[:, :, -1:]
            rgb_cond: Float[Tensor, "Hc Wc 3"] = img_cond[
                :, :, :3
            ] * mask_cond + self.background_color[None, None, :] * (1 - mask_cond)
            c2w_cond = torch.as_tensor(
                np.array(metas["extrinsicMatrix"]["elements"]).astype(np.float32).reshape(4, 4).transpose(1, 0)[[0,2,1,3]],
                dtype=torch.float32,
            )
            data.append({"rgb_cond": rgb_cond, "c2w_cond": c2w_cond})

        data_aggr = {}
        for k in data[0].keys():
            data_aggr[k] = torch.stack([d[k] for d in data], dim=0)

        out = {
            "view_index": torch.as_tensor(view_index),
            "rays_o": self.rays_o[view_index],
            "rays_d": self.rays_d[view_index],
            "mvp_mtx": self.mvp_mtx[view_index],
            "intrinsic": self.intrinsic[view_index],
            "intrinsic_normed": self.intrinsic_normed[view_index],
            "c2w": self.c2w[view_index],
            "camera_positions": self.camera_positions[view_index],
        }
        out = {k: v.unsqueeze(0) for k, v in out.items()}  # [1, D] for each item
        out.update({**data_aggr})
        out["index"] = torch.as_tensor(scene_index)
        return out

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


class ObjaverseDataModule(pl.LightningDataModule):
    cfg: ObjaverseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjaverseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ObjaverseDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjaverseDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ObjaverseOrbitDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = {
        "root_dir": "/mnt/pfs/data/lvis_100view_new",
        "scene_list": "/mnt/pfs/users/guoyuanchen/oh-my-data/lvis_100view_min_mask_percent_0.002.json",
        "version": "v2",
        "num_views_per_scene": 100,
        "num_views_input": 4,
        "background_color": [1.0, 1.0, 1.0],
        "train_indices": [0, 100],
        "val_indices": [100, 110],
        "test_indices": [110, 120],
        "height": 128,
        "width": 128,
        "rand_max_height": 128,
        "rand_max_width": 128,
        "batch_size": 1,
        "eval_height": 512,
        "eval_width": 512,
        "eval_batch_size": 1,
    }
    datasets = {
        "random": ObjaverseDataset(
            ObjaverseDataModuleConfig(
                **conf, train_sup_views="random", num_views_output=4
            ),
            split="train",
        ),
        "random_remain": ObjaverseDataset(
            ObjaverseDataModuleConfig(
                **conf, train_sup_views="random_remain", num_views_output=4
            ),
            split="train",
        ),
        "random_instant3d": ObjaverseDataset(
            ObjaverseDataModuleConfig(
                **conf, train_sup_views="random_instant3d", num_views_output=4
            ),
            split="train",
        ),
        "random_include_input": ObjaverseDataset(
            ObjaverseDataModuleConfig(
                **conf, train_sup_views="random_include_input", num_views_output=8
            ),
            split="train",
        ),
    }

    for name, dataset in datasets.items():
        print(name)
        for i in range(10):
            dataset[i]

    import pdb

    pdb.set_trace()
