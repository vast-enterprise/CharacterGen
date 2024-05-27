import json
import math
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
import imageio
import cv2
from torch.utils.data import DataLoader, Dataset
from einops import rearrange

import lrm
from ..utils.config import parse_structured
from ..utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
    get_intrinsic_from_fov,
)
from ..utils.typing import *


@dataclass
class SDXL4VEvalDataModuleConfig:
    root_dir: str = ""
    suffix: str = ""
    n_cases_per_image: int = 4
    resolution: int = 1024
    camera_pose_spec: str = ""

    return_first_n_cases: int = -1  # for debugging purpose

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


class Fixed4VOrbitDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: SDXL4VEvalDataModuleConfig = cfg

        self.n_views = self.cfg.n_test_views
        self.split = split

        self.all_images = [
            f for f in os.listdir(self.cfg.root_dir) if f.endswith(self.cfg.suffix)
        ]
        if self.cfg.return_first_n_cases > 0:
            self.all_images = self.all_images[: self.cfg.return_first_n_cases]

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

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        intrinsic: Float[Tensor, "B 3 3"] = get_intrinsic_from_fov(
            40 * math.pi / 180,
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

        self.c2w_cond = torch.as_tensor(
            json.load(open(self.cfg.camera_pose_spec))
        ).float()

    def __len__(self):
        return len(self.all_images) * self.cfg.n_cases_per_image * self.n_views

    def __getitem__(self, index):
        case_index = index // self.n_views
        view_index = index % self.n_views
        image_index = case_index // self.cfg.n_cases_per_image
        image_case_index = case_index % self.cfg.n_cases_per_image

        image_path = os.path.join(self.cfg.root_dir, self.all_images[image_index])
        name = os.path.basename(image_path).replace(self.cfg.suffix, "")
        image = torch.from_numpy(
            (
                cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(
                    np.float32
                )
                / 255.0
            )
        )
        H, W, _ = image.shape
        assert H % self.cfg.resolution == 0 and W % self.cfg.resolution == 0
        Nh, Nw = H // self.cfg.resolution, W // self.cfg.resolution
        assert Nh * Nw == self.cfg.n_cases_per_image
        images = rearrange(image, "(Nh H) (Nw W) C -> (Nh Nw) H W C", Nh=Nh, Nw=Nw)
        image = images[image_case_index]

        rgb_cond = rearrange(
            image, "(Nh H) (Nw W) C -> (Nh Nw) H W C", Nh=2, Nw=2
        )  # 2x2 grid
        c2w_cond = self.c2w_cond

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
        out.update({"rgb_cond": rgb_cond, "c2w_cond": c2w_cond})
        out["index"] = torch.as_tensor(case_index)
        out["case_index"] = torch.as_tensor(image_case_index)
        out["name"] = name
        return out

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


class SDXL4VDataModule(pl.LightningDataModule):
    cfg: SDXL4VEvalDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SDXL4VEvalDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "test", "predict"]:
            self.test_dataset = SDXL4VOrbitDataset(self.cfg, "test")

    def prepare_data(self):
        pass

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
    dataset = SDXL4VOrbitDataset(
        SDXL4VEvalDataModuleConfig(
            root_dir="/mnt/pfs/users/guoyuanchen/sdxl-finetune/test_results/prompt_ding_results",
            suffix="_ddim50.jpg",
            n_cases_per_image=4,
            resolution=1024,
            camera_pose_spec="/mnt/pfs/users/guoyuanchen/large-reconstruction-model-deploy/c2w_instant3d.json",
        ),
        "test",
    )
    breakpoint()
