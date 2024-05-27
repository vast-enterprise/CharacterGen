import json
import math
import os
import random
from dataclasses import dataclass, field

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


def _parse_scene_list_single(scene_list_path: str, root_data_dir: str):
    if scene_list_path.endswith(".json"):
        with open(scene_list_path) as f:
            all_scenes = [
                os.path.join(root_data_dir, p[:2], p) for p in json.loads(f.read())
            ]
    elif scene_list_path.endswith(".txt"):
        with open(scene_list_path) as f:
            all_scenes = [os.path.join(root_data_dir, p.strip()) for p in f.readlines()]
    else:
        raise NotImplementedError

    return all_scenes


def _parse_scene_list(
    scene_list_path: Union[str, List[str]], root_data_dir: Union[str, List[str]]
):
    all_scenes = []
    if isinstance(scene_list_path, str):
        scene_list_path = [scene_list_path]
    if isinstance(root_data_dir, str):
        root_data_dir = [root_data_dir]
    for scene_list_path_, root_data_dir_ in zip(scene_list_path, root_data_dir):
        all_scenes += _parse_scene_list_single(scene_list_path_, root_data_dir_)
    return all_scenes


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
    requires_normal: bool = True
    requires_depth: bool = True

    num_views_per_scene: int = 24
    num_views_input: int = 1
    num_views_output: int = 3

    relative_pose: bool = False
    relative_pose_type: str = "simple"
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
    cond_background_perturb: Optional[str] = None

    # extra condition
    extra_cond_root_dir: Optional[Any] = None
    extra_cond_type: str = ""
    extra_cond_relative_pose: bool = False
    extra_cond_prob: float = 0.0
    extra_cond_requires_normal: bool = False
    extra_cond_requires_depth: bool = False

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
        assert split in ["train", "val", "test"]
        self.cfg: ObjaverseDataModuleConfig = cfg
        self.all_scenes = _parse_scene_list(self.cfg.scene_list, self.cfg.root_dir)

        if self.cfg.extra_cond_root_dir is None:
            self.all_scenes_extra_cond = None
        else:
            self.all_scenes_extra_cond = _parse_scene_list(
                self.cfg.scene_list, self.cfg.extra_cond_root_dir
            )

        self.background_color = torch.as_tensor(self.cfg.background_color)
        self.split = split
        if self.split == "train" and self.cfg.train_indices is not None:
            self.all_scenes = self.all_scenes[
                self.cfg.train_indices[0] : self.cfg.train_indices[1]
            ]
            self.all_scenes = self.all_scenes * self.cfg.repeat
            if self.all_scenes_extra_cond is not None:
                self.all_scenes_extra_cond = self.all_scenes_extra_cond[
                    self.cfg.train_indices[0] : self.cfg.train_indices[1]
                ]
                self.all_scenes_extra_cond = (
                    self.all_scenes_extra_cond * self.cfg.repeat
                )
        elif self.split == "val" and self.cfg.val_indices is not None:
            self.all_scenes = self.all_scenes[
                self.cfg.val_indices[0] : self.cfg.val_indices[1]
            ]
            if self.all_scenes_extra_cond is not None:
                self.all_scenes_extra_cond = self.all_scenes_extra_cond[
                    self.cfg.val_indices[0] : self.cfg.val_indices[1]
                ]
        elif self.split == "test" and self.cfg.test_indices is not None:
            self.all_scenes = self.all_scenes[
                self.cfg.test_indices[0] : self.cfg.test_indices[1]
            ]
            if self.all_scenes_extra_cond is not None:
                self.all_scenes_extra_cond = self.all_scenes_extra_cond[
                    self.cfg.test_indices[0] : self.cfg.test_indices[1]
                ]
        self.annotation_file = (
            "transforms.json" if self.cfg.version == "v1" else "meta.json"
        )
        self.frame_namespace = "frames" if self.cfg.version == "v1" else "locations"

    def __len__(self):
        return len(self.all_scenes)

    def load_data_extra_cond(
        self,
        is_cond,
        scene_index,
        view_index,
        size,
        background_color,
        cond_id=None,
        crop_size=None,
        ref_c2w=None,
    ):
        if self.cfg.extra_cond_type not in ["64v", "20v"]:
            raise NotImplementedError

        scene_dir = self.all_scenes_extra_cond[scene_index]
        with open(os.path.join(scene_dir, self.annotation_file)) as f:
            transforms = json.loads(f.read())

        view_index_str = transforms[self.frame_namespace][view_index]["name"]
        # FIXME: hard-coded
        fovy = (
            0.6911111611634243
            if "camera_angle_x" not in transforms
            else transforms["camera_angle_x"]
        )
        img_path = os.path.join(
            scene_dir, f"render_{view_index_str}.{self.cfg.image_suffix}"
        )
        img = torch.from_numpy(
            np.asarray(
                Image.fromarray(imageio.v2.imread(img_path))
                .convert("RGBA")
                .resize(size)
            )
            / 255.0
        ).float()
        mask: Float[Tensor, "H W 1"] = img[:, :, -1:]
        H, W, _ = img.shape
        background: Float[Tensor, "H W 3"] = background_color[None, None, :].expand(
            H, W, -1
        )
        if is_cond and self.cfg.cond_background_perturb is not None:
            if self.cfg.cond_background_perturb.startswith("U"):
                scale = int(self.cfg.cond_background_perturb[1:])
                background = (
                    background + torch.rand_like(background) * scale / 255.0
                ).clamp(0.0, 1.0)
            elif self.cfg.cond_background_perturb.startswith("N"):
                scale = int(self.cfg.cond_background_perturb[1:])
                background = (
                    background + torch.randn_like(background) * scale / 255.0
                ).clamp(0.0, 1.0)
            elif self.cfg.cond_background_perturb.startswith("R256D16N"):
                if random.random() < 0.5:
                    scale = int(self.cfg.cond_background_perturb[8:])
                    random_size = random.randint(256, self.cfg.cond_width)
                    mask256 = cv2.resize(
                        (mask[..., 0] > 0).numpy().astype(np.uint8) * 255,
                        (random_size, random_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    dilate_size = max(int(abs(torch.randn(1).item()) * 16), 1)
                    mask256_dilate = (
                        torch.from_numpy(
                            cv2.dilate(
                                mask256,
                                np.ones((dilate_size, dilate_size), dtype=np.uint8),
                            )
                        ).float()
                        / 255.0
                    )[..., None]
                    background = background_color[None, None, :].expand(random_size, random_size, -1)
                    background = (background + torch.randn_like(background) * 4 / 255.0).clamp(0.0, 1.0)
                    background_perturb = (
                        background + torch.randn_like(background) * scale / 255.0
                    ).clamp(0.0, 1.0)
                    background = (
                        background * (1.0 - mask256_dilate)
                        + background_perturb * mask256_dilate
                    )
                    background = (
                        F.interpolate(
                            background[None].permute(0, 3, 1, 2),
                            (self.cfg.cond_height, self.cfg.cond_width),
                            mode="bicubic",
                        )
                        .clamp(0.0, 1.0)
                        .permute(0, 2, 3, 1)[0]
                    )
            else:
                raise NotImplementedError
        rgb: Float[Tensor, "H W 3"] = img[:, :, :3] * mask + background * (1 - mask)

        depth_path = os.path.join(scene_dir, f"depth_{view_index_str}.exr")
        if os.path.exists(depth_path) and self.cfg.extra_cond_requires_depth:
            depth: Float[Tensor, "H W 1"] = torch.from_numpy(
                cv2.resize(
                    np.asarray(imageio.v2.imread(depth_path)),
                    size,
                )[..., 0:1]
            )
            mask[depth > 100.0] = 0.0
            depth[~(mask > 0.5)] = 0.0  # set invalid depth to 0
        else:
            depth = torch.zeros_like(mask)

        c2w = torch.as_tensor(
            transforms[self.frame_namespace][view_index]["transform_matrix"],
            dtype=torch.float32,
        )

        if self.cfg.relative_pose:
            if ref_c2w is None:
                # use the first frame as the reference frame
                ref_c2w = c2w.clone()
            if self.cfg.relative_pose_type == "simple":
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
            elif self.cfg.relative_pose_type == "azimuth":
                _, ref_azimuth, _ = c2w_to_polar(ref_c2w)
                elevation, azimuth, distance = c2w_to_polar(c2w)
                c2w = polar_to_c2w(elevation, azimuth - ref_azimuth, distance)
            else:
                raise NotImplementedError

        normal_path = os.path.join(scene_dir, f"normal_{view_index_str}.webp")
        if os.path.exists(normal_path) and self.cfg.extra_cond_requires_normal:
            normal: Float[Tensor, "H W 3"] = torch.from_numpy(
                np.asarray(
                    Image.fromarray(imageio.v2.imread(normal_path))
                    .convert("RGB")
                    .resize(size)
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

        width, height = size
        intrinsic = get_intrinsic_from_fov(fovy, H=height, W=width)

        intrinsic_normed = intrinsic.clone()
        intrinsic_normed[..., 0, 2] /= width
        intrinsic_normed[..., 1, 2] /= height
        intrinsic_normed[..., 0, 0] /= width
        intrinsic_normed[..., 1, 1] /= height

        focal_length = 0.5 * height / math.tan(0.5 * fovy)
        directions_unnormed: Float[Tensor, "H W 3"] = get_ray_directions(
            H=height,
            W=width,
            focal=focal_length,
            normalize=False,
        )
        directions: Float[Tensor, "H W 3"] = F.normalize(directions_unnormed, dim=-1)
        rays_o, rays_d_unnormed = get_rays(directions_unnormed, c2w, keepdim=True)
        _, rays_d = get_rays(directions, c2w, keepdim=True)

        camera_positions: Float[Tensor, "3"] = c2w[0:3, 3]

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            fovy, width / height, 0.1, 1000.0
        )
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(c2w, proj_mtx)

        if crop_size is not None:
            crop_width, crop_height = crop_size
            a0 = np.random.randint(0, height - crop_height + 1)
            a1 = a0 + crop_height
            b0 = np.random.randint(0, width - crop_width + 1)
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

        return {
            "rgb": rgb,
            "mask": mask,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "rays_d_unnormed": rays_d_unnormed,
            "depth": depth,
            "normal": normal,
            "intrinsic": intrinsic,
            "intrinsic_normed": intrinsic_normed,
            "camera_positions": camera_positions,
            "mvp_mtx": mvp_mtx,
            "c2w": c2w,
            "ref_c2w": ref_c2w,
        }

    def load_data(
        self,
        is_cond,
        scene_index,
        view_index,
        size,
        background_color,
        cond_id=None,
        crop_size=None,
        ref_c2w=None,
    ):
        scene_dir = self.all_scenes[scene_index]
        with open(os.path.join(scene_dir, self.annotation_file)) as f:
            transforms = json.loads(f.read())

        view_index_str = transforms[self.frame_namespace][view_index]["index"]
        fovy = transforms["camera_angle_x"]
        img_path = os.path.join(
            scene_dir, f"render_{view_index_str}.{self.cfg.image_suffix}"
        )
        img = torch.from_numpy(
            np.asarray(
                Image.fromarray(imageio.v2.imread(img_path))
                .convert("RGBA")
                .resize(size)
            )
            / 255.0
        ).float()
        mask: Float[Tensor, "H W 1"] = img[:, :, -1:]
        H, W, _ = img.shape
        background: Float[Tensor, "H W 3"] = background_color[None, None, :].expand(
            H, W, -1
        )
        if is_cond and self.cfg.cond_background_perturb is not None:
            if self.cfg.cond_background_perturb.startswith("U"):
                scale = int(self.cfg.cond_background_perturb[1:])
                background = (
                    background + torch.rand_like(background) * scale / 255.0
                ).clamp(0.0, 1.0)
            elif self.cfg.cond_background_perturb.startswith("N"):
                scale = int(self.cfg.cond_background_perturb[1:])
                background = (
                    background + torch.randn_like(background) * scale / 255.0
                ).clamp(0.0, 1.0)
            elif self.cfg.cond_background_perturb.startswith("R256D16N"):
                if random.random() < 0.5:
                    scale = int(self.cfg.cond_background_perturb[8:])
                    random_size = random.randint(256, self.cfg.cond_width)
                    mask256 = cv2.resize(
                        (mask[..., 0] > 0).numpy().astype(np.uint8) * 255,
                        (random_size, random_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    dilate_size = max(int(abs(torch.randn(1).item()) * 16), 1)
                    mask256_dilate = (
                        torch.from_numpy(
                            cv2.dilate(
                                mask256,
                                np.ones((dilate_size, dilate_size), dtype=np.uint8),
                            )
                        ).float()
                        / 255.0
                    )[..., None]
                    background = background_color[None, None, :].expand(random_size, random_size, -1)
                    background = (background + torch.randn_like(background) * 4 / 255.0).clamp(0.0, 1.0)
                    background_perturb = (
                        background + torch.randn_like(background) * scale / 255.0
                    ).clamp(0.0, 1.0)
                    background = (
                        background * (1.0 - mask256_dilate)
                        + background_perturb * mask256_dilate
                    )
                    background = (
                        F.interpolate(
                            background[None].permute(0, 3, 1, 2),
                            (self.cfg.cond_height, self.cfg.cond_width),
                            mode="bicubic",
                        )
                        .clamp(0.0, 1.0)
                        .permute(0, 2, 3, 1)[0]
                    )
                else:
                    scale = 4  # hard-coded scale
                    background = (
                        background + torch.randn_like(background) * scale / 255.0
                    ).clamp(0.0, 1.0)                    
            else:
                raise NotImplementedError
        rgb: Float[Tensor, "H W 3"] = img[:, :, :3] * mask + background * (1 - mask)

        depth_path = os.path.join(scene_dir, f"depth_{view_index_str}.exr")
        if os.path.exists(depth_path):
            depth: Float[Tensor, "H W 1"] = torch.from_numpy(
                cv2.resize(
                    np.asarray(imageio.v2.imread(depth_path)),
                    size,
                )[..., 0:1]
            )
            mask[depth > 100.0] = 0.0
            depth[~(mask > 0.5)] = 0.0  # set invalid depth to 0
        else:
            depth = torch.zeros_like(mask)

        c2w = torch.as_tensor(
            transforms[self.frame_namespace][view_index]["transform_matrix"],
            dtype=torch.float32,
        )

        if self.cfg.cond_c2w_perturb_type is not None and is_cond:
            elevation, azimuth, distance = c2w_to_polar(c2w)
            # print(f'before perturb {elevation * 180. / math.pi:.1f} {azimuth * 180. / math.pi:.1f} {distance:.2f}')
            scales = torch.randn(3) * torch.as_tensor(self.cfg.cond_c2w_perturb_scale)
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
                if cond_id > 0:
                    c2w = c2w_perturb
            elif self.cfg.cond_c2w_perturb_type == "zero_except_first":
                if cond_id > 0:
                    c2w = torch.zeros_like(c2w)
            else:
                raise NotImplementedError

        if self.cfg.relative_pose:
            if ref_c2w is None:
                # use the first frame as the reference frame
                ref_c2w = c2w.clone()
            if self.cfg.relative_pose_type == "simple":
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
            elif self.cfg.relative_pose_type == "azimuth":
                _, ref_azimuth, _ = c2w_to_polar(ref_c2w)
                elevation, azimuth, distance = c2w_to_polar(c2w)
                c2w = polar_to_c2w(elevation, azimuth - ref_azimuth, distance)
            else:
                raise NotImplementedError

        normal_path = os.path.join(scene_dir, f"normal_{view_index_str}.webp")
        if os.path.exists(normal_path):
            normal: Float[Tensor, "H W 3"] = torch.from_numpy(
                np.asarray(
                    Image.fromarray(imageio.v2.imread(normal_path))
                    .convert("RGB")
                    .resize(size)
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

        if self.cfg.cond_rgb_perturb_type is not None and is_cond:
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
                if cond_id > 0:
                    rgb = rgb_perturb
            else:
                raise NotImplementedError

        if (
            self.cfg.cond_rgb_swap_patch_type is not None
            and self.cfg.cond_rgb_swap_patches > 0
            and is_cond
        ):
            assert self.cfg.cond_rgb_swap_patch_type in ["all", "except_first"]
            if self.cfg.cond_rgb_swap_patch_type == "all" or (
                self.cfg.cond_rgb_swap_patch_type == "except_first" and cond_id > 0
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

        width, height = size
        intrinsic = get_intrinsic_from_fov(fovy, H=height, W=width)

        intrinsic_normed = intrinsic.clone()
        intrinsic_normed[..., 0, 2] /= width
        intrinsic_normed[..., 1, 2] /= height
        intrinsic_normed[..., 0, 0] /= width
        intrinsic_normed[..., 1, 1] /= height

        focal_length = 0.5 * height / math.tan(0.5 * fovy)
        directions_unnormed: Float[Tensor, "H W 3"] = get_ray_directions(
            H=height,
            W=width,
            focal=focal_length,
            normalize=False,
        )
        directions: Float[Tensor, "H W 3"] = F.normalize(directions_unnormed, dim=-1)
        rays_o, rays_d_unnormed = get_rays(directions_unnormed, c2w, keepdim=True)
        _, rays_d = get_rays(directions, c2w, keepdim=True)

        camera_positions: Float[Tensor, "3"] = c2w[0:3, 3]

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            fovy, width / height, 0.1, 1000.0
        )
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(c2w, proj_mtx)

        if crop_size is not None:
            crop_width, crop_height = crop_size
            a0 = np.random.randint(0, height - crop_height + 1)
            a1 = a0 + crop_height
            b0 = np.random.randint(0, width - crop_width + 1)
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

        return {
            "rgb": rgb,
            "mask": mask,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "rays_d_unnormed": rays_d_unnormed,
            "depth": depth,
            "normal": normal,
            "intrinsic": intrinsic,
            "intrinsic_normed": intrinsic_normed,
            "camera_positions": camera_positions,
            "mvp_mtx": mvp_mtx,
            "c2w": c2w,
            "ref_c2w": ref_c2w,
        }

    def __getitem__(self, index):
        scene_index = index
        if self.cfg.return_first_n_cases > 0:
            scene_index %= self.cfg.return_first_n_cases

        if self.split != "train" or self.cfg.train_random_background is None:
            background_color = self.background_color
        elif self.cfg.train_random_background == "random_gray":
            background_color = torch.as_tensor(
                [random.random()] * 3, dtype=torch.float32
            )
        elif self.cfg.train_random_background == "random":
            background_color = torch.from_numpy(np.random.random(3)).float()

        use_extra_cond = False
        if self.cfg.train_input_views is not None:
            assert len(self.cfg.train_input_views) == self.cfg.num_views_input
            assert self.cfg.extra_cond_root_dir is None
            cond_ids = self.cfg.train_input_views
        else:
            if (
                self.cfg.extra_cond_root_dir is not None
                and random.random() < self.cfg.extra_cond_prob
            ):
                # use extra condition
                use_extra_cond = True
                if self.cfg.extra_cond_type == "64v":
                    assert self.cfg.num_views_input == 4
                    group_id = random.randint(0, 2)
                    cond_ids = list(range(1 + group_id * 16, 1 + group_id * 16 + 16, 4))
                    start_id = random.randint(0, 1) * 2
                    cond_ids = cond_ids[start_id:] + cond_ids[:start_id]
                    num_views_from_extra_cond = 4
                elif self.cfg.extra_cond_type == "20v":
                    assert self.cfg.num_views_input == 4
                    group_id = random.randint(0, 3)
                    cond_ids = list(range(1 + group_id * 5, 1 + group_id * 5 + 4))
                    start_id = random.randint(0, 3)
                    cond_ids = cond_ids[start_id:] + cond_ids[:start_id]
                    num_views_from_extra_cond = 4
                else:
                    raise NotImplementedError
            else:
                cond_ids = np.random.choice(
                    self.cfg.num_views_per_scene,
                    self.cfg.num_views_input,
                    replace=False,
                )
                num_views_from_extra_cond = 0

        if use_extra_cond:
            remain_set = list(set(range(self.cfg.num_views_per_scene)))
        else:
            remain_set = list(set(range(self.cfg.num_views_per_scene)) - set(cond_ids))

        if self.cfg.train_sup_views == "random":
            sup_ids = np.random.choice(
                self.cfg.num_views_per_scene, self.cfg.num_views_output, replace=False
            )
        elif self.cfg.train_sup_views == "random_remain":
            sup_ids = np.random.choice(
                remain_set, self.cfg.num_views_output, replace=False
            )
        elif self.cfg.train_sup_views == "random_instant3d":
            assert self.cfg.extra_cond_root_dir is None
            sup_ids = np.random.choice(
                np.concatenate(
                    [
                        cond_ids,
                        np.random.choice(
                            remain_set, self.cfg.num_views_output, replace=False
                        ),
                    ]
                ),
                self.cfg.num_views_output,
                replace=False,
            )
        elif self.cfg.train_sup_views == "random_instant3d_include_first":
            assert self.cfg.extra_cond_root_dir is None
            sup_ids = np.random.choice(
                np.concatenate(
                    [
                        cond_ids[1:],
                        np.random.choice(
                            remain_set, self.cfg.num_views_output, replace=False
                        ),
                    ]
                ),
                self.cfg.num_views_output - 1,
                replace=False,
            )
            sup_ids = np.concatenate([cond_ids[0:1], sup_ids])
        elif self.cfg.train_sup_views == "random_include_first":
            sup_ids = np.random.choice(
                self.cfg.num_views_per_scene,
                self.cfg.num_views_output - 1,
                replace=False,
            )
            sup_ids = np.concatenate([cond_ids[0:1], sup_ids])
            if use_extra_cond:
                num_views_from_extra_cond += 1
        elif self.cfg.train_sup_views == "random_include_input":
            assert self.cfg.num_views_output >= self.cfg.num_views_input
            sup_ids = np.concatenate(
                [
                    cond_ids,
                    np.random.choice(
                        remain_set,
                        self.cfg.num_views_output - self.cfg.num_views_input,
                        replace=False,
                    ),
                ]
            )
            if use_extra_cond:
                num_views_from_extra_cond += self.cfg.num_views_input
        else:
            raise NotImplementedError

        view_ids = np.concatenate([cond_ids, sup_ids])

        if self.split != "train" and self.cfg.eval_input_views is not None:
            assert len(self.cfg.eval_input_views) == self.cfg.num_views_input
            view_ids[: self.cfg.num_views_input] = self.cfg.eval_input_views

        data_cond, data_sup = [], []

        ref_c2w = None
        for i, view_index in enumerate(view_ids):
            if i < self.cfg.num_views_input:
                is_cond = True
                data_cur = data_cond
                crop_height, crop_width = self.cfg.cond_height, self.cfg.cond_width
                resize_height, resize_width = self.cfg.cond_height, self.cfg.cond_width
                full_height, full_width = None, None
            else:
                is_cond = False
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
                    full_height = self.cfg.rand_max_height
                    full_width = int(
                        np.round(full_height * self.cfg.width / self.cfg.height)
                    )
                else:
                    crop_height, crop_width = self.cfg.eval_height, self.cfg.eval_width
                    resize_height, resize_width = (
                        self.cfg.eval_height,
                        self.cfg.eval_width,
                    )
                    full_height, full_width = (
                        self.cfg.eval_height,
                        self.cfg.eval_width,
                    )

            if not is_cond:
                if i < num_views_from_extra_cond:
                    data_cur_full_ = self.load_data_extra_cond(
                        is_cond,
                        scene_index,
                        view_index,
                        (full_width, full_height),
                        background_color,
                        crop_size=None,
                        ref_c2w=ref_c2w,
                    )
                else:
                    data_cur_full_ = self.load_data(
                        is_cond,
                        scene_index,
                        view_index,
                        (full_width, full_height),
                        background_color,
                        crop_size=None,
                        ref_c2w=ref_c2w,
                    )
            else:
                data_cur_full_ = {}

            if i < num_views_from_extra_cond:
                data_cur_ = self.load_data_extra_cond(
                    is_cond,
                    scene_index,
                    view_index,
                    (resize_width, resize_height),
                    background_color,
                    crop_size=(crop_width, crop_height),
                    ref_c2w=ref_c2w,
                    cond_id=i,
                )
            else:
                data_cur_ = self.load_data(
                    is_cond,
                    scene_index,
                    view_index,
                    (resize_width, resize_height),
                    background_color,
                    crop_size=(crop_width, crop_height),
                    ref_c2w=ref_c2w,
                    cond_id=i,
                )

            if (
                (self.cfg.relative_pose and not use_extra_cond)
                or (self.cfg.extra_cond_relative_pose and use_extra_cond)
                and ref_c2w is None
            ):
                # use c2w of the first frame as the reference frame
                ref_c2w = data_cur_["ref_c2w"]

            data_cur_.pop("ref_c2w", None)
            data_cur_full_.pop("ref_c2w", None)

            data_cur.append(
                {
                    **{f"{k}_full": v for k, v in data_cur_full_.items()},
                    **{k: v for k, v in data_cur_.items()},
                }
            )

        data_out = {}
        for k in data_cond[0].keys():
            data_out[k + "_cond"] = torch.stack([d[k] for d in data_cond], dim=0)
        for k in data_sup[0].keys():
            data_out[k] = torch.stack([d[k] for d in data_sup], dim=0)

        return {
            **data_out,
            "index": torch.as_tensor(scene_index),
            "index_full": torch.as_tensor(scene_index),
            "background_color": background_color,
            "background_color_full": background_color,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        if self.split == "train":
            full_height = self.cfg.rand_max_height
            full_width = int(np.round(full_height * self.cfg.width / self.cfg.height))
            batch.update(
                {
                    "height": self.cfg.height,
                    "width": self.cfg.width,
                    "height_full": full_height,
                    "width_full": full_width,
                }
            )
        else:
            batch.update(
                {
                    "height": self.cfg.eval_height,
                    "width": self.cfg.eval_width,
                    "height_full": self.cfg.eval_height,
                    "width_full": self.cfg.eval_width,
                }
            )
        return batch


class ObjaverseOrbitDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: ObjaverseDataModuleConfig = cfg

        self.split = split
        self.n_views = self.cfg.n_test_views

        self.all_scenes = _parse_scene_list(self.cfg.scene_list, self.cfg.root_dir)

        if self.split == "test" and self.cfg.test_indices is not None:
            self.all_scenes = self.all_scenes[
                self.cfg.test_indices[0] : self.cfg.test_indices[1]
            ]
        else:
            raise ValueError("Only support split in ['test'].")

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
        self.annotation_file = (
            "transforms.json" if self.cfg.version == "v1" else "meta.json"
        )
        self.frame_namespace = "frames" if self.cfg.version == "v1" else "locations"

        if self.cfg.eval_input_views is None:
            self.view_ids = random.sample(
                range(self.cfg.num_views_per_scene),
                k=self.cfg.num_views_input,
            )
        else:
            self.view_ids = self.cfg.eval_input_views
            assert len(self.view_ids) == self.cfg.num_views_input

    def __len__(self):
        return len(self.all_scenes) * self.n_views

    def __getitem__(self, index):
        scene_index = index // self.n_views
        view_index = index % self.n_views

        scene_dir = self.all_scenes[scene_index]
        with open(os.path.join(scene_dir, self.annotation_file)) as f:
            transforms = json.loads(f.read())

        data = []
        for input_view_index in self.view_ids:
            view_index_str = transforms[self.frame_namespace][input_view_index]["index"]
            fovy = transforms["camera_angle_x"]
            img_path = os.path.join(
                scene_dir, f"render_{view_index_str}.{self.cfg.image_suffix}"
            )
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
                transforms[self.frame_namespace][input_view_index]["transform_matrix"],
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

    config = ObjaverseDataModuleConfig(
        root_dir="/mnt/pfs/data/88w_36view_objaverse",
        scene_list="/mnt/pfs/users/guoyuanchen/oh-my-data/objaverse20w_36view_min_mask_percent_0.002_x64view.json",
        version="v2",
        num_views_per_scene=36,
        num_views_input=4,
        num_views_output=6,
        train_sup_views="random_include_input",
        extra_cond_root_dir="/mnt/pfs/data/88w_64view_objaverse_t2",
        extra_cond_relative_pose=True,
        extra_cond_prob=0.5,
        extra_cond_type="64v",
        background_color=[1.0, 1.0, 1.0],
        train_indices=[0, 100],
        val_indices=[100, 110],
        test_indices=[110, 120],
        height=256,
        width=256,
        rand_max_height=256,
        rand_max_width=256,
        batch_size=2,
        eval_height=512,
        eval_width=512,
        eval_batch_size=1,
    )
    dataset_train = ObjaverseDataset(config, split="train")
    dataset_val = ObjaverseDataset(config, split="val")

    import pdb

    pdb.set_trace()
