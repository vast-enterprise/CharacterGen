from dataclasses import dataclass, field
from collections import defaultdict
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch
import torch.nn as nn
import numpy as np
import math

import lrm
from lrm.utils.typing import *
from lrm.utils.base import BaseModule, BaseObject

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * np.arctan2(w, 2 * fx)
    fov_y = 2 * np.arctan2(h, 2 * fy)
    return fov_x, fov_y


class Camera:
    def __init__(self, R, T, FoVx, FoVy, height, width, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, device="cuda") -> None:
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def from_c2w(c2w, intrinsic, height, width, device):
        c2w[:3, 1:3] *= -1
        w2c = torch.inverse(c2w).cpu().numpy()
        R = w2c[:3,:3].transpose()  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        FoVx, FoVy = intrinsic_to_fov(intrinsic.cpu().numpy(), w=width, h=height)
        return Camera(R=R, T=T, FoVx=FoVx, FoVy=FoVy, height=height, width=width, device=device)

class GaussianModel(NamedTuple):
    xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor

class GSLayer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 128
        out_keys: Tuple[str] = ()
        out_channels: Tuple[int] = ()
        init_scaling: float = -5.0
        init_density: float = 0.1

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        assert len(self.cfg.out_channels) == len(self.cfg.out_keys)

        self.out_layers = nn.ModuleList()
        
        for key, out_ch in zip(self.cfg.out_keys, self.cfg.out_channels):
            layer = nn.Linear(self.cfg.in_channels, out_ch)

            # initialize
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, self.cfg.init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                inverse_sigmoid = lambda x: np.log(x / (1 - x))
                nn.init.constant_(layer.bias, inverse_sigmoid(self.cfg.init_density))

            self.out_layers.append(layer)

    def forward(self, x):
        ret = {}
        for k, layer in zip(self.cfg.out_keys, self.out_layers):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = torch.exp(v)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                v = torch.reshape(v, (v.shape[0], -1, 3))
            ret[k] = v
        return GaussianModel(**ret)

class GS3DRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        # background_cls: str = ""
        # background: dict = field(default_factory=dict)
        gs_out: dict = field(default_factory=dict)
        sh_degree: int = 3
        scaling_modifier: float = 1.0

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        # self.background = lrm.find(self.cfg.background_cls)(self.cfg.background)
        self.gs_net = GSLayer(self.cfg.gs_out)

    def forward_single_view(self,
        gs: GaussianModel,
        viewpoint_camera: Camera
        ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        bg_color = torch.as_tensor([1, 1, 1], dtype=torch.float32, device=self.device)

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.cfg.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.cfg.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        # if pipe.compute_cov3D_python:
        #     cov3D_precomp = pc.get_covariance(scaling_modifier)
        # else:
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        # if override_color is None:
            # if pipe.convert_SHs_python:
            #     shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            #     dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            # else:
        shs = gs.shs
        # else:
            # colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        return {
            "comp_rgb": rendered_image.permute(1, 2, 0)
        }
    
    def forward_single_batch(
        self,
        gs_hidden_features: Float[Tensor, "Np Cp"],
        c2ws: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int
    ):
        gs: GaussianModel = self.gs_net(gs_hidden_features)
        out_list = []
        for c2w, intrinsic in zip(c2ws, intrinsics):
            out_list.append(self.forward_single_view(gs, Camera.from_c2w(c2w, intrinsic, height, width, self.device)))
        
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["points"] = gs.xyz

        return out

    def forward(self, 
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        **kwargs):
        batch_size = gs_hidden_features.shape[0]
        out_list = []
        for b in range(batch_size):
            out_list.append(self.forward_single_batch(gs_hidden_features[b], c2w[b], intrinsic[b], height, width))

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}

        return out
        