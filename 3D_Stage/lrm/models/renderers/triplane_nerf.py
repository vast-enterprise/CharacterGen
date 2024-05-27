from dataclasses import dataclass, field
from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
import nerfacc
from einops import rearrange, reduce

from ..isosurface import MarchingCubeCPUHelper  

import lrm
from ..renderers.base import BaseRenderer
from ...utils.ops import validate_empty_rays, chunk_batch, get_activation, scale_tensor
from ...utils.typing import *


class TriplaneNeRFRenderer(BaseRenderer):
    @dataclass
    class Config(BaseRenderer.Config):
        estimator: str = "occgrid"

        feature_reduction: str = "concat"
        density_activation: str = "trunc_exp"
        density_bias: float = -1.0

        num_samples_per_ray: int = 64
        randomized: bool = True
        eval_chunk_size: int = 0
        isosurface_resolution: int = 256
        isosurface_outlier_n_faces_threshold: float = 0.01

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        super().configure(*args, **kwargs)

        assert self.cfg.feature_reduction in ["concat", "mean"]

        if self.cfg.estimator == "occgrid":
            self.estimator = nerfacc.OccGridEstimator(
                roi_aabb=self.bbox.view(-1), resolution=32, levels=1
            )
            self.estimator.occs.fill_(True)
            self.estimator.binaries.fill_(True)
            self.render_step_size = (
                1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
            )
            self.randomized = self.cfg.randomized
        else:
            raise NotImplementedError(
                "Unknown estimator, should be one of ['occgrid']",
            )
        self.isosurface_helper = MarchingCubeCPUHelper(
            self.cfg.isosurface_resolution
        ).to(self.device)

        self.isosurface_helper.points_range = (-self.cfg.radius, self.cfg.radius)

    def query_triplane(
        self,
        positions: Float[Tensor, "*B N 3"],
        triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
    ) -> Dict[str, Tensor]:
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]
        assert triplanes.ndim == 5 and positions.ndim == 3

        # positions in (-radius, radius)
        # normalized to (-1, 1) for grid sample
        positions = scale_tensor(
            positions, (-self.cfg.radius, self.cfg.radius), (-1, 1)
        )

        indices2D: Float[Tensor, "B 3 N 2"] = torch.stack(
            (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
            dim=-3,
        )
        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3),
            rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3),
            align_corners=False,
            mode="bilinear",
        )
        if self.cfg.feature_reduction == "concat":
            out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
        elif self.cfg.feature_reduction == "mean":
            out = reduce(out, "(B Np) Cp () N -> B N Cp", Np=3, reduction="mean")
        else:
            raise NotImplementedError

        net_out: Dict[str, Float[Tensor, "B N ..."]] = self.decoder(out)
        assert "density" in net_out
        net_out["density"] = get_activation(self.cfg.density_activation)(
            net_out["density"] + self.cfg.density_bias
        )

        if not batched:
            net_out = {k: v.squeeze(0) for k, v in net_out.items()}

        return net_out

    def forward_single(
        self,
        triplane: Float[Tensor, "3 Cp Hp Wp"],
        rays_o: Float[Tensor, "Nv H W 3"],
        rays_d: Float[Tensor, "Nv H W 3"],
        background_color: Optional[Float[Tensor, "3"]],
    ) -> Dict[str, Tensor]:
        Nv, H, W, _ = rays_o.shape
        rays_o_flatten, rays_d_flatten = rays_o.view(-1, 3), rays_d.view(-1, 3)
        n_rays = rays_o_flatten.shape[0]
        with torch.no_grad():
            ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                rays_o_flatten,
                rays_d_flatten,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
            )
        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )

        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        query_call = partial(self.query_triplane, triplanes=triplane)
        if self.training:
            geo_out = query_call(positions)
            rgb_fg_all = self.material(viewdirs=t_dirs, positions=positions, **geo_out)
            comp_rgb_bg = self.background(
                dirs=rays_d_flatten, color_spec=background_color
            )
        else:
            geo_out = chunk_batch(
                query_call,
                self.cfg.eval_chunk_size,
                positions,
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background,
                self.cfg.eval_chunk_size,
                dirs=rays_d_flatten,
                color_spec=background_color,
            )

        weights: Float[Tensor, "Nr 1"]
        weights_, trans_, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        # FIXME: this depth value does not correspond to the one provided by the dataset!
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        bg_color = comp_rgb_bg
        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb,
            "comp_rgb_fg": comp_rgb_fg,
            "comp_rgb_bg": comp_rgb_bg,
            "opacity": opacity,
            "depth": depth,
        }
        out = {k: v.view(Nv, H, W, -1) for k, v in out.items()}
        return out

    def forward(
        self,
        triplanes: Float[Tensor, "B 3 Cp Hp Wp"],
        rays_o: Float[Tensor, "B Nv H W 3"],
        rays_d: Float[Tensor, "B Nv H W 3"],
        background_color: Optional[Float[Tensor, "B 3"]] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        batch_size = triplanes.shape[0]
        out_list = []
        for b in range(batch_size):
            out_list.append(
                self.forward_single(
                    triplanes[b],
                    rays_o[b],
                    rays_d[b],
                    background_color=background_color[b]
                    if background_color is not None
                    else None,
                )
            )

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}

        return out

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()

    def isosurface(self, triplane: Float[Tensor, "3 Cp Hp Wp"]):
        grid_vertices = scale_tensor(
            self.isosurface_helper.grid_vertices.to(triplane.device),
            self.isosurface_helper.points_range,
            (-self.cfg.radius, self.cfg.radius),
        )
        import ipdb
        # ipdb.set_trace()
        triplane_out = chunk_batch(
            partial(self.query_triplane, triplanes=triplane), self.cfg.eval_chunk_size, grid_vertices,
        )

        density = triplane_out["density"]

        density_threshold = density.mean()
        level = -(density - 100)
        # ipdb.set_trace()
        mesh: Mesh = self.isosurface_helper(level)

        # ipdb.set_trace()
        mesh.v_pos = scale_tensor(
            mesh.v_pos, (0, 1), self.bbox
        )
        
        mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)

        import trimesh
        tmesh = trimesh.Trimesh(mesh.v_pos.cpu().numpy(), mesh.t_pos_idx.cpu().numpy())
        tmesh.export("ttt.obj")

        # ipdb.set_trace()
        return mesh

    def query(
        self, triplane: Float[Tensor, "3 Cp Hp Wp"], points: Float[Tensor, "*N 3"]
    ):
        input_shape = points.shape[:-1]
        triplane_out = chunk_batch(
            partial(self.query_triplane, triplanes=triplane), self.cfg.eval_chunk_size, points.view(-1, 3)
        )
        triplane_out = {
            k: v.view(*input_shape, v.shape[-1]) for k, v in triplane_out.items()
        }
        return triplane_out