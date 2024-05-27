from dataclasses import dataclass, field
from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
import nerfacc
from nerfacc.data_specs import RayIntervals
from nerfacc.pdf import importance_sampling
from einops import rearrange, reduce

import lrm
from ..renderers.base import BaseRenderer
from ...utils.ops import (
    validate_empty_rays,
    chunk_batch,
    get_activation,
    scale_tensor,
    rays_intersect_bbox,
)
from ...utils.typing import *


class TriplaneNeRFProposalRenderer(BaseRenderer):
    @dataclass
    class Config(BaseRenderer.Config):
        feature_reduction: str = "concat"
        density_activation: str = "trunc_exp"
        density_bias: float = -1.0

        num_coarse_samples_per_ray: int = 32
        num_fine_samples_per_ray: int = 32
        sampling_type: str = "uniform"
        randomized: bool = True
        eval_chunk_size: int = 0

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        super().configure(*args, **kwargs)
        assert self.cfg.feature_reduction in ["concat", "mean"]

    def query_triplane(
        self,
        positions: Float[Tensor, "*B N 3"],
        triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
        coarse: bool,
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

        net_out: Dict[str, Float[Tensor, "B N ..."]] = self.decoder(
            out, exclude=["density"] if coarse else ["density_coarse"]
        )
        if coarse:
            net_out["density"] = net_out.pop("density_coarse")
        net_out["density"] = get_activation(self.cfg.density_activation)(
            net_out["density"] + self.cfg.density_bias
        )

        if not batched:
            net_out = {k: v.squeeze(0) for k, v in net_out.items()}

        return net_out

    def render_core(
        self,
        rays_o: Float[Tensor, "B Nv H W 3"],
        rays_d: Float[Tensor, "B Nv H W 3"],
        t_starts: Float[Tensor, "B Nv H W Ns 1"],
        t_ends: Float[Tensor, "B Nv H W Ns 1"],
        triplanes: Float[Tensor, "B 3 Cp Hp Wp"],
        coarse: bool,
    ):
        assert rays_o.shape == rays_d.shape and t_starts.shape == t_ends.shape
        assert rays_o.shape[:4] == t_starts.shape[:4]
        B, n_views, H, W, n_samples, _ = t_starts.shape
        n_all_rays = B * n_views * H * W
        n_rays = n_views * H * W

        # Flatten
        rays_o = rays_o.reshape(B, n_rays, 3)
        rays_d = rays_d.reshape(B, n_rays, 3)
        t_starts = t_starts.reshape(B, n_rays, n_samples, 1)
        t_ends = t_ends.reshape(B, n_rays, n_samples, 1)

        t_mids = (t_starts + t_ends) * 0.5
        t_dirs = rays_d[:, :, None, :].repeat(1, 1, n_samples, 1)

        sampled_pts = rays_o[:, :, None, :] + t_mids * rays_d[:, :, None, :]

        query_call = partial(self.query_triplane, triplanes=triplanes, coarse=coarse)

        if self.training:
            mlp_out = self.query_triplane(
                positions=sampled_pts.view(B, n_rays * n_samples, 3),
                triplanes=triplanes,
                coarse=coarse,
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=sampled_pts,
                features=mlp_out["features"].view(B, n_rays, n_samples, -1),
            )
            comp_rgb_bg = self.background(dirs=rays_d.view(n_all_rays, 3))
        else:
            mlp_out = chunk_batch(
                query_call,
                self.cfg.eval_chunk_size,
                sampled_pts.view(B, n_rays * n_samples, 3),
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=sampled_pts,
                features=mlp_out["features"].view(B, n_rays, n_samples, -1),
            )
            comp_rgb_bg = chunk_batch(
                self.background,
                self.cfg.eval_chunk_size,
                dirs=rays_d.view(n_all_rays, 3),
            )

        weights: Float[Tensor, "(B Nr) Ns 1"]
        trans: Float[Tensor, "(B Nr) Ns 1"]

        weights_, trans_, _ = nerfacc.render_weight_from_density(
            t_starts.reshape(n_all_rays, n_samples),
            t_ends.reshape(n_all_rays, n_samples),
            mlp_out["density"][..., 0].reshape(n_all_rays, n_samples),
        )

        weights, trans = weights_[..., None], trans_[..., None]

        opacity: Float[Tensor, "(B Nr) 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None
        )
        # FIXME: this depth value does not correspond to the one provided by the dataset!
        depth: Float[Tensor, "(B Nr) 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=t_mids.reshape(n_all_rays, n_samples, 1),
        )
        comp_rgb_fg: Float[Tensor, "(B Nr) Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=rgb_fg_all.reshape(n_all_rays, n_samples, -1),
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
        out = {k: v.view(B, n_views, H, W, -1) for k, v in out.items()}

        sample_out = {
            "weights": weights.reshape(B, n_views, H, W, n_samples, 1),
            "trans": trans.reshape(B, n_views, H, W, n_samples, 1),
        }

        out.update(sample_out)
        return out

    def forward(
        self,
        triplanes: Float[Tensor, "B 3 Cp Hp Wp"],
        rays_o: Float[Tensor, "B Nv H W 3"],
        rays_d: Float[Tensor, "B Nv H W 3"],
        **kwargs,
    ) -> Dict[str, Tensor]:
        B, n_views, H, W, _ = rays_d.shape
        n_coarse_samples = self.cfg.num_coarse_samples_per_ray
        n_fine_samples = self.cfg.num_fine_samples_per_ray

        rays_o_flatten = rays_o.reshape(-1, 3)
        rays_d_flatten = rays_d.reshape(-1, 3)
        n_all_rays = rays_o_flatten.shape[0]

        t_near, t_far, rays_valid = rays_intersect_bbox(
            rays_o_flatten, rays_d_flatten, self.cfg.radius
        )

        # First stage: coarse uniform sampling
        with torch.no_grad():
            cdfs = torch.cat(
                [
                    torch.zeros((n_all_rays, 1), device=self.device),
                    torch.ones((n_all_rays, 1), device=self.device),
                ],
                dim=-1,
            )
            intervals = RayIntervals(vals=cdfs)
            intervals, _ = importance_sampling(
                intervals, cdfs, n_coarse_samples, stratified=self.cfg.randomized
            )
            t_vals = _transform_stot(
                self.cfg.sampling_type,
                intervals.vals,
                t_near.reshape(n_all_rays, 1),
                t_far.reshape(n_all_rays, 1),
            )

        t_starts, t_ends = t_vals[..., :-1], t_vals[..., 1:]

        coarse_out = self.render_core(
            rays_o=rays_o,
            rays_d=rays_d,
            t_starts=t_starts.reshape(B, n_views, H, W, n_coarse_samples, 1),
            t_ends=t_ends.reshape(B, n_views, H, W, n_coarse_samples, 1),
            triplanes=triplanes,
            coarse=True,
        )

        # Second stage: importance sampling
        with torch.no_grad():
            coarse_trans = (
                coarse_out["trans"].reshape(n_all_rays, n_coarse_samples).detach()
            )
            cdfs = 1.0 - torch.cat(
                [coarse_trans, torch.zeros(n_all_rays, 1, device=self.device)], dim=-1
            )
            intervals, _ = importance_sampling(
                intervals, cdfs, n_fine_samples, stratified=self.cfg.randomized
            )
            t_vals = _transform_stot(
                self.cfg.sampling_type,
                intervals.vals,
                t_near.reshape(n_all_rays, 1),
                t_far.reshape(n_all_rays, 1),
            )

        t_starts, t_ends = t_vals[..., :-1], t_vals[..., 1:]

        fine_out = self.render_core(
            rays_o=rays_o,
            rays_d=rays_d,
            t_starts=t_starts.reshape(B, n_views, H, W, n_fine_samples, 1),
            t_ends=t_ends.reshape(B, n_views, H, W, n_fine_samples, 1),
            triplanes=triplanes,
            coarse=False,
        )

        out = fine_out
        for k, v in coarse_out.items():
            out[k + "_coarse"] = v

        return out

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()


def _transform_stot(
    transform_type: Literal["uniform", "lindisp"],
    s_vals: torch.Tensor,
    t_min: torch.Tensor,
    t_max: torch.Tensor,
) -> torch.Tensor:
    if transform_type == "uniform":
        _contract_fn, _icontract_fn = lambda x: x, lambda x: x
    elif transform_type == "lindisp":
        _contract_fn, _icontract_fn = lambda x: 1 / x, lambda x: 1 / x
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")
    s_min, s_max = _contract_fn(t_min), _contract_fn(t_max)
    icontract_fn = lambda s: _icontract_fn(s * s_max + (1 - s) * s_min)
    return icontract_fn(s_vals)
