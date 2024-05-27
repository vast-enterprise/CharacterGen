import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange

import lrm
from lrm.systems.base import BaseLossConfig, BaseSystem
from lrm.utils.ops import binary_cross_entropy, rays_intersect_bbox, get_plucker_rays
from lrm.utils.typing import *
from lrm.models.lpips import LPIPS


@dataclass
class MultiviewLRMHybridLossConfig(BaseLossConfig):
    lambda_mse_nerf: Any = 0.0
    lambda_smooth_l1_nerf: Any = 0.0
    lambda_lpips_nerf: Any = 0.0
    lambda_mask_nerf: Any = 0.0

    lambda_mse_dmtet: Any = 0.0
    lambda_smooth_l1_dmtet: Any = 0.0
    lambda_lpips_dmtet: Any = 0.0
    lambda_mask_dmtet: Any = 0.0
    lambda_depth_l1_dmtet: Any = 0.0
    lambda_normal_cos_dmtet: Any = 0.0
    lambda_sdf_reg: Any = 0.0
    lambda_surface: Any = 0.0
    lambda_outside_space: Any = 0.0
    lambda_inside_space: Any = 0.0
    sdf_supervision_limit: int = 0


class MultiviewLRMHybrid(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        loss: MultiviewLRMHybridLossConfig = MultiviewLRMHybridLossConfig()

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        decoder_cls: str = ""
        decoder: dict = field(default_factory=dict)

        material_cls: str = ""
        material: dict = field(default_factory=dict)

        background_cls: str = ""
        background: dict = field(default_factory=dict)

        renderer_nerf_cls: str = ""
        renderer_nerf: dict = field(default_factory=dict)

        renderer_dmtet_cls: str = ""
        renderer_dmtet: dict = field(default_factory=dict)

        iterate: bool = False

    cfg: Config

    def configure(self):
        super().configure()
        self.image_tokenizer = lrm.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        if self.cfg.image_tokenizer.modulation:
            self.camera_embedder = lrm.find(self.cfg.camera_embedder_cls)(
                self.cfg.camera_embedder
            )
        self.tokenizer = lrm.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = lrm.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = lrm.find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = lrm.find(self.cfg.decoder_cls)(self.cfg.decoder)
        self.material = lrm.find(self.cfg.material_cls)(self.cfg.material)
        self.background = lrm.find(self.cfg.background_cls)(self.cfg.background)
        self.renderer_nerf = lrm.find(self.cfg.renderer_nerf_cls)(
            self.cfg.renderer_nerf, self.decoder, self.material, self.background
        )
        self.renderer_dmtet = lrm.find(self.cfg.renderer_dmtet_cls)(
            self.cfg.renderer_dmtet, self.decoder, self.material, self.background
        )

        self.exporter = lrm.find(self.cfg.exporter_cls)(
            self.cfg.exporter, self.renderer_dmtet
        )

    def on_fit_start(self):
        super().on_fit_start()
        self.lpips_loss_fn = LPIPS()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # batch["rgb_cond"]: B, N_cond, H, W, 3
        # batch["rgb"]: B, N_render, H, W, 3
        # batch["c2w_cond"]: B, N_cond, 4, 4
        # for single image input (like LRM), N_cond = 1

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        # Camera modulation
        camera_embeds: Optional[Float[Tensor, "B Nv Cc"]]
        if self.cfg.image_tokenizer.modulation:
            camera_embeds = self.camera_embedder(**batch)
        else:
            camera_embeds = None

        input_image_tokens: Float[Tensor, "B Nv Cit Nit"] = self.image_tokenizer(
            rearrange(batch["rgb_cond"], "B Nv H W C -> B Nv C H W"),
            modulation_cond=camera_embeds,
            plucker_rays=rearrange(
                get_plucker_rays(batch["rays_o_cond"], batch["rays_d_cond"]),
                "B Nv H W C -> B Nv C H W",
            )
            if "rays_o_cond" in batch
            else None,
        )
        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=n_input_views
        )

        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))

        return scene_codes

    def forward_renderer_nerf(
        self, batch: Dict[str, Any], scene_codes
    ) -> Dict[str, Any]:
        render_out = self.renderer_nerf(scene_codes, **batch)
        return render_out

    def forward_renderer_dmtet(
        self, batch: Dict[str, Any], scene_codes
    ) -> Dict[str, Any]:
        extra_sdf_query = None
        extra_sdf_label = None
        if self.training and (
            self.C(self.cfg.loss.lambda_surface) > 0.0
            or self.C(self.cfg.loss.lambda_outside_space) > 0.0
            or self.C(self.cfg.loss.lambda_inside_space) > 0.0
        ):
            batch_size = batch["rays_o"].shape[0]
            rays_o, rays_d_unnormed, depth = (
                batch["rays_o"],
                batch["rays_d_unnormed"],
                batch["depth"],
            )
            # for invalid rays, near = 0 and far = 0
            near, far, rays_valid = rays_intersect_bbox(
                batch["rays_o"],
                batch["rays_d_unnormed"],
                radius=self.renderer_nerf.cfg.radius,
            )
            extra_sdf_query = []
            extra_sdf_label = []
            for b in range(batch_size):
                extra_sdf_query_, extra_sdf_label_ = [], []
                rays_valid_ = rays_valid[b]
                rays_o_, rays_d_unnormed_ = (
                    rays_o[b][rays_valid_],
                    rays_d_unnormed[b][rays_valid_],
                )
                near_, far_, depth_ = (
                    near[b][rays_valid_],
                    far[b][rays_valid_],
                    depth[b][rays_valid_],
                )
                depth_valid_ = (depth_ > 1.0e-4)[..., 0]
                depth_ = torch.lerp(far_, depth_, depth_valid_[..., None].float())
                depth_free_ = torch.lerp(near_, depth_, torch.rand_like(depth_))
                rays_o_depth_valid_, rays_d_unnormed_depth_valid_ = (
                    rays_o_[depth_valid_],
                    rays_d_unnormed_[depth_valid_],
                )
                depth_surf_ = depth_[depth_valid_]
                depth_inside_ = torch.lerp(
                    depth_surf_, depth_surf_ + 0.1, torch.rand_like(depth_surf_)
                )
                pts_free_ = rays_o_ + rays_d_unnormed_ * depth_free_
                pts_surf_ = (
                    rays_o_depth_valid_ + rays_d_unnormed_depth_valid_ * depth_surf_
                )
                pts_inside_ = (
                    rays_o_depth_valid_ + rays_d_unnormed_depth_valid_ * depth_inside_
                )

                if self.cfg.loss.sdf_supervision_limit > 0:
                    pts_free_choices_ = torch.randperm(
                        pts_free_.shape[0], device=pts_free_.device
                    )[: self.cfg.loss.sdf_supervision_limit]
                    pts_inout_choices_ = torch.randperm(
                        pts_surf_.shape[0], device=pts_surf_.device
                    )[: self.cfg.loss.sdf_supervision_limit]
                    pts_free_ = pts_free_[pts_free_choices_]
                    pts_surf_ = pts_surf_[pts_inout_choices_]
                    pts_inside_ = pts_inside_[pts_inout_choices_]

                if self.C(self.cfg.loss.lambda_surface) > 0.0:
                    extra_sdf_query_.append(pts_surf_)
                    extra_sdf_label_.append(
                        torch.ones(
                            (pts_surf_.shape[0], 1),
                            dtype=torch.long,
                            device=pts_surf_.device,
                        )
                        * 0
                    )
                if self.C(self.cfg.loss.lambda_outside_space) > 0.0:
                    extra_sdf_query_.append(pts_free_)
                    extra_sdf_label_.append(
                        torch.ones(
                            (pts_free_.shape[0], 1),
                            dtype=torch.long,
                            device=pts_free_.device,
                        )
                        * 1
                    )
                if self.C(self.cfg.loss.lambda_inside_space) > 0.0:
                    extra_sdf_query_.append(pts_inside_)
                    extra_sdf_label_.append(
                        torch.ones(
                            (pts_inside_.shape[0], 1),
                            dtype=torch.long,
                            device=pts_inside_.device,
                        )
                        * -1
                    )
                extra_sdf_query.append(torch.cat(extra_sdf_query_, dim=0))
                extra_sdf_label.append(torch.cat(extra_sdf_label_, dim=0))
        render_out = self.renderer_dmtet(
            scene_codes, extra_sdf_query=extra_sdf_query, **batch
        )

        render_out.update({"extra_sdf_label": extra_sdf_label})

        return render_out

    def compute_loss_nerf(self, batch, out, lambda_suffix="_nerf"):
        loss = 0.0

        for suffix in ["", "_coarse"]:
            if not f"comp_rgb{suffix}" in out:
                continue

            comp_rgb: Float[Tensor, "B Nv H W 3"] = out["comp_rgb{}".format(suffix)]
            gt_rgb: Float[Tensor, "B Nv H W 3"] = batch["rgb"]

            self.log(f"train/comp_rgb_min{suffix}{lambda_suffix}", comp_rgb.min())

            loss_mse = F.mse_loss(comp_rgb, gt_rgb, reduction="mean")
            self.log(f"train/loss_mse{suffix}{lambda_suffix}", loss_mse)
            loss += loss_mse * self.C(
                self.cfg.loss[f"lambda_mse{suffix}{lambda_suffix}"]
            )

            loss_smooth_l1 = F.smooth_l1_loss(
                comp_rgb, gt_rgb, beta=0.1, reduction="mean"
            )
            self.log(f"train/loss_smooth_l1{suffix}{lambda_suffix}", loss_smooth_l1)
            loss += loss_smooth_l1 * self.C(
                self.cfg.loss[f"lambda_smooth_l1{suffix}{lambda_suffix}"]
            )

            if self.C(self.cfg.loss[f"lambda_lpips{suffix}{lambda_suffix}"]) > 0:
                loss_lpips = self.lpips_loss_fn(
                    rearrange(comp_rgb, "B Nv H W C -> (B Nv) C H W"),
                    rearrange(gt_rgb, "B Nv H W C -> (B Nv) C H W"),
                    input_range=(0, 1),
                ).mean()
                self.log(f"train/loss_lpips{suffix}{lambda_suffix}", loss_lpips)
                loss += loss_lpips * self.C(
                    self.cfg.loss[f"lambda_lpips{suffix}{lambda_suffix}"]
                )

            loss_mask = binary_cross_entropy(
                out[f"opacity{suffix}"].clamp(1e-5, 1 - 1e-5), batch["mask"]
            )
            self.log(f"train/loss_mask{suffix}{lambda_suffix}", loss_mask)
            loss += loss_mask * self.C(
                self.cfg.loss[f"lambda_mask{suffix}{lambda_suffix}"]
            )

        return loss

    def compute_loss_dmtet(self, batch, out, lambda_suffix="_dmtet"):
        loss = 0.0

        comp_rgb: Float[Tensor, "B Nv H W 3"] = out["comp_rgb"]
        gt_rgb: Float[Tensor, "B Nv H W 3"] = batch["rgb"]

        self.log(f"train/comp_rgb_min{lambda_suffix}", comp_rgb.min())

        loss_mse = F.mse_loss(comp_rgb, gt_rgb, reduction="mean")
        self.log(f"train/loss_mse{lambda_suffix}", loss_mse)
        loss += loss_mse * self.C(self.cfg.loss[f"lambda_mse{lambda_suffix}"])

        loss_smooth_l1 = F.smooth_l1_loss(comp_rgb, gt_rgb, beta=0.1, reduction="mean")
        self.log(f"train/loss_smooth_l1{lambda_suffix}", loss_smooth_l1)
        loss += loss_smooth_l1 * self.C(
            self.cfg.loss[f"lambda_smooth_l1{lambda_suffix}"]
        )

        if self.C(self.cfg.loss[f"lambda_lpips{lambda_suffix}"]) > 0:
            loss_lpips = self.lpips_loss_fn(
                rearrange(comp_rgb, "B Nv H W C -> (B Nv) C H W"),
                rearrange(gt_rgb, "B Nv H W C -> (B Nv) C H W"),
                input_range=(0, 1),
            ).mean()
            self.log(f"train/loss_lpips{lambda_suffix}", loss_lpips)
            loss += loss_lpips * self.C(self.cfg.loss[f"lambda_lpips{lambda_suffix}"])

        loss_mask = binary_cross_entropy(
            out["opacity"].clamp(1e-5, 1 - 1e-5), batch["mask"]
        )
        self.log(f"train/loss_mask{lambda_suffix}", loss_mask)
        loss += loss_mask * self.C(self.cfg.loss[f"lambda_mask{lambda_suffix}"])

        # NOTE: do not apply this to TriplaneNeRFRenderer
        valid_pixels = batch["mask"] == 1.0

        loss_depth_l1 = F.l1_loss(
            out["depth"][valid_pixels], batch["depth"][valid_pixels]
        )
        self.log(f"train/loss_depth_l1{lambda_suffix}", loss_depth_l1)
        loss += loss_depth_l1 * self.C(self.cfg.loss[f"lambda_depth_l1{lambda_suffix}"])

        loss_normal_cos = (
            1.0
            - (
                (out["comp_normal"][valid_pixels.squeeze(-1)] * 2.0 - 1.0)
                * (batch["normal"][valid_pixels.squeeze(-1)] * 2.0 - 1.0)
            ).sum(-1)
        ).mean()
        self.log(f"train/loss_normal_cos{lambda_suffix}", loss_normal_cos)
        loss += loss_normal_cos * self.C(
            self.cfg.loss[f"lambda_normal_cos{lambda_suffix}"]
        )

        if "sdf_reg" in out:
            # force empty space to have something
            loss_sdf_reg = out["sdf_reg"].mean()
            self.log(f"train/loss_sdf_reg", loss_sdf_reg)
            loss += loss_sdf_reg + self.C(self.cfg.loss.lambda_sdf_reg)

        if (
            self.C(self.cfg.loss.lambda_surface) > 0.0
            or self.C(self.cfg.loss.lambda_outside_space) > 0.0
            or self.C(self.cfg.loss.lambda_inside_space) > 0.0
        ):
            loss_sdf_surf, loss_sdf_free, loss_sdf_inside = 0.0, 0.0, 0.0
            batch_size = len(out["sdf_ex_query"])
            for sdf_ex_query_, extra_sdf_label_ in zip(
                out["sdf_ex_query"], out["extra_sdf_label"]
            ):
                loss_sdf_surf += (
                    torch.abs(sdf_ex_query_) * (extra_sdf_label_ == 0).float()
                ).mean() / batch_size
                loss_sdf_free += (
                    F.relu(-sdf_ex_query_) * (extra_sdf_label_ > 0).float()
                ).mean() / batch_size
                loss_sdf_inside += (
                    F.relu(sdf_ex_query_) * (extra_sdf_label_ < 0).float()
                ).mean() / batch_size

            loss = (
                loss
                + loss_sdf_free * self.C(self.cfg.loss.lambda_outside_space)
                + loss_sdf_surf * self.C(self.cfg.loss.lambda_surface)
                + loss_sdf_inside * self.C(self.cfg.loss.lambda_inside_space)
            )
            self.log("train/loss_sdf_free", loss_sdf_free)
            self.log("train/loss_sdf_surf", loss_sdf_surf)
            self.log("train/loss_sdf_inside", loss_sdf_inside)

        return loss

    def training_step(self, batch, batch_idx):
        batch_full = {
            k.replace("_full", ""): v for k, v in batch.items() if "_full" in k
        }

        scene_codes = self(batch)

        loss = 0.0

        if self.cfg.iterate:
            if self.true_global_step % 2 == 0:
                out = self.forward_renderer_nerf(batch, scene_codes)
                loss += self.compute_loss_nerf(batch, out)
            else:
                out = self.forward_renderer_dmtet(batch_full, scene_codes)
                loss += self.compute_loss_dmtet(batch_full, out)
        else:
            out_nerf = self.forward_renderer_nerf(batch, scene_codes)
            loss += self.compute_loss_nerf(batch, out_nerf)
            out_dmtet = self.forward_renderer_dmtet(batch_full, scene_codes)
            loss += self.compute_loss_dmtet(batch_full, out_dmtet)
            out = {
                **{f"{k}_nerf": v for k, v in out_nerf.items()},
                **{f"{k}_dmtet": v for k, v in out_dmtet.items()},
            }

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        # will execute self.on_check_train every self.cfg.check_train_every_n_steps steps
        self.check_train(batch, out)

        return {"loss": loss}

    def get_input_visualizations(self, batch):
        return [
            {
                "type": "rgb",
                "img": rearrange(batch["rgb_cond"], "B N H W C -> (B H) (N W) C"),
                "kwargs": {"data_format": "HWC"},
            }
        ]

    def get_output_visualizations(self, batch, outputs):
        out = outputs
        images = []
        if "rgb" in batch:
            images += [
                {
                    "type": "rgb",
                    "img": rearrange(batch["rgb"], "B N H W C -> (B H) (N W) C"),
                    "kwargs": {"data_format": "HWC"},
                }
            ]

        for suffix in ["", "_coarse"]:
            if not f"comp_rgb{suffix}_nerf" in out:
                continue
            images += [
                {
                    "type": "rgb",
                    "img": rearrange(
                        out[f"comp_rgb{suffix}_nerf"], "B N H W C -> (B H) (N W) C"
                    ),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "grayscale",
                    "img": rearrange(
                        out[f"depth{suffix}_nerf"], "B N H W C -> (B H) (N W) C"
                    )[..., 0],
                    "kwargs": {"cmap": None, "data_range": None},
                },
            ]

        if "rgb_full" in batch:
            images += [
                {
                    "type": "rgb",
                    "img": rearrange(batch["rgb_full"], "B N H W C -> (B H) (N W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": rearrange(
                        batch["normal_full"], "B N H W C -> (B H) (N W) C"
                    ),
                    "kwargs": {"data_format": "HWC"},
                },
            ]

        if f"comp_rgb_dmtet" in out:
            images += [
                {
                    "type": "rgb",
                    "img": rearrange(
                        out["comp_rgb_dmtet"], "B N H W C -> (B H) (N W) C"
                    ),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": rearrange(
                        out["comp_normal_dmtet"], "B N H W C -> (B H) (N W) C"
                    ),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
        return images

    def on_check_train(self, batch, outputs):
        self.save_image_grid(
            f"it{self.true_global_step}-train.jpg",
            self.get_output_visualizations(batch, outputs),
            name="train_step_output",
            step=self.true_global_step,
        )

    def validation_step(self, batch, batch_idx):
        batch_full = {
            k.replace("_full", ""): v for k, v in batch.items() if "_full" in k
        }
        scene_codes = self(batch)
        out_nerf = self.forward_renderer_nerf(batch, scene_codes)
        out_dmtet = self.forward_renderer_dmtet(batch_full, scene_codes)
        out = {
            **{f"{k}_nerf": v for k, v in out_nerf.items()},
            **{f"{k}_dmtet": v for k, v in out_dmtet.items()},
        }
        if (
            self.cfg.check_val_limit_rank > 0
            and self.global_rank < self.cfg.check_val_limit_rank
        ):
            self.save_image_grid(
                f"it{self.true_global_step}-validation-{self.global_rank}_{batch_idx}-input.jpg",
                self.get_input_visualizations(batch),
                name=f"validation_step_input_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )
            self.save_image_grid(
                f"it{self.true_global_step}-validation-{self.global_rank}_{batch_idx}.jpg",
                self.get_output_visualizations(batch, out),
                name=f"validation_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )

    def test_step(self, batch, batch_idx):
        # not saved to wandb
        scene_codes = self(batch)
        # out_nerf = self.forward_renderer_nerf(batch, scene_codes)
        out_dmtet = self.forward_renderer_dmtet(batch, scene_codes)
        out = {
            # **{f"{k}_nerf": v for k, v in out_nerf.items()},
            **{f"{k}_dmtet": v for k, v in out_dmtet.items()}
        }
        batch_size = batch["index"].shape[0]
        for b in range(batch_size):
            if batch["view_index"][b, 0] == 0:
                self.save_mesh(
                    f"it{self.true_global_step}-test-mesh/{batch['index'][b]}.obj",
                    out["mesh_dmtet"][b].v_pos,
                    out["mesh_dmtet"][b].t_pos_idx,
                )
                self.save_image_grid(
                    f"it{self.true_global_step}-test/{batch['index'][b]}-input.jpg",
                    [
                        {
                            "type": "rgb",
                            "img": rearrange(
                                batch["rgb_cond"][b], "N H W C -> H (N W) C"
                            ),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ],
                )
            images = []
            if "comp_rgb_nerf" in out:
                images += [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb_nerf"][b][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "grayscale",
                        "img": out["depth_nerf"][b][0, ..., 0],
                        "kwargs": {"cmap": None, "data_range": None},
                    },
                ]
            if "comp_rgb_dmtet" in out:
                images += [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb_dmtet"][b][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_normal_dmtet"][b][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][b]}/{batch['view_index'][b,0]}.png",
                images,
            )

    def on_test_end(self):
        if self.global_rank == 0:
            self.save_img_sequences(
                f"it{self.true_global_step}-test",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
            )
