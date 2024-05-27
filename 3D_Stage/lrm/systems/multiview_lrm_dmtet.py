import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange

import lrm
from lrm.systems.base import BaseLossConfig, BaseSystem
from lrm.utils.misc import time_recorder as tr
from lrm.utils.ops import binary_cross_entropy, rays_intersect_bbox, get_plucker_rays
from lrm.utils.typing import *
from lrm.models.lpips import LPIPS


@dataclass
class MultiviewLRMDMTetLossConfig(BaseLossConfig):
    lambda_mse: Any = 0.0
    lambda_smooth_l1: Any = 0.0
    lambda_lpips: Any = 0.0
    lambda_mask: Any = 0.0
    lambda_depth_l1: Any = 0.0
    lambda_normal_cos: Any = 0.0
    lambda_sdf_reg: Any = 0.0
    lambda_surface: Any = 0.0
    lambda_outside_space: Any = 0.0
    lambda_inside_space: Any = 0.0
    sdf_supervision_limit: int = 0


class MultiviewLRMDMTet(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        loss: MultiviewLRMDMTetLossConfig = MultiviewLRMDMTetLossConfig()

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

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

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
        self.renderer = lrm.find(self.cfg.renderer_cls)(
            self.cfg.renderer, self.decoder, self.material, self.background
        )

        self.exporter = lrm.find(self.cfg.exporter_cls)(
            self.cfg.exporter, self.renderer
        )

    def on_fit_start(self):
        super().on_fit_start()
        self.lpips_loss_fn = LPIPS()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # batch["rgb_cond"]: B, N_cond, H, W, 3
        # batch["rgb"]: B, N_render, H, W, 3
        # batch["c2w_cond"]: B, N_cond, 4, 4
        # for single image input (like LRM), N_cond = 1

        batch_size = batch["rgb_cond"].shape[0]
        # Camera modulation
        camera_embeds: Optional[Float[Tensor, "B Nv Cc"]]
        if self.cfg.image_tokenizer.modulation:
            camera_embeds = self.camera_embedder(**batch)
        else:
            camera_embeds = None

        tr.start("Image feature")
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
        tr.end("Image feature")
        input_image_tokens = rearrange(input_image_tokens, "B Nv C Nt -> B (Nv Nt) C")

        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size)

        tr.start("Transformer backbone")
        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )
        tr.end("Transformr backbone")

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))

        return scene_codes

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
                radius=self.renderer.cfg.radius,
            )
            extra_sdf_query = []
            extra_sdf_label = []
            tr.start("prepare extra sdf")
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
            tr.end("prepare extra sdf")
        tr.start("render")
        render_out = self.renderer(
            scene_codes, extra_sdf_query=extra_sdf_query, **batch
        )
        tr.end("render")

        render_out.update({"extra_sdf_label": extra_sdf_label})

        return render_out

    def training_step(self, batch, batch_idx):
        scene_codes = self(batch)
        out = self.forward_renderer_dmtet(batch, scene_codes)

        loss = 0.0

        comp_rgb: Float[Tensor, "B Nv H W 3"] = out["comp_rgb"]
        gt_rgb: Float[Tensor, "B Nv H W 3"] = batch["rgb"]

        self.log("train/comp_rgb_min", comp_rgb.min())

        loss_mse = F.mse_loss(comp_rgb, gt_rgb, reduction="mean")
        self.log(f"train/loss_mse", loss_mse)
        loss += loss_mse * self.C(self.cfg.loss.lambda_mse)

        loss_smooth_l1 = F.smooth_l1_loss(comp_rgb, gt_rgb, beta=0.1, reduction="mean")
        self.log(f"train/loss_smooth_l1", loss_smooth_l1)
        loss += loss_smooth_l1 * self.C(self.cfg.loss.lambda_smooth_l1)

        if self.C(self.cfg.loss.lambda_lpips) > 0:
            loss_lpips = self.lpips_loss_fn(
                rearrange(comp_rgb, "B Nv H W C -> (B Nv) C H W"),
                rearrange(gt_rgb, "B Nv H W C -> (B Nv) C H W"),
                input_range=(0, 1),
            ).mean()
            self.log("train/loss_lpips", loss_lpips)
            loss += loss_lpips * self.C(self.cfg.loss.lambda_lpips)

        loss_mask = binary_cross_entropy(
            out["opacity"].clamp(1e-5, 1 - 1e-5), batch["mask"]
        )
        self.log("train/loss_mask", loss_mask)
        loss += loss_mask * self.C(self.cfg.loss.lambda_mask)

        # NOTE: do not apply this to TriplaneNeRFRenderer
        valid_pixels = batch["mask"] == 1.0

        loss_depth_l1 = F.l1_loss(
            out["depth"][valid_pixels], batch["depth"][valid_pixels]
        )
        self.log("train/loss_depth_l1", loss_depth_l1)
        loss += loss_depth_l1 * self.C(self.cfg.loss.lambda_depth_l1)

        loss_normal_cos = (
            1.0
            - (
                (out["comp_normal"][valid_pixels.squeeze(-1)] * 2.0 - 1.0)
                * (batch["normal"][valid_pixels.squeeze(-1)] * 2.0 - 1.0)
            ).sum(-1)
        ).mean()
        self.log("train/loss_normal_cos", loss_normal_cos)
        loss += loss_normal_cos * self.C(self.cfg.loss["lambda_normal_cos"])

        if "sdf_reg" in out:
            # force empty space to have something
            loss_sdf_reg = out["sdf_reg"].mean()
            self.log("train/loss_sdf_reg", loss_sdf_reg)
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

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        # will execute self.on_check_train every self.cfg.check_train_every_n_steps steps
        self.check_train(batch, out)

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs["loss"].isnan():
            print(self.global_rank, batch["scene_id"])
            print("NaN! Training failed!")
            exit(1)

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
                },
                {
                    "type": "rgb",
                    "img": rearrange(batch["normal"], "B N H W C -> (B H) (N W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
            ]

        images += [
            {
                "type": "rgb",
                "img": rearrange(out["comp_rgb"], "B N H W C -> (B H) (N W) C"),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(out["comp_normal"], "B N H W C -> (B H) (N W) C"),
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
        scene_codes = self(batch)
        out = self.forward_renderer_dmtet(batch, scene_codes)
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

    # def test_step(self, batch, batch_idx):
    #     # not saved to wandb
    #     scene_codes = self(batch)
    #     out = self.forward_renderer_dmtet(batch, scene_codes)
    #     batch_size = batch["index"].shape[0]
    #     for b in range(batch_size):
    #         if batch["view_index"][b, 0] == 0:
    #             exporter_output = self.exporter(
    #                 [
    #                     # f"it{self.true_global_step}-test-mesh/{batch['name'][b]}-{batch['case_index'][b]}"
    #                     f"it{self.true_global_step}-test-mesh/{batch['index'][b]}",
    #                 ],
    #                 scene_codes[b][None],
    #             )

    #             # meshes saved to config.exp_root_dir/config.tag@timestamp
    #             for eout in exporter_output:
    #                 save_func_name = f"save_{eout.save_type}"
    #                 save_func = getattr(self, save_func_name)
    #                 save_func(f"{eout.save_name}", **eout.params)

    #             # self.save_mesh(
    #             #     f"it{self.true_global_step}-test-mesh/{batch['index'][b]}.obj",
    #             #     out["mesh"][b].v_pos,
    #             #     out["mesh"][b].t_pos_idx,
    #             # )

    #             self.save_image_grid(
    #                 f"it{self.true_global_step}-test/{batch['index'][b]}-input.jpg",
    #                 [
    #                     {
    #                         "type": "rgb",
    #                         "img": rearrange(
    #                             batch["rgb_cond"][b], "N H W C -> H (N W) C"
    #                         ),
    #                         "kwargs": {"data_format": "HWC"},
    #                     },
    #                 ],
    #             )
    #         self.save_image_grid(
    #             f"it{self.true_global_step}-test/{batch['index'][b]}/{batch['view_index'][b,0]}.png",
    #             [
    #                 {
    #                     "type": "rgb",
    #                     "img": out["comp_rgb"][b][0],
    #                     "kwargs": {"data_format": "HWC"},
    #                 },
    #                 {
    #                     "type": "rgb",
    #                     "img": out["comp_normal"][b][0],
    #                     "kwargs": {"data_format": "HWC"},
    #                 },
    #             ],
    #         )

    def test_step(self, batch, batch_idx):
        # not saved to wandb
        scene_codes = self(batch)
        out = self.forward_renderer_dmtet(batch, scene_codes)
        batch_size = batch["index"].shape[0]
        for b in range(batch_size):
            if batch["view_index"][b, 0] == 0:
                exporter_output = self.exporter(
                    [
                        # f"it{self.true_global_step}-test-mesh/{batch['name'][b]}"
                        f"it{self.true_global_step}-test-mesh/{batch['index'][b]}",
                    ],
                    scene_codes[b][None],
                )

                # meshes saved to config.exp_root_dir/config.tag@timestamp
                for eout in exporter_output:
                    save_func_name = f"save_{eout.save_type}"
                    save_func = getattr(self, save_func_name)
                    save_func(f"{eout.save_name}", **eout.params)

                # self.save_mesh(
                #     f"it{self.true_global_step}-test-mesh/{batch['index'][b]}.obj",
                #     out["mesh"][b].v_pos,
                #     out["mesh"][b].t_pos_idx,
                # )

                self.save_image_grid(
                    # f"it{self.true_global_step}-test/{batch['name'][b]}-input.jpg",
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
            self.save_image_grid(
                # f"it{self.true_global_step}-test/{batch['name'][b]}/{batch['view_index'][b,0]}.png",
                f"it{self.true_global_step}-test/{batch['index'][b]}/{batch['view_index'][b,0]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][b][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][b][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
            )

    def on_test_end(self):
        if self.global_rank == 0:
            self.save_img_sequences(
                f"it{self.true_global_step}-test",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
            )
