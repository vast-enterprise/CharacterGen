from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch3d.loss import chamfer_distance

import lrm
from lrm.systems.base import BaseSystem
from lrm.utils.base import BaseModule
from lrm.utils.typing import *

class Naive3DGS(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super().configure()
        # self.image_tokenizer = lrm.find(self.cfg.image_tokenizer_cls)(
        #     self.cfg.image_tokenizer
        # )
        # self.tokenizer = lrm.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        # self.backbone = lrm.find(self.cfg.backbone_cls)(self.cfg.backbone)
        # self.post_processor = lrm.find(self.cfg.post_processor_cls)(
        #     self.cfg.post_processor
        # )
        self.renderer = lrm.find(self.cfg.renderer_cls)(self.cfg.renderer)
        self.learnable_embedding = nn.Parameter(torch.randn(1, 8192, 128)).requires_grad_(True)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # # batch['rgb_cond']: B, N_cond, H, W, 3
        # # batch['rgb']: B, N_render, H, W, 3
        # # batch['c2w_cond']: B, N_cond, 4, 4
        # # for single image input (like LRM), N_cond = 1
        # batch_size = batch["rgb_cond"].shape[0]
        # camera_modulation = batch["c2w_cond"].view(*batch["c2w_cond"].shape[:-2], -1)
        # input_image_tokens: Float[Tensor, "B Cit Nit"] = self.image_tokenizer(
        #     batch["rgb_cond"].permute(0, 1, 4, 2, 3),
        #     modulation_cond=camera_modulation,
        # )
        # tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size)
        # tokens = self.backbone(
        #     tokens,
        #     encoder_hidden_states=input_image_tokens.permute(0, 2, 1),
        #     modulation_cond=camera_modulation.view(batch_size, -1),
        # )
        # scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        batch_size = batch["rgb_cond"].shape[0]
        scene_codes = self.learnable_embedding.expand(batch_size, -1, -1)
        render_out = self.renderer(scene_codes, **batch)
        return render_out

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.0

        loss_mse = F.mse_loss(out["comp_rgb"], batch["rgb"])
        self.log("train/loss_mse", loss_mse)
        loss += loss_mse * self.C(self.cfg.loss.lambda_mse)

        if self.C(self.cfg.loss.lambda_chamfer) > 0 and "points" in batch:
            loss_cd, _ = chamfer_distance(out["points"][..., :3], batch["points"][..., :3])
            loss += self.C(self.cfg.loss.lambda_chamfer) * loss_cd
            self.log("train/loss_chamfer", loss_cd.item())

        # TODO: perceptual loss
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0,0]}-input.png",
            [
                {
                    "type": "rgb",
                    "img": rearrange(batch["rgb_cond"][0], "N H W C -> H (N W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
            ],
            name="validation_step_input",
            step=self.true_global_step,
        )
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0,0]}.png",
            [
                {
                    "type": "rgb",
                    "img": rearrange(out["comp_rgb"][0], "N H W C -> H (N W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": rearrange(out["comp_normal"][0], "N H W C -> H (N W) C"),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": rearrange(out["opacity"][0], "N H W C -> H (N W) C")[..., 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ] if "opacity" in out else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        raise NotImplementedError
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
