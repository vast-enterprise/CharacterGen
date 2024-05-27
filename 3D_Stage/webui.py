import os
import json
import cv2
import numpy as np
import torch, lrm
from lrm.utils.config import load_config
from datetime import datetime
import gradio as gr

device = "cuda"

class Inference_API:

    def __init__(self):
        # Load config
        self.cfg = load_config("configs/infer.yaml", makedirs=False)
        # Load system
        self.system = lrm.find(self.cfg.system_cls)(self.cfg.system).to(device)
        self.system.eval()

    def process_images(self, img_input0, img_input1, img_input2, img_input3):
        meta = json.load(open("material/meta.json"))
        c2w_cond = [np.array(loc["transform_matrix"]) for loc in meta["locations"]]
        c2w_cond = torch.from_numpy(np.stack(c2w_cond, axis=0)).float()[None].to(device)
        
        # Prepare input data
        rgb_cond = []
        files = [img_input0, img_input1, img_input2, img_input3]
        for file in files:
            image = np.array(file)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            rgb = cv2.resize(image, (self.cfg.data.cond_width, 
                                     self.cfg.data.cond_height)).astype(np.float32) / 255.0
            rgb_cond.append(rgb)
        assert len(rgb_cond) == 4, "Please provide 4 images"

        rgb_cond = torch.from_numpy(np.stack(rgb_cond, axis=0)).float()[None].to(device)

        # Run inference
        with torch.no_grad():
            scene_codes = self.system({"rgb_cond": rgb_cond, "c2w_cond": c2w_cond})
            exporter_output = self.system.exporter([f"{i:02d}" for i in range(rgb_cond.shape[0])], scene_codes)

        # Save output
        save_dir = os.path.join("./outputs", datetime.now().strftime("@%Y%m%d-%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
        self.system.set_save_dir(save_dir)
        output_files = []

        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            save_func = getattr(self.system, save_func_name)
            save_func(f"{out.save_name}", **out.params)
            output_files.append(f"{save_dir}/{out.save_name}")

        return save_dir, output_files[0]

inferapi = Inference_API()

# Define the interface
with gr.Blocks() as demo:
    gr.Markdown("# [SIGGRAPH'24] CharacterGen: Efficient 3D Character Generation from Single Images with Multi-View Pose Calibration")
    gr.Markdown("# 3D Stage: Four View Images to 3D Mesh")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                img_input0 = gr.Image(type="pil", label="Back Image", image_mode="RGBA", width=256, height=384)
                img_input1 = gr.Image(type="pil", label="Front Image", image_mode="RGBA", width=256, height=384)
            with gr.Row():
                img_input2 = gr.Image(type="pil", label="Right Image", image_mode="RGBA", width=256, height=384)
                img_input3 = gr.Image(type="pil", label="Left Image", image_mode="RGBA", width=256, height=384)
            with gr.Row():
                gr.Examples(
                    examples=
                    [["material/examples/1/1.png",
                    "material/examples/1/2.png",
                    "material/examples/1/3.png",
                    "material/examples/1/4.png"]],
                    label="Example Images",
                    inputs=[img_input0, img_input1, img_input2, img_input3]
                )
            submit_button = gr.Button("Process")
        with gr.Column():
            output_dir = gr.Textbox(label="Output Directory")
            output_mesh = gr.Model3D(label="Output Mesh", height=512)

    submit_button.click(inferapi.process_images, inputs=[img_input0, img_input1, img_input2, img_input3],
                        outputs=[output_dir, output_mesh])

# Run the interface
demo.launch()