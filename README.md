# CharacterGen: Efficient 3D Character Generation from Single Images with Multi-View Pose Calibration

This is the official codebase of SIGGRAPH'24 (TOG) [CharacterGen](https://charactergen.github.io/).

![teaser](./materials/teaser.png)

- [x] Rendering Script of VRM model, including blender and three-js.
- [x] Inference code for 2D generation stage.
- [x] Inference code for 3D generation stage.

## Quick Start

### 1. Prepare environment

`pip install -r requirements.txt`


### 2. Download the weight

To be uploaded soon.

### 3. Run the script

#### Run 2D Stage

```
cd 2D_Stage
python webui.py
```

#### Run 3D Stage

```
cd 3D_Stage
python webui.py
```

## Get the Anime3D Dataset

Due to the policy, we cannot redistribute the raw data of VRM format 3D character.
You can download the vroid dataset follow [PAniC-3D](https://github.com/ShuhongChen/panic3d-anime-reconstruction) instruction.
And the you can render the script with blender or three-js with our released rendering script.

### Blender

First, you should install  [Blender](https://www.blender.org/) and  [the VRM addon for Blender](https://github.com/saturday06/VRM-Addon-for-Blender).

The you can render the VRM and export the obj of VRM under some fbx animation.

```
blender -b --python render_script/blender/render.py importVrmPath importFbxPath outputFolder [is_apose]
```

The last input argument represents whether you use apose; if used, output apose; otherwise, output the action of any frame in the fbx.

### [three-vrm](https://github.com/pixiv/three-vrm)

**Much quicker than blender VRM add-on.**

Install [Node.js](https://nodejs.org/) first to use the npm environment.

```
cd render_script/three-js
npm install three @pixiv/three-vrm
```

If you want to render depth-map images of VRM, you should replace three-vrm with [my version](/home/zjp/CharacterGen/render_script/three-js/src/three-vrm.js).

Fisrt, run the backend to catch the data from the frontend (default port is `17070`), remember to change the folder path.

```
pip install fastapi uvicorn aiofiles pillow numpy
python up_backend.py
```

Second, run the frontend to render the images.

```
npm run dev
```

The open the website http://localhost:5173/, it use 2 threads to render the image, which costs about 1 day.

## Our Result

| Single Input Image | 2D Multi-View Images | 3D Character |
|-------|-------|-------|
| ![](./materials/input/1.png) | ![](./materials/ours_multiview/1.png) | <img alt="threestudio" src="./materials/videos/1.gif" width="100%"> |
| ![](./materials/input/2.png) | ![](./materials/ours_multiview/2.png) | <img alt="threestudio" src="./materials/videos/2.gif" width="100%"> |
| ![](./materials/input/3.png) | ![](./materials/ours_multiview/3.png) | <img alt="threestudio" src="./materials/videos/3.gif" width="100%"> |

# Acknowledgements

This project is built upon [Tune-A-Video](https://github.com/showlab/Tune-A-Video) and [TripoSR](https://github.com/VAST-AI-Research/TripoSR).
And the rendering scripts is build upon [three-vrm](https://github.com/pixiv/three-vrm) and [VRM-Addon-for-Blender](https://github.com/saturday06/VRM-Addon-for-Blender).
Thanks very much to many friends for their unselfish help with our work. We're extremely grateful to [Yuanchen](https://github.com/bennyxguo), [Yangguang](https://scholar.google.com/citations?user=a7AMvgkAAAAJ), and Yuan Liang for their guidance on code details and ideas.
We thank all the authors for their great repos and help.

# Citation

If you find our code or paper helps, 

```bibtex
@article{peng2024charactergen,
  title     ={Subdivision-based mesh convolution networks},
  author    ={Peng, Hao-Yang and Zhang, Jia-Peng and Guo, Meng-Hao and Cao, Yan-Pei, and Hu, Shi-Min},
  journal   ={ACM Transactions on Graphics (TOG)},
  year      ={2024},
  volume    ={43},
  number    ={4},
  doi       ={10.1145/3658217}
}
```