import json
import os
import numpy as np
from PIL import Image
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from skimage.metrics import structural_similarity as cssim
from skimage.metrics import peak_signal_noise_ratio as cpsnr
import lpips

use_apose = True
root_image_dir = "/mnt/pfs/users/zhangjiapeng/workspace_for_IP_Adapter/valid_768"
gen_root_dir = "/mnt/pfs/users/zhangjiapeng/workspace_for_IP_Adapter/zero123/zero123/bk"
# gen_root_dir = "/mnt/pfs/users/zhangjiapeng/workspace_for_IP_Adapter/tune-a-video/output"
valid_paths = json.load(open("valid_paths.json", "r"))

def cal_ssim(img1, img2):
    img1, img2 = np.array(img1.data.cpu()), np.array(img2.data.cpu())
    # print(img1.shape, img2.shape)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    # print(img1.shape, img2.shape)
    ssim_score = cssim(img1, img2, data_range=1, multichannel=True)
    return ssim_score

def cal_psnr(img1, img2):
    img1, img2 = np.array(img1.data.cpu()), np.array(img2.data.cpu())
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    psnr_score = psnr(img1, img2, data_range=1)
    return psnr_score

def cal_lpips(img1, img2, loss_fn):
    lpips_loss = loss_fn.forward(img1, img2)
    return lpips_loss

def process_image(img):
    nimg = (255 - img[...,3:]) + img[...,:3]
    return nimg

loss_fn = lpips.LPIPS(net="vgg").to("cuda")

gt_images = []
gen_images = []
for vp in valid_paths:
    if use_apose and not "a_pose" in vp: continue
    if not use_apose and "a_pose" in vp: continue
    fid = vp.replace("_a_pose", "")
    if use_apose:
        suffix = ["000", "002", "004", "006"]
    else:
        suffix = ["007", "005", "003", "001"]
    print(suffix)
    for idx, sf in enumerate(suffix):
        gt_image = torch.from_numpy(process_image(np.asarray(Image.open(os.path.join(root_image_dir, vp, fid + "_" + sf + "_rgb.png")).resize((256, 256))))).cuda()
        gt_images.append(gt_image / 255.)
        gen_images.append((torch.from_numpy(np.asarray(Image.open(os.path.join(gen_root_dir, fid  + "_007_rgb", f"zero123_{idx}.png")).convert("RGB"))).cuda() / 255.))
        # gen_images.append((torch.from_numpy(np.asarray(Image.open(os.path.join(gen_root_dir, fid  + "_007_rgb_seg", fid  + f"_007_rgb_seg-{idx}.png")).convert("RGB"))).cuda() / 255.))
    # get ssim and lpips
ssim = 0
lpips = 0
with torch.no_grad():
    for idx in range(len(gt_images)):
        # print(gt_images[idx].shape, gen_images[idx].shape)
        ssim += cal_ssim(gt_images[idx], gen_images[idx])
        lpips += cal_lpips(gt_images[idx].permute(2,0,1), gen_images[idx].permute(2,0,1), loss_fn)
    print(ssim / len(gt_images))
    print(lpips / len(gt_images))
    # get fid 
    fid = FrechetInceptionDistance(feature=64).set_dtype(torch.float32).cuda()
    gt_all = (torch.stack(gt_images).permute(0,3,1,2) * 255).to(torch.uint8)
    gen_all = (torch.stack(gen_images).permute(0,3,1,2) * 255).to(torch.uint8)
    fid.update(gt_all, real=True)
    fid.update(gen_all, real=False)
    print(fid.compute())