from inspect import signature

import torch
from torch import nn
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False



__all__ = ["SparseDrive"]

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Optional
from mmcv.parallel import DataContainer as DC

import matplotlib.pyplot as plt
import numpy as np

# def lidar_to_pseudo_image_no_resize(
#     lidar, batch_lidar2img, out_h=256, out_w=704, max_depth=100.0):
#     """
#     lidar: list[8] of (Ni,5) numpy arrays
#     batch_lidar2img: list[6] of tensor(8,4,4) - 6 cameras, each has batch_size=8 transformation matrices
#     return: pseudo (8,6,3,H,W)
#     """
#     import torch
#     import numpy as np
#     # Helper: to float32 tensor
#     def to_tensor(x, device=None):
#         if isinstance(x, torch.Tensor):
#             tensor = x.float()
#         elif isinstance(x, np.ndarray):
#             tensor = torch.from_numpy(x).float()
#         else:
#             tensor = torch.tensor(x, dtype=torch.float32)
#         # 如果指定了设备，则移动到该设备
#         if device is not None:
#             tensor = tensor.to(device)
#         return tensor
#     # 首先确定设备：优先使用CUDA，如果有的话
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#     # 1. lidar: pad list → (B,N,5)
#     if isinstance(lidar, list):
#         # 先将所有lidar数据转换为tensor并放到device上
#         lidar_tensors = []
#         for p in lidar:
#             tensor = to_tensor(p, device)
#             lidar_tensors.append(tensor)
#         B = len(lidar_tensors)  # should be 8
#         maxN = max(p.shape[0] for p in lidar_tensors)
#         # 创建填充后的tensor，直接放在device上
#         lidar_tensor = torch.zeros((B, maxN, 5), dtype=torch.float32, device=device)
#         for i, p in enumerate(lidar_tensors):
#             lidar_tensor[i, :p.shape[0]] = p
#         lidar = lidar_tensor
#     else:
#         lidar = to_tensor(lidar, device)
#         B = lidar.shape[0]
#     print(f"lidar device: {lidar.device}")
#     # 2. batch_lidar2img: list[6] of (8,4,4) → (8,6,4,4)
#     first_cam = batch_lidar2img[0]
#     if isinstance(first_cam, torch.Tensor):
#         cam_device = first_cam.device
#     else:
#         cam_device = device
#     print(f"Camera parameters will be moved to: {cam_device}")
#     # 处理每个相机的变换矩阵
#     lidar2img_tensors = []
#     for i in range(6):
#         cam_matrix = batch_lidar2img[i]  # should be (8,4,4)
#         tensor = to_tensor(cam_matrix, cam_device)
#         lidar2img_tensors.append(tensor)
#     # Stack along camera dimension: from list[6] of (8,4,4) to (6,8,4,4)
#     lidar2img = torch.stack(lidar2img_tensors, dim=0)  # (6,8,4,4)
#     # Transpose to get (8,6,4,4)
#     lidar2img = lidar2img.transpose(0, 1)  # (8,6,4,4)
#     # 确保lidar2img与lidar在相同的设备上
#     if lidar2img.device != lidar.device:
#         print(f"Moving lidar2img from {lidar2img.device} to {lidar.device}")
#         lidar2img = lidar2img.to(lidar.device)
#     print(f"lidar shape: {lidar.shape}")  # should be (8, N, 5)
#     print(f"lidar2img shape: {lidar2img.shape}")  # should be (8,6,4,4)
#     print(f"lidar2img device: {lidar2img.device}")
#     # 3. Output buffers - 直接在目标设备上创建
#     _, N, _ = lidar.shape
#     pseudo = torch.zeros((B, 6, 3, out_h, out_w), device=lidar.device)
#     depth_buffer = torch.full((B, 6, out_h, out_w), float('inf'), device=lidar.device)
#     # 4. lidar preprocess
#     xyz = lidar[..., :3]       # (B,N,3)
#     intensity = lidar[..., 3]  # (B,N)
#     ones = torch.ones((B, N, 1), device=lidar.device)
#     pts_h = torch.cat([xyz, ones], dim=-1)  # (B,N,4)
#     # 5. Projection
#     for b in range(B):
#         pts = pts_h[b]        # (N,4)
#         inten = intensity[b]  # (N,)
#         for cam in range(6):
#             P = lidar2img[b, cam, :3, :]  # (3,4)
#             # Project all points at once
#             proj = (P @ pts.T)  # (3,N)
#             x, y, z = proj[0], proj[1], proj[2]
#             # Filter valid points (z > 0)
#             valid = (z > min_dist)      # 1. 加上 min_dist
#             u = x / z
#             v = y / z
#             inside = (valid &
#                     (u >= 1) & (u < out_w - 1) &   # 2. 留边
#                     (v >= 1) & (v < out_h - 1))
#             if not inside.any():
#                 continue
#             # Convert to integer pixel coordinates
#             u_i = u[inside].long()
#             v_i = v[inside].long()
#             z_i = z[inside]
#             inten_i = inten[inside]
#             # Update depth buffer and pseudo image
#             for uu, vv, zz, ii in zip(u_i, v_i, z_i, inten_i):
#                 if zz < depth_buffer[b, cam, vv, uu]:
#                     depth_buffer[b, cam, vv, uu] = zz
#                     # Channel 0: normalized depth
#                     pseudo[b, cam, 0, vv, uu] = min(zz / max_depth, 1.0)
#                     # Channel 1: intensity (clipped to [0,1])
#                     pseudo[b, cam, 1, vv, uu] = min(max(ii, 0.0), 1.0)
#                     # Channel 2: mask (always 1 for valid projection)
#                     pseudo[b, cam, 2, vv, uu] = 1.0
    
#     print(f"Final pseudo image device: {pseudo.device}")
#     print(f"Final pseudo image shape: {pseudo.shape}")
    
#     return pseudo

def lidar_to_pseudo_image_no_resize(
        lidar, batch_lidar2img,
        out_h=256, out_w=704,
        min_dist=1.0):
    """
    与 nuScenes 官方 map_pointcloud_to_image 深度分支像素级对齐的 batch 版。
    lidar: list[8] of (Ni,5) numpy arrays
    batch_lidar2img: list[6] of tensor(8,4,4)  # 6 相机，每帧一个 3×4 投影矩阵
    return: pseudo (8,6,1,H,W)  单通道，原始深度值；无效位置 = 0
    """
    import torch
    import numpy as np
    # ---------- 工具 ----------
    def to_tensor(x, device=None):
        if isinstance(x, torch.Tensor):
            t = x.float()
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
        else:
            t = torch.tensor(x, dtype=torch.float32)
        return t.to(device) if device else t
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # ---------- 1. lidar → (B,N,5) ----------
    if isinstance(lidar, list):
        lidar_tensors = [to_tensor(p, device) for p in lidar]
        B = len(lidar_tensors)
        maxN = max(p.shape[0] for p in lidar_tensors)
        lidar = torch.zeros((B, maxN, 5), dtype=torch.float32, device=device)
        for i, p in enumerate(lidar_tensors):
            lidar[i, :p.shape[0]] = p
    else:
        lidar = to_tensor(lidar, device)
        B = lidar.shape[0]
    # ---------- 2. lidar2img → (B,6,4,4) ----------
    lidar2img = torch.stack([to_tensor(m, device) for m in batch_lidar2img], dim=0).transpose(0, 1)  # (B,6,4,4)
    # ---------- 3. 输出 buffer ----------
    depth_img = torch.zeros((B, 6, out_h, out_w), device=device)  # 单通道深度图
    # ---------- 4. 投影 ----------
    xyz = lidar[..., :3]          # (B,N,3)
    ones = torch.ones((B, xyz.shape[1], 1), device=device)
    pts_h = torch.cat([xyz, ones], dim=-1)  # (B,N,4)
    for b in range(B):
        pts = pts_h[b]                                    # (N,4)
        for cam in range(6):
            P = lidar2img[b, cam, :3, :]                  # (3,4)
            proj = P @ pts.T                              # (3,N)
            x, y, z = proj[0], proj[1], proj[2]
            # ---- 过滤：与官方完全一致 ----
            valid = z > min_dist                          # 1. 距离下限
            u = x / z
            v = y / z
            inside = (
                valid
                & (u >= 1) & (u < out_w - 1)              # 2. 留 1 像素边距
                & (v >= 1) & (v < out_h - 1))
            if not inside.any():
                continue
            u_i = u[inside].long()
            v_i = v[inside].long()
            z_i = z[inside]
            # ---- 随机覆盖（无 z-buffer，同原版）----
            # 把 (v_i, u_i) 展成 1D 索引
            idx = v_i * out_w + u_i
            depth_img[b, cam].view(-1).scatter_(dim=0, index=idx, src=z_i, reduce='replace')

    return depth_img.unsqueeze(2)  # (B,6,1,H,W)


@DETECTORS.register_module()
class SparseDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        # img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)
        self.img_backbone = build_backbone(img_backbone)
        if pretrained is not None:
            self.img_backbone.pretrained = pretrained
        # if img_neck is not None:
        #     self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func

        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 

    @auto_fp16(apply_to=("img",), out_fp32=True)
    # def extract_feat(self, img, lidar=None, return_depth=False, return_loss=False, rescale=False, metas=None, velocity=None, data=None, img_metas=None, **kwargs):
    def extract_feat(self, img, lidar=None, return_depth=False, return_loss=False, rescale=False, metas=None, data=None, img_metas=None, **kwargs):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        # print("lidar shape in extract_feat:", lidar.shape if lidar is not None else None)
        # outputs = self.img_backbone(img, lidar=lidar, velocity=velocity)
        outputs = self.img_backbone(img, lidar=lidar)
        if isinstance(outputs, tuple):
            feature_maps = outputs[0]
            extra = outputs[1:]
        else:
            feature_maps = outputs
            extra = None

        new_feature_maps = []
        for feat in feature_maps:
            assert isinstance(feat, torch.Tensor), f"Expected Tensor, got {type(feat)}"
            # new_feature_maps.append(
            #     feat.view(bs, num_cams, *feat.shape[1:])
            # )
            if feat.shape[0] == bs * num_cams:
                feat = feat.view(bs, num_cams, *feat.shape[1:])
            elif feat.shape[0] == bs:
                feat = feat.unsqueeze(1)  # 保持维度一致，形状变为 [bs, 1, C, H, W]
            else:
                raise RuntimeError(
                    f"Unexpected feat shape {feat.shape}, expected batch {bs} or {bs*num_cams}"
                )
            new_feature_maps.append(feat)
        feature_maps = new_feature_maps

        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal") if metas else None)
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)

        if return_depth:
            return feature_maps, depths
        else:
            return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        lidar_points = data.get("points", None)
        batch_lidar2img = data.get("lidar2img", None)
        lidar_images = lidar_to_pseudo_image_no_resize(lidar_points, batch_lidar2img)
        lidar_images = lidar_images.to(torch.float16)
        # lidar_images = converter(lidar_points) if lidar_points is not None else None
        allowed_keys = {"metas", "img_metas"} 
        data_filtered = {k: v for k, v in data.items() if k in allowed_keys}
        feature_maps, depths = self.extract_feat(img, lidar_images, return_depth=True, **data_filtered)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            ) # depth(48,512,8,22),  gt_depth()
        return output
  
    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        lidar_points = data.get("points", None)
        batch_lidar2img = data.get("lidar2img", None)
        lidar_images = lidar_to_pseudo_image_no_resize(
                    lidar_points, 
                    batch_lidar2img)
        if img.dtype == torch.half:    # 只有 FP16 模式才 half()
            lidar_images = lidar_images.half()
        # print(">>> lidar_points:", lidar_points)
        # img(1,6,3.256.704)
        # lidar_images = converter(lidar_points) if lidar_points is not None else None # 8,4,256,704
        feature_maps = self.extract_feat(img=img, lidar=lidar_images, return_depth=False, **data)
        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)



# @DETECTORS.register_module()
# class SparseDrive(BaseDetector):
#     def __init__(
#         self,
#         img_backbone=None,
#         head=None,
#         img_neck=None,
#         init_cfg=None,
#         train_cfg=None,
#         test_cfg=None,
#         pretrained=None,
#         use_grid_mask=True,
#         use_deformable_func=False,
#         depth_branch=None,
#     ):
#         super(SparseDrive, self).__init__(init_cfg=init_cfg)
#         # if pretrained is not None:
#         #     backbone.pretrained = pretrained
#         # self.img_backbone = build_backbone(img_backbone)
#         if isinstance(img_backbone, nn.Module):
#             self.img_backbone = img_backbone
#         else:
#             self.img_backbone = build_backbone(img_backbone)
#         if img_neck is not None:
#             self.img_neck = build_neck(img_neck)
#         self.head = build_head(head)
#         self.use_grid_mask = use_grid_mask
#         if use_deformable_func:
#             assert DAF_VALID, "deformable_aggregation needs to be set up."
#         self.use_deformable_func = use_deformable_func
#         if depth_branch is not None:
#             self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
#         else:
#             self.depth_branch = None
#         if use_grid_mask:
#             self.grid_mask = GridMask(
#                 True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
#             ) 

#     @auto_fp16(apply_to=("img",), out_fp32=True)
#     def extract_feat(self, img, return_depth=False, metas=None):
#         # if metas is not None:
#         #     print('metas keys:', list(metas.keys()))
#         #     print('lidar:', metas.get('lidar'))
#         #     print('velocity:', metas.get('velocity'))
#         # else:
#         #     print('metas is None')
#         bs = img.shape[0]
#         if img.dim() == 5:  # multi-view
#             num_cams = img.shape[1]
#             img = img.flatten(end_dim=1)
#         else:
#             num_cams = 1
#         if self.use_grid_mask:
#             img = self.grid_mask(img)

#         lidar_points = data.get("points", None)
#         lidar_images = converter(lidar_points) if lidar_points is not None else None   

#         # backbone_sig = signature(self.img_backbone.forward)
#         # kwargs = {}
#         # if 'metas' in backbone_sig.parameters:
#         #     kwargs['metas'] = metas
#         # if 'lidar' in backbone_sig.parameters:
#         #     lidar_tensor = metas.get('lidar', metas.get('points', None))  # Look for 'lidar' or 'points'
#         #     if lidar_tensor is not None:
#         #         if hasattr(lidar_tensor, 'data'):
#         #             lidar_tensor = lidar_tensor.data
#         #         kwargs['lidar'] = lidar_tensor
#         # if 'velocity' in backbone_sig.parameters:
#         #     kwargs['velocity'] = metas.get('velocity')
#         # if 'num_cams' in backbone_sig.parameters:
#         #     kwargs['num_cams'] = num_cams  
#         # feature_maps = self.img_backbone(img, **kwargs)    
#         if "metas" in signature(self.img_backbone.forward).parameters:
#             feature_maps = self.img_backbone(img, num_cams, metas=metas)
#         else:
#             feature_maps = self.img_backbone(img)
#         if self.img_neck is not None:
#             feature_maps = list(self.img_neck(feature_maps))
#         for i, feat in enumerate(feature_maps):
#             feature_maps[i] = torch.reshape(
#                 feat, (bs, num_cams) + feat.shape[1:]
#             )
#         if return_depth and self.depth_branch is not None:
#             depths = self.depth_branch(feature_maps, metas.get("focal"))
#         else:
#             depths = None
#         if self.use_deformable_func:
#             feature_maps = feature_maps_format(feature_maps)
#         if return_depth:
#             return feature_maps, depths
#         return feature_maps

#     @force_fp32(apply_to=("img",))
#     def forward(self, img, **data):
#         if self.training:
#             return self.forward_train(img, **data)
#         else:
#             return self.forward_test(img, **data)

#     def forward_train(self, img, **data):
#         feature_maps, depths = self.extract_feat(img, True, data)
#         model_outs = self.head(feature_maps, data)
#         output = self.head.loss(model_outs, data)
#         if depths is not None and "gt_depth" in data:
#             output["loss_dense_depth"] = self.depth_branch.loss(
#                 depths, data["gt_depth"]
#             )
#         return output

#     def forward_test(self, img, **data):
#         if isinstance(img, list):
#             return self.aug_test(img, **data)
#         else:
#             return self.simple_test(img, **data)

#     def simple_test(self, img, **data):
#         feature_maps = self.extract_feat(img)

#         model_outs = self.head(feature_maps, data)
#         results = self.head.post_process(model_outs, data)
#         output = [{"img_bbox": result} for result in results]
#         return output

#     def aug_test(self, img, **data):
#         # fake test time augmentation
#         for key in data.keys():
#             if isinstance(data[key], list):
#                 data[key] = data[key][0]
#         return self.simple_test(img[0], **data)
