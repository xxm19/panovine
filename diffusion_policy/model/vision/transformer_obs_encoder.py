import copy

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from typing import Optional, Tuple

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)


def rot6d_to_mat(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1, 6)
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def mat_to_rot6d(mat: torch.Tensor) -> torch.Tensor:
    return mat[..., :3, :2].reshape(*mat.shape[:-2], 6)


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = F.normalize(axis, dim=-1)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(axis.shape[:-1] + (3, 3))
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).reshape(
        (1,) * len(axis.shape[:-1]) + (3, 3)
    )
    sin = torch.sin(angle)[..., None, None]
    cos = torch.cos(angle)[..., None, None]
    return eye + sin * K + (1.0 - cos) * (K @ K)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class TransformerObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str='vit_base_patch16_clip_224.openai',
            global_pool: str='',
            transforms: list=None,
            n_emb: int=768,
            pretrained: bool=False,
            frozen: bool=False,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            feature_aggregation: str=None,
            downsample_ratio: int=32,
            add_camera_poses: bool=False,
            imagenet_norm: bool=True,   # normalize input with imagenet mean and std
            concat_camera_poses: bool=False,
            camera_pose_mode: str='legacy',
            use_camera_attention: bool=False,
            camera_attn_heads: int=8,
            camera_attn_dropout: float=0.0,
            use_relative_pose_bias: bool=False,
            relative_pose_type: str='se3',
            relative_pose_mlp_hidden: int=256,
            use_arc_bias: bool=False,
            arc_bins: int=32,
            arc_max: float=1.0,
            camera_s_key: str='camera_s',
            pose_noise_std: Optional[dict]=None,
            pose_noise_global_prob: float=0.0,
            camera_pose_mlp_hidden: int=256,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )
        self.model_name = model_name

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
            
        # handle feature aggregation
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit'):
            if self.feature_aggregation is None:
                pass
            elif self.feature_aggregation != 'cls':
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = 'cls'
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        if imagenet_norm:
            if transforms is None:
                transforms = []
            transforms.append(torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                # for shared model, we only need to create one model
                if share_rgb_model and len(rgb_keys) > 1:
                    continue
                
                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                proj = nn.Identity()
                n_emb_tmp = n_emb // 2 if concat_camera_poses else n_emb
                if feature_size != n_emb_tmp:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb_tmp)
                key_projection_map[key] = proj

                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                if attr.get('ignore_by_policy', False):
                    continue
                if key == camera_s_key:
                    continue
                if not add_camera_poses and (key == 'camera_pos' or key == 'camera_ori'):
                    continue
                if camera_pose_mode == 'mlp' and (key == 'camera_pos' or key == 'camera_ori'):
                    continue
                dim = shape[-1] if key == 'camera_pos' or key == 'camera_ori' else np.prod(shape)
                proj = nn.Identity()
                if concat_camera_poses and (key == 'camera_pos' or key == 'camera_ori'):
                    n_emb_tmp = n_emb // 4
                elif key == 'camera_pos' or key == 'camera_ori':
                    n_emb_tmp = n_emb // 2
                else:
                    n_emb_tmp = n_emb
                if dim != n_emb_tmp:
                    proj = nn.Linear(in_features=dim, out_features=n_emb_tmp)
                key_projection_map[key] = proj

                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_projection_map = key_projection_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.add_camera_poses = add_camera_poses
        self.concat_camera_poses = concat_camera_poses
        self.camera_pose_mode = camera_pose_mode
        self.use_camera_attention = use_camera_attention
        self.camera_attn_heads = camera_attn_heads
        self.camera_attn_dropout = camera_attn_dropout
        self.use_relative_pose_bias = use_relative_pose_bias
        self.relative_pose_type = relative_pose_type
        self.relative_pose_mlp_hidden = relative_pose_mlp_hidden
        self.use_arc_bias = use_arc_bias
        self.arc_bins = arc_bins
        self.arc_max = arc_max
        self.camera_s_key = camera_s_key
        self.pose_noise_std = pose_noise_std
        self.pose_noise_global_prob = pose_noise_global_prob
        self.camera_pose_mlp_hidden = camera_pose_mlp_hidden
        self.num_cameras = len(rgb_keys)

        if self.camera_pose_mode not in ('legacy', 'mlp'):
            raise ValueError(f"Unsupported camera_pose_mode: {self.camera_pose_mode}")
        # camera_pose_mode is relevant only when camera pose inputs are enabled.
        # Keep the strict mlp-mode requirement limited to that case.
        if (
            self.add_camera_poses
            and self.camera_pose_mode == 'mlp'
            and not self.use_camera_attention
        ):
            raise ValueError("camera_pose_mode='mlp' requires use_camera_attention=True.")

        if self.use_camera_attention:
            self.camera_attn = nn.MultiheadAttention(
                n_emb,
                camera_attn_heads,
                dropout=camera_attn_dropout,
                batch_first=True
            )
            self.camera_attn_ln = nn.LayerNorm(n_emb)
            if self.feature_aggregation is None:
                raise ValueError("use_camera_attention requires feature_aggregation='cls' for ViT.")

        if self.add_camera_poses and self.camera_pose_mode == 'mlp':
            pose_out_dim = n_emb // 2 if concat_camera_poses else n_emb
            self.camera_pose_mlp = nn.Sequential(
                nn.Linear(9, camera_pose_mlp_hidden),
                nn.ReLU(),
                nn.Linear(camera_pose_mlp_hidden, pose_out_dim)
            )
        else:
            self.camera_pose_mlp = None

        if self.use_relative_pose_bias:
            if self.relative_pose_type != 'se3':
                raise ValueError(f"Unsupported relative_pose_type: {self.relative_pose_type}")
            self.relative_pose_mlp = nn.Sequential(
                nn.Linear(9, relative_pose_mlp_hidden),
                nn.ReLU(),
                nn.Linear(relative_pose_mlp_hidden, camera_attn_heads)
            )
        else:
            self.relative_pose_mlp = None

        if self.use_arc_bias:
            self.arc_bias = nn.Embedding(arc_bins, camera_attn_heads)
        else:
            self.arc_bias = None

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        # Return: B, N, C
        
        if self.model_name.startswith('vit'):
            # vit uses the CLS token
            if self.feature_aggregation == 'cls':
                return feature[:, [0], :]
            
            # or use all tokens
            assert self.feature_aggregation is None 
            return feature
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1, keepdim=True)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1, keepdim=True)
        else:
            assert self.feature_aggregation is None
            return feature

    def _apply_pose_noise(
        self, camera_pos: torch.Tensor, camera_ori: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.pose_noise_std is None:
            return camera_pos, camera_ori

        trans_std = float(self.pose_noise_std.get('trans', 0.0))
        rot_deg = float(self.pose_noise_std.get('rot_deg', 0.0))
        if trans_std <= 0.0 and rot_deg <= 0.0:
            return camera_pos, camera_ori

        B, T, V, _ = camera_pos.shape
        pos_noise = torch.zeros_like(camera_pos)
        if trans_std > 0.0:
            pos_noise = torch.randn_like(camera_pos) * trans_std
            if self.pose_noise_global_prob > 0.0:
                global_mask = torch.rand((B, T, 1, 1), device=camera_pos.device)
                global_apply = (global_mask < self.pose_noise_global_prob).float()
                global_noise = torch.randn((B, T, 1, 3), device=camera_pos.device) * trans_std
                pos_noise = pos_noise + global_apply * global_noise
        camera_pos = camera_pos + pos_noise

        if rot_deg > 0.0:
            rot_std = rot_deg * np.pi / 180.0
            axis = torch.randn((B, T, V, 3), device=camera_ori.device, dtype=camera_ori.dtype)
            angle = torch.randn((B, T, V), device=camera_ori.device, dtype=camera_ori.dtype) * rot_std
            rot_noise = axis_angle_to_matrix(axis, angle)
            if self.pose_noise_global_prob > 0.0:
                global_mask = torch.rand((B, T, 1, 1), device=camera_ori.device) < self.pose_noise_global_prob
                global_axis = torch.randn((B, T, 1, 3), device=camera_ori.device, dtype=camera_ori.dtype)
                global_angle = torch.randn((B, T, 1), device=camera_ori.device, dtype=camera_ori.dtype) * rot_std
                global_rot = axis_angle_to_matrix(global_axis, global_angle)
                rot_noise = torch.where(
                    global_mask[..., None],
                    global_rot.expand_as(rot_noise),
                    rot_noise
                )
            rot_mat = rot6d_to_mat(camera_ori.reshape(-1, 6)).reshape(B, T, V, 3, 3)
            rot_mat = rot_noise @ rot_mat
            camera_ori = mat_to_rot6d(rot_mat)

        return camera_pos, camera_ori

    def _compute_relative_pose_bias(
        self, camera_pos: torch.Tensor, camera_ori: torch.Tensor
    ) -> torch.Tensor:
        B, T, V, _ = camera_pos.shape
        pos = camera_pos.reshape(B * T, V, 3)
        rot = rot6d_to_mat(camera_ori.reshape(B * T * V, 6)).reshape(B * T, V, 3, 3)

        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rot_j = rot[:, None, :, :, :]
        rot_i = rot[:, :, None, :, :]
        rel_rot = torch.matmul(rot_j.transpose(-1, -2), rot_i)
        rel_rot6d = mat_to_rot6d(rel_rot)
        rel_feat = torch.cat([rel_pos, rel_rot6d], dim=-1)
        return self.relative_pose_mlp(rel_feat)

    def _compute_arc_bias(self, camera_s: Optional[torch.Tensor]) -> torch.Tensor:
        V = self.num_cameras
        if camera_s is None:
            base = torch.linspace(0.0, 1.0, V, device=self.device, dtype=self.dtype)
            camera_s = base.reshape(1, 1, V, 1)
        if camera_s.dim() == 3:
            camera_s = camera_s.unsqueeze(-1)
        B, T, V, _ = camera_s.shape
        rel = torch.abs(camera_s.unsqueeze(3) - camera_s.unsqueeze(2)).squeeze(-1)
        rel = torch.clamp(rel / max(self.arc_max, 1e-6), 0.0, 1.0)
        bins = (rel * (self.arc_bins - 1)).round().long()
        arc_bias = self.arc_bias(bins)
        return arc_bias.reshape(B * T, V, V, -1)

    def _encode_camera_tokens(self, obs_dict):
        imgs = [obs_dict[key] for key in self.rgb_keys]
        B, T = imgs[0].shape[:2]
        V = len(imgs)
        for key, img in zip(self.rgb_keys, imgs):
            assert img.shape[2:] == self.key_shape_map[key]

        if self.share_rgb_model:
            img_stack = torch.stack(imgs, dim=2)
            img = img_stack.reshape(B * T * V, *img_stack.shape[3:])
            img = self.key_transform_map[self.rgb_keys[0]](img)
            raw_feature = self.key_model_map[self.rgb_keys[0]](img)
            feature = self.aggregate_feature(raw_feature)
            emb = self.key_projection_map[self.rgb_keys[0]](feature)
        else:
            emb_list = []
            for key, img in zip(self.rgb_keys, imgs):
                img = img.reshape(B * T, *img.shape[2:])
                img = self.key_transform_map[key](img)
                raw_feature = self.key_model_map[key](img)
                feature = self.aggregate_feature(raw_feature)
                emb_list.append(self.key_projection_map[key](feature))
            img_embs = []
            for emb_item in emb_list:
                if emb_item.shape[1] != 1:
                    raise ValueError("Camera attention expects one token per view (set feature_aggregation='cls').")
                emb_item = emb_item.squeeze(1).reshape(B, T, 1, -1)
                img_embs.append(emb_item)
            img_emb = torch.cat(img_embs, dim=2)

        if self.share_rgb_model:
            if emb.shape[1] != 1:
                raise ValueError("Camera attention expects one token per view (set feature_aggregation='cls').")
            emb = emb.squeeze(1)
            img_emb = emb.reshape(B, T, V, -1)

        camera_pos = None
        camera_ori = None
        if self.add_camera_poses:
            camera_pos = obs_dict["camera_pos"]
            camera_ori = obs_dict["camera_ori"]
            if camera_pos.dim() == 3:
                camera_pos = camera_pos.unsqueeze(1).repeat(1, T, 1, 1)
                camera_ori = camera_ori.unsqueeze(1).repeat(1, T, 1, 1)
            camera_pos, camera_ori = self._apply_pose_noise(camera_pos, camera_ori)
            if self.camera_pose_mode == 'mlp':
                pose_input = torch.cat([camera_pos, camera_ori], dim=-1)
                pose_emb = self.camera_pose_mlp(pose_input)
            else:
                pose_pos = self.key_projection_map["camera_pos"](camera_pos.reshape(-1, 3))
                pose_ori = self.key_projection_map["camera_ori"](camera_ori.reshape(-1, 6))
                if self.concat_camera_poses:
                    pose_pos = pose_pos.reshape(B, T, V, self.n_emb // 4)
                    pose_ori = pose_ori.reshape(B, T, V, self.n_emb // 4)
                else:
                    pose_pos = pose_pos.reshape(B, T, V, self.n_emb // 2)
                    pose_ori = pose_ori.reshape(B, T, V, self.n_emb // 2)
                pose_emb = torch.cat([pose_pos, pose_ori], dim=-1)

            if self.concat_camera_poses:
                cam_tokens = torch.cat([img_emb, pose_emb], dim=-1)
            else:
                cam_tokens = img_emb + pose_emb
        else:
            cam_tokens = img_emb

        cam_tokens = cam_tokens.reshape(B * T, V, -1)
        if self.use_camera_attention:
            bias = None
            if self.use_relative_pose_bias:
                if camera_pos is None or camera_ori is None:
                    raise ValueError("Relative pose bias requires camera_pos and camera_ori.")
                bias = self._compute_relative_pose_bias(camera_pos, camera_ori)
            if self.use_arc_bias:
                arc_bias = self._compute_arc_bias(obs_dict.get(self.camera_s_key))
                if bias is None:
                    bias = arc_bias
                else:
                    bias = bias + arc_bias
            if bias is not None:
                bias = bias.permute(0, 3, 1, 2).reshape(B * T * self.camera_attn_heads, V, V)
                attn_out, _ = self.camera_attn(cam_tokens, cam_tokens, cam_tokens, attn_mask=bias)
            else:
                attn_out, _ = self.camera_attn(cam_tokens, cam_tokens, cam_tokens)
            cam_tokens = self.camera_attn_ln(cam_tokens + attn_out)
        cam_tokens = cam_tokens.reshape(B, T * V, -1)
        return cam_tokens
        
    def forward(self, obs_dict):
        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        if self.use_camera_attention:
            cam_tokens = self._encode_camera_tokens(obs_dict)
            embeddings.append(cam_tokens)
        else:
            if self.share_rgb_model:
                img = torch.cat([obs_dict[key] for key in self.rgb_keys], dim=1)
                B, T = img.shape[:2]
                assert B == batch_size
                assert img.shape[2:] == self.key_shape_map[self.rgb_keys[0]]
                img = img.reshape(B * T, *img.shape[2:])
                img = self.key_transform_map[self.rgb_keys[0]](img)
                raw_feature = self.key_model_map[self.rgb_keys[0]](img)
                feature = self.aggregate_feature(raw_feature)
                emb = self.key_projection_map[self.rgb_keys[0]](feature)
                if self.concat_camera_poses:
                    assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb // 2
                    emb = emb.reshape(B, -1, self.n_emb // 2)
                else:
                    assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
                    emb = emb.reshape(B, -1, self.n_emb)
                if self.add_camera_poses:
                    camera_pos = obs_dict["camera_pos"].reshape(-1, 3)
                    camera_ori = obs_dict["camera_ori"].reshape(-1, 6)
                    camera_pos_emb = self.key_projection_map["camera_pos"](camera_pos)
                    camera_ori_emb = self.key_projection_map["camera_ori"](camera_ori)
                    if self.concat_camera_poses:
                        camera_pos_emb = camera_pos_emb.reshape(B, -1, self.n_emb // 4)
                        camera_ori_emb = camera_ori_emb.reshape(B, -1, self.n_emb // 4)
                    else:
                        camera_pos_emb = camera_pos_emb.reshape(B, -1, self.n_emb // 2)
                        camera_ori_emb = camera_ori_emb.reshape(B, -1, self.n_emb // 2)
                    camera_pose_emb = torch.cat([camera_pos_emb, camera_ori_emb], dim=-1)
                    assert camera_pose_emb.shape == emb.shape
                    if self.concat_camera_poses:
                        emb = torch.cat([emb, camera_pose_emb], dim=-1)
                    else:
                        emb = emb + camera_pose_emb
                embeddings.append(emb)
            else:
                for key in self.rgb_keys:
                    img = obs_dict[key]
                    B, T = img.shape[:2]
                    assert B == batch_size
                    assert img.shape[2:] == self.key_shape_map[key]
                    img = img.reshape(B * T, *img.shape[2:])
                    img = self.key_transform_map[key](img)
                    raw_feature = self.key_model_map[key](img)
                    feature = self.aggregate_feature(raw_feature)
                    emb = self.key_projection_map[key](feature)
                    assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
                    emb = emb.reshape(B, -1, self.n_emb)
                    embeddings.append(emb)

        # process lowdim input
        for key in self.low_dim_keys:
            if key == 'camera_pos' or key == 'camera_ori' or key == self.camera_s_key:
                continue
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.reshape(B,T,-1)
            emb = self.key_projection_map[key](data)
            assert emb.shape[-1] == self.n_emb
            embeddings.append(emb)
        
        # concatenate all features along t
        result = torch.cat(embeddings, dim=1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 3
        assert example_output.shape[0] == 1

        return example_output.shape


def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize('../diffusion_policy/config'):
        cfg = hydra.compose('train_diffusion_transformer_umi_workspace')
        OmegaConf.resolve(cfg)

    shape_meta = cfg.task.shape_meta
    encoder = TransformerObsEncoder(
        shape_meta=shape_meta
    )
