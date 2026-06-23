import copy
import logging
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from filelock import FileLock
import shutil

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.random_eraser import RandomErasing

logger = logging.getLogger(__name__)

register_codecs()

# Not stored in zarr; filled in __getitem__ when using relative obs + absolute side channels.
_SYNTHETIC_LOWDIM_KEYS = frozenset({"base_length_abs", "joint_angle_abs"})


def _coerce_trailing_shape(data: np.ndarray, expected_shape: tuple, key: str) -> np.ndarray:
    if data.shape[1:] == expected_shape:
        return data
    logger.warning(
        "Key %s shape %s != expected %s; slicing/padding to match.",
        key,
        data.shape[1:],
        expected_shape,
    )
    out = np.zeros((data.shape[0],) + expected_shape, dtype=data.dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(data.shape[1:], expected_shape))
    out[(slice(None),) + slices] = data[(slice(None),) + slices]
    return out


class FreeSpaceVineDataset(BaseDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str] = None,
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_duration: Optional[float] = None,
        random_dropout: bool = False,
        random_dropout_prob: float = 0.05,
        random_erase: bool = False,
        random_erase_prob: float = 0.5,
        random_erase_scale: list = [0.0, 0.1],
        random_erase_ratio: list = [0.3, 3.3],
        sparse_query_frequency_down_sample_steps: int = 1,
        steering_sample_ratio: Optional[float] = None,
        steering_joint_delta_threshold: float = 1e-3,
        use_relative_action_obs: bool = False,
    ):
        if cache_dir is None:
            with zarr.ZipStore(dataset_path, mode="r") as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, store=zarr.MemoryStore()
                )
            print("loaded!")
        else:
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split(".")[0]
            cache_name = "_".join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + ".zarr.mdb")
            lock_path = cache_dir.joinpath(cache_name + ".lock")

            print("Acquiring lock on cache.")
            with FileLock(lock_path):
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(
                            str(cache_path),
                            writemap=True,
                            metasync=False,
                            sync=False,
                            map_async=True,
                            lock=False,
                        ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode="r") as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store, store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e

            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(group=zarr.group(store))

        self.num_robot = 1
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta["obs"]
        lowdim_shapes = dict()
        for key, attr in obs_shape_meta.items():
            if attr.get("ignore_by_policy", True):
                continue
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)
                lowdim_shapes[key] = shape_meta["obs"][key]["shape"]

            horizon = shape_meta["obs"][key]["horizon"]
            key_horizon[key] = horizon

            latency_steps = shape_meta["obs"][key]["latency_steps"]
            key_latency_steps[key] = latency_steps

            down_sample_steps = shape_meta["obs"][key]["down_sample_steps"]
            key_down_sample_steps[key] = down_sample_steps

        key_horizon["action"] = shape_meta["action"]["horizon"]
        key_latency_steps["action"] = shape_meta["action"]["latency_steps"]
        key_down_sample_steps["action"] = shape_meta["action"]["down_sample_steps"]

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if "wrt" not in key and key not in _SYNTHETIC_LOWDIM_KEYS:
                self.sampler_lowdim_keys.append(key)

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration,
            sparse_query_frequency_down_sample_steps=sparse_query_frequency_down_sample_steps,
        )
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.lowdim_shapes = lowdim_shapes
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False
        self.steering_sample_ratio = steering_sample_ratio
        self.steering_joint_delta_threshold = steering_joint_delta_threshold
        self.use_relative_action_obs = use_relative_action_obs

        if random_erase:
            self.random_erase = RandomErasing(
                p=random_erase_prob, scale=random_erase_scale, ratio=random_erase_ratio
            )
        else:
            self.random_erase = None

        self.random_dropout = random_dropout
        self.random_dropout_prob = random_dropout_prob
        self.sparse_query_frequency_down_sample_steps = sparse_query_frequency_down_sample_steps
        self._apply_steering_sampling(seed=seed)

    def _get_joint_angle_array(self) -> Optional[np.ndarray]:
        if "joint_angle" in self.sampler.replay_buffer:
            joint_arr = self.sampler.replay_buffer["joint_angle"]
        elif "action" in self.sampler.replay_buffer and self.sampler.replay_buffer["action"].shape[1] > 1:
            joint_arr = self.sampler.replay_buffer["action"][:, 1:]
        else:
            return None
        joint_arr = np.asarray(joint_arr)
        if joint_arr.ndim == 1:
            joint_arr = joint_arr[:, None]
        return joint_arr

    def _compute_steering_mask(self, indices: list) -> Optional[np.ndarray]:
        joint_arr = self._get_joint_angle_array()
        if joint_arr is None:
            logger.warning("Steering sampling requested but no joint-angle data found.")
            return None

        action_horizon = self.key_horizon["action"]
        action_down_sample_steps = self.key_down_sample_steps["action"]
        offsets = np.arange(action_horizon, dtype=np.int64) * action_down_sample_steps

        idxs = np.asarray([idx[0] for idx in indices], dtype=np.int64)
        end_idxs = np.asarray([idx[2] for idx in indices], dtype=np.int64)

        action_idxs = idxs[:, None] + offsets[None, :]
        if self.action_padding:
            action_idxs = np.minimum(action_idxs, (end_idxs - 1)[:, None])
        joint_seq = joint_arr[action_idxs]
        joint_delta = np.max(
            np.abs(joint_seq - joint_seq[:, :1, :]),
            axis=(1, 2),
        )
        return joint_delta > self.steering_joint_delta_threshold

    def _apply_steering_sampling(self, seed: int) -> None:
        if self.steering_sample_ratio is None:
            return
        if not (0.0 <= self.steering_sample_ratio <= 1.0):
            raise ValueError("steering_sample_ratio must be within [0, 1].")

        indices = list(self.sampler.indices)
        if not indices:
            return

        steering_mask = self._compute_steering_mask(indices)
        if steering_mask is None:
            return

        steering_idx = np.flatnonzero(steering_mask)
        growing_idx = np.flatnonzero(~steering_mask)
        n_s = int(steering_idx.shape[0])
        n_g = int(growing_idx.shape[0])
        total = len(indices)
        assert n_s + n_g == total
        r = self.steering_sample_ratio

        if n_s == 0:
            logger.warning("No steering samples detected; skipping steering rebalancing.")
            return

        # Keep every sequence once (no subsampling). Duplicate steering sequences only
        # so that steering rows / (all rows) ~= r after augmentation.
        out = list(indices)
        rng = np.random.default_rng(seed=seed)
        n_extra = 0
        if r < 1.0:
            # n_s + n_extra = r * (total + n_extra)  =>  n_extra = (r * total - n_s) / (1 - r)
            n_extra = max(0, int(round((r * total - n_s) / (1.0 - r))))
            if n_extra > 0:
                dup_src = rng.choice(steering_idx, size=n_extra, replace=True)
                out.extend(indices[i] for i in dup_src)
        rng.shuffle(out)
        self.sampler.indices = out

        n_steering_rows = n_s + n_extra
        n_total_out = len(out)
        logger.info(
            "Steering oversample: kept all %d sequences (growing=%d, steering=%d). "
            "Target steering fraction=%.3f; after duplicates steering=%d / total=%d (%.3f).",
            total,
            n_g,
            n_s,
            r,
            n_steering_rows,
            n_total_out,
            n_steering_rows / n_total_out if n_total_out else 0.0,
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration,
            sparse_query_frequency_down_sample_steps=self.sparse_query_frequency_down_sample_steps,
        )
        val_set.val_mask = self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        data_cache = {key: list() for key in self.lowdim_keys + ["action"]}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=32,
            num_workers=16,
        )
        for batch in tqdm(dataloader, desc="iterating dataset to get normalization"):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch["obs"][key]))
            data_cache["action"].append(copy.deepcopy(batch["action"]))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            B, T = data_cache[key].shape[:2]
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B * T, -1)

        normalizer["action"] = get_range_normalizer_from_stat(
            array_to_stats(data_cache["action"])
        )

        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            normalizer[key] = get_range_normalizer_from_stat(stat)

        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def _apply_relative_representation(
        self, data: Dict[str, np.ndarray], action: np.ndarray
    ) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Convert selected low-dim observations and action to be relative to the latest
        observation frame. For this task we use direct numeric deltas on base length
        and joint representations instead of pose transforms.
        """
        ref = {key: data[key][-1] for key in data.keys()}
        data = {key: data[key] - ref[key] for key in data.keys()}
        action = action - np.concatenate([ref["base_encoder"], ref["joint_angle"]], axis=-1)
        return data, action

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if key not in data:
                continue
            if self.random_dropout and np.random.rand() < self.random_dropout_prob:
                obs_dict[key] = torch.zeros(
                    (data[key].shape[0], 3, data[key].shape[1], data[key].shape[2]),
                    dtype=torch.float32,
                )
            else:
                rgb = torch.from_numpy(
                    np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
                )
                obs_dict[key] = rgb
            del data[key]
        for key in self.sampler_lowdim_keys:
            expected = tuple(self.shape_meta["obs"][key]["shape"])
            data[key] = _coerce_trailing_shape(data[key], expected, key)

        action_expected = tuple(self.shape_meta["action"]["shape"])
        data["action"] = _coerce_trailing_shape(data["action"], action_expected, "action")
        if self.use_relative_action_obs:
            # Synthetic absolute channels: only if present in resolved shape_meta (lowdim_keys).
            base_length_abs = (
                data["base_encoder"].copy() if "base_length_abs" in self.lowdim_keys else None
            )
            joint_angle_abs = (
                data["joint_angle"].copy()
                if "joint_angle_abs" in self.lowdim_keys and "joint_angle" in data
                else None
            )
            data, data["action"] = self._apply_relative_representation(data, data["action"])
            if base_length_abs is not None:
                data["base_length_abs"] = base_length_abs
            if joint_angle_abs is not None:
                data["joint_angle_abs"] = joint_angle_abs
        else:
            # Same keys as absolute sources when shape_meta includes *_abs but not training relative.
            if "joint_angle_abs" in self.lowdim_keys and "joint_angle" in data:
                data["joint_angle_abs"] = data["joint_angle"].copy()
            if "base_length_abs" in self.lowdim_keys and "base_encoder" in data:
                data["base_length_abs"] = data["base_encoder"].copy()

        for key in self.lowdim_keys:
            if key not in data:
                continue
            expected = tuple(self.shape_meta["obs"][key]["shape"])
            data[key] = _coerce_trailing_shape(data[key], expected, key)
            obs_dict[key] = torch.from_numpy(data[key].astype(np.float32))
            del data[key]

        torch_data = {
            "obs": obs_dict,
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data
