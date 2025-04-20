import torch
import numpy as np
import cv2 as cv
from scipy import ndimage

# import imageio
import imageio.v2 as imageio
from random import randint
import os
import random
import shutil
import yaml


this_dir, this_filename = os.path.split(__file__)
data_folder = os.path.join(this_dir, "imgs")


class ArtifactAugmentation:
    def transparentOverlay(
        self,
        src,
        mask,
        scale=2,
        mask_threshold=0.3,
        overlay_path="imgs/dark_spots/small_spot.png",
        width_slack=(0.3, 0.3),
        height_slack=(0.3, 0.3),
        ignore_index=None,
    ):
        src = src.permute(1, 2, 0)
        overlay = imageio.imread(overlay_path) / 255.0
        overlay = cv.resize(overlay, (0, 0), fx=scale, fy=scale)
        index = np.round(ndimage.measurements.center_of_mass(overlay[..., 3])).astype(
            np.int64
        )
        h_overlay, w_overlay, _ = overlay.shape
        h_img, w_img, _ = src.shape

        min_vert = -int(h_overlay * height_slack[0])
        max_vert = max(min_vert + 1, h_img + int(h_overlay * height_slack[1]))

        min_hor = -int(w_overlay * width_slack[0])
        max_hor = max(min_hor + 1, w_img + int(w_overlay * width_slack[1]))

        attempt = 0
        while attempt < 10:
            try:
                he = randint(
                    min_vert,
                    max_vert,
                )

                wi = randint(min_hor, max_hor)
                pos = (he - index[0], wi - index[1])

                # Assert conditions
                assert pos[0] + h_overlay > 0 and pos[0] - h_overlay - h_img < 0
                assert pos[1] + w_overlay > 0 and pos[1] - w_overlay - w_img < 0

                from_y_art = max(-pos[0], 0)
                from_x_art = max(-pos[1], 0)
                from_y = -min(-pos[0], 0)
                from_x = -min(-pos[1], 0)
                until_y = min(h_overlay - from_y_art + from_y, h_img)
                until_x = min(w_overlay - from_x_art + from_x, w_img)
                until_y_art = from_y_art + until_y - from_y
                until_x_art = from_x_art + until_x - from_x

                alpha = torch.from_numpy(overlay[:, :, 3])
                overlayed = (
                    alpha[from_y_art:until_y_art, from_x_art:until_x_art, np.newaxis]
                    * overlay[from_y_art:until_y_art, from_x_art:until_x_art, :3]
                    + (1 - alpha[from_y_art:until_y_art, from_x_art:until_x_art, np.newaxis])
                    * src[from_y:until_y, from_x:until_x, :]
                )
                src[from_y:until_y, from_x:until_x, :] = overlayed

                ood_indices = torch.from_numpy(
                    overlay[from_y_art:until_y_art, from_x_art:until_x_art, 3] > mask_threshold
                )

                if ignore_index is not None:
                    ignore_mask = mask == ignore_index

                mask[from_y:until_y, from_x:until_x][ood_indices] = 0

                if ignore_index is not None:
                    mask[ignore_mask] = ignore_index

                return src.permute(2, 0, 1), mask
            
            except AssertionError:
                attempt += 1
                if attempt == 10:
                    raise RuntimeError("Failed to overlay after maximum attempts")




class ArtifactBatchProcessor:
    def __init__(self, src_folders, overlay_folders=None, output_folders=None):
        # 获取当前文件的目录
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, 'config.yaml')

        # 使用确定的路径加载配置文件
        with open(config_path, 'r') as file:
            config = yaml.full_load(file)

        # 从配置文件读取默认的 overlay 路径
        default_overlay_paths = config["default_overlay_paths"]

        self.src_folders = {key: os.path.expanduser(val) for key, val in src_folders.items()}

        if overlay_folders is None:
            overlay_folders = {k: (default_overlay_paths[k], 1) for k in default_overlay_paths}
        else:
            for k, v in default_overlay_paths.items():
                if k in overlay_folders:
                    if isinstance(overlay_folders[k], tuple):
                        overlay_folders[k] = (v, overlay_folders[k][1])
                    else:
                        overlay_folders[k] = (v, overlay_folders[k])
                else:
                    overlay_folders[k] = (v, 1)

        self.overlay_folders = {k: (os.path.expanduser(v[0]), v[1]) for k, v in overlay_folders.items()}
        self.output_folders = {key: os.path.expanduser(val) for key, val in output_folders.items()}
        self.augmenter = ArtifactAugmentation()

        self._check_directories()

    def _check_directories(self):
        for folder in self.src_folders.values():
            if not os.path.exists(folder) or not os.path.isdir(folder):
                raise FileNotFoundError(f"The source folder '{folder}' does not exist.")
        for folder, _ in self.overlay_folders.values():
            if not os.path.exists(folder) or not os.path.isdir(folder):
                raise FileNotFoundError(f"The overlay folder '{folder}' does not exist.")
        for folder in self.output_folders.values():
            if not os.path.exists(folder):
                os.makedirs(folder)

    def _process_image(self, src_image_path, overlay_image_paths, output_folder):
        src = imageio.imread(src_image_path)
        mask = np.ones((src.shape[0], src.shape[1]), dtype=np.float32)

        src_tensor = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0  # (H, W, C) -> (C, H, W)
        mask_tensor = torch.from_numpy(mask).float()

        for overlay_image_path in overlay_image_paths:
            overlay = imageio.imread(overlay_image_path)
            overlay_mask = cv.cvtColor(overlay[:, :, :3], cv.COLOR_RGB2GRAY)
            overlay_mask = cv.resize(overlay_mask, (src.shape[1], src.shape[0]))

            augmented_image, updated_mask = self.augmenter.transparentOverlay(
                src=src_tensor,
                mask=mask_tensor,
                scale=0.5,
                mask_threshold=0.3,
                overlay_path=overlay_image_path,
            )

            src_tensor = augmented_image  # Update the src_tensor with the newly augmented image
            mask_tensor = updated_mask  # Update the mask_tensor with the updated mask

        augmented_image = augmented_image.permute(1, 2, 0).numpy() * 255.0
        updated_mask = updated_mask.numpy() * 255.0

        base_name = os.path.basename(src_image_path)
        augmented_image_path = os.path.join(output_folder, f"{base_name}")

        imageio.imwrite(augmented_image_path, augmented_image.astype(np.uint8))

        print(f"Augmented image saved to {augmented_image_path}")

    def process_all_images(self):
        for key, src_folder in self.src_folders.items():
            src_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
            overlay_files = {k: [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))] for k, (folder, _) in self.overlay_folders.items()}

            for src_image_path in src_files:
                overlay_image_paths = []
                for category, files in overlay_files.items():
                    count = self.overlay_folders[category][1] 
                    if files: 
                        overlay_image_paths.extend(random.choices(files, k=count))
                print(f"Processing {src_image_path} with overlays {overlay_image_paths}")
                self._process_image(src_image_path, overlay_image_paths, self.output_folders[key])

    def copy_masks(self, mask_folders):
            for key, mask_src_folder in mask_folders.items():
                mask_output_folder = os.path.join(self.output_folders[key], '..', 'masks')
                if not os.path.exists(mask_output_folder):
                    os.makedirs(mask_output_folder)
                for mask_file in os.listdir(mask_src_folder):
                    full_file_name = os.path.join(mask_src_folder, mask_file)
                    if os.path.isfile(full_file_name):
                        shutil.copy(full_file_name, mask_output_folder)
                        print(f"Copied mask {full_file_name} to {mask_output_folder}")