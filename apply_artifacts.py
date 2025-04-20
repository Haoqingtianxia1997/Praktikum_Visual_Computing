import yaml
from artifacts.artifacts import ArtifactBatchProcessor

config_path = "./artifacts/config.yaml"
with open(config_path, 'r') as file:
    config = yaml.full_load(file)

src_folders = config["src_folders"]
custom_overlay = config["overlay_counts"]
output_folders = config["output_folders"]
mask_folders = config["mask_folders"]

processor = ArtifactBatchProcessor(src_folders, overlay_folders=custom_overlay, output_folders=output_folders)
processor.process_all_images()
processor.copy_masks(mask_folders)
