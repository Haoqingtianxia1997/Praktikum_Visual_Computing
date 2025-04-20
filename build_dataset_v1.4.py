'''
Befor you run this file:
1.  Copy the file to your project root directory.
2.  Copy <bad_data.txt> to your project root directory.
3.  If you have a pre-obtained list of validation set ids,
    then copy <val_id_list.txt> to your project root directory
    and set 'fixed=Ture' when calling the function 'divide_dataset_to_val'.
    If not, then set 'fixed=False' and don't need the <val_id_list.txt>.
'''
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from torchvision.io import read_image
import torch.nn.functional as F
import torch
import numpy as np
import os
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import random
import shutil
import tifffile as tiff
import argparse


def load_tif_as_tensor(img_path: str, cut: int, is_mask: bool = True, is_filtrate: bool = True) -> torch.Tensor:
    img = tiff.imread(img_path)
    if is_mask:
        img_tensor = torch.tensor(img).unsqueeze(0)
        h, w = img_tensor.shape[-2:]
        img_tensor = img_tensor[:, cut:h - cut, cut:w - cut]
        if is_filtrate:
            img_tensor = torch.where(img_tensor > 1, 0, img_tensor)
            img_tensor = (img_tensor > 0.5).byte() * 255
        else:
            img_tensor = torch.where(img_tensor > 1, 2, img_tensor)
            img_tensor = img_tensor.masked_fill(img_tensor == 1, 255)
            img_tensor = img_tensor.masked_fill(img_tensor == 0, 1)
            img_tensor = img_tensor.masked_fill(img_tensor == 2, 0)
    else:
        img_tensor = torch.tensor(img).permute(2, 0, 1)
        h, w = img_tensor.shape[-2:]
        img_tensor = img_tensor[:, cut:h - cut, cut:w - cut]

    return img_tensor


def load_png_as_tensor(img_path: str, cut: int, is_mask: bool = True, is_filtrate: bool = True) -> torch.Tensor:
    img = read_image(img_path)
    h, w = img.shape[-2:]
    img = img[:, cut:h-cut, cut:w-cut]
    if is_mask:
        if is_filtrate:
            img = torch.where(img > 1, 0, img)
            img = (img > 0.5).byte() * 255
        else:
            img = torch.where(img > 1, 2, img)
            img = img.masked_fill(img == 1, 255)
            img = img.masked_fill(img == 0, 1)
            img = img.masked_fill(img == 2, 0)
    return img


def calc_padding_size(img_tensor: torch.Tensor, size: int) -> Tuple[int, int, int, int]:
    h, w = img_tensor.shape[-2:]
    pad_h_size = (h // size + 1) * size - h
    pad_w_size = (w // size + 1) * size - w
    return 0, pad_w_size, 0, pad_h_size


def cutting_and_save(img_tensor: torch.Tensor, size: int, name: str, des_dir: str, is_mask: bool = True) -> None:
    tensor2pil = ToPILImage()
    img_pil = tensor2pil(img_tensor)
    width, height = img_pil.size
    cols = width // size
    rows = height // size
    counter = 0
    for i in range(cols):
        for j in range(rows):
            box = (i * size, j * size, (i + 1) * size, (j + 1) * size)
            grid = img_pil.crop(box)
            if is_mask:
                grid.save(os.path.join(des_dir, f'{name}_{counter}_mask.png'))
            else:
                grid.save(os.path.join(des_dir, f'{name}_{counter}.png'))
            counter += 1


def build_dataset_basic(
        src_dir: str,
        des_dir: str,
        is_mask: bool = True,
        size: int = 512,
        cut: int = 10,
        is_pad: bool = True,
        is_filtrate: bool = True,
        suffix: str = 'png'
) -> None:
    if is_filtrate:
        assert os.path.exists('bad_data.txt'), f'<bad_data.txt> not exist!'
        bad_data = read_txt_as_list('bad_data.txt')
        bcss = [os.path.join(src_dir, file) for file in os.listdir(src_dir)
                if not (os.path.basename(file).split('.')[0] in bad_data)]
    else:
        bcss = [os.path.join(src_dir, file) for file in os.listdir(src_dir)]

    for i, image in enumerate(bcss):
        if suffix == 'png':
            img = load_png_as_tensor(img_path=image, is_mask=is_mask, cut=cut, is_filtrate=is_filtrate)
        elif suffix == 'tif':
            img = load_tif_as_tensor(img_path=image, is_mask=is_mask, cut=cut, is_filtrate=is_filtrate)
        else:
            raise ValueError('Image suffix does not exist!')

        if is_pad:
            padding_size = calc_padding_size(img_tensor=img, size=size)
            img = F.pad(input=img, pad=padding_size, mode='reflect')
        filename = os.path.basename(image)
        filename, _ = os.path.splitext(filename)
        # when run on the server, the name of img is different
        if suffix == 'tif':
            filename = filename.split('_')[1]

        cutting_and_save(img_tensor=img, name=filename, des_dir=des_dir, is_mask=is_mask, size=size)
    if is_mask:
        print(f'{len(bcss)} original masks into {len(os.listdir(des_dir))} patches with size {size}')
    else:
        print(f'{len(bcss)} original images into {len(os.listdir(des_dir))} patches with size {size}')


def data_into_trash(src_dataset_dir: str, src_images_dir: str, src_masks_dir: str, low: int, high: int) -> None:
    src_masks = [os.path.join(src_masks_dir, file) for file in os.listdir(src_masks_dir)]
    des_dir = os.path.join(src_dataset_dir, 'trash')
    os.makedirs(des_dir, exist_ok=True)
    pil2tensor = transforms.ToTensor()
    for idx, src_mask in enumerate(src_masks):
        mask_pil = Image.open(src_mask)
        # mask = pil2tensor(mask_pil)
        mask_arr = np.asarray(mask_pil)
        num_zero_pixels = np.count_nonzero(mask_arr == 0)
        num_one_pixels = np.count_nonzero(mask_arr == 1)
        # num_zero_pixels = torch.sum(mask == 0).item()
        # num_one_pixels = torch.sum(mask == 1).item()
        mask_pil.close()
        if num_zero_pixels <= low or num_zero_pixels >= high or num_one_pixels >= 1:
            mask_name = os.path.basename(src_mask)
            parts = mask_name.split('_')
            img_name = parts[0] + '_' + parts[1] + '.png'
            src_image = os.path.join(src_images_dir, img_name)
            des_img = os.path.join(des_dir, img_name)
            des_mask = os.path.join(des_dir, mask_name)
            shutil.move(src_image, des_img)
            shutil.move(src_mask, des_mask)
    print('After selecting:')
    print(f'selected data -> {len(os.listdir(src_masks_dir))}')
    print(f'trash -> {int(len(os.listdir(des_dir)) / 2)}')


def recover_trash(src_dir: str, des_images_dir: str, des_masks_dir: str) -> None:
    trash_dir = os.path.join(src_dir, 'trash')
    src_datas = os.listdir(trash_dir)
    for idx, src_data in enumerate(src_datas):
        length = len(src_data.split('_'))
        if length == 3:
            shutil.move(os.path.join(trash_dir, src_data), os.path.join(des_masks_dir, src_data))
        else:
            shutil.move(os.path.join(trash_dir, src_data), os.path.join(des_images_dir, src_data))


def read_txt_as_list(file_path: str) -> List:
    with open(file_path, 'r') as file:
        list_str = file.read()
    return eval(list_str)


def save_list_as_txt(des_dir: str, name: str, data_list: List) -> None:
    file_dir = os.path.join(des_dir, name)
    data_str = str(data_list)
    with open(file_dir, 'w') as file:
        file.write(data_str)


def divide_dataset_to_client(src_images_dir: str, src_masks_dir: str, client_num: int = 3, t_num: int = 2) -> None:
    val_id_list = read_txt_as_list('./val/val_id_list.txt')
    src_images = os.listdir(src_images_dir)
    src_id_list = [item.split('.')[0] for item in src_images]
    remain_id_list = [item for item in src_id_list if not (item in val_id_list)]
    data_npc = len(remain_id_list) // (client_num * t_num)

    for i in range(client_num):
        client_dir = f'./client_{i+1}'
        if os.path.exists(client_dir):
            shutil.rmtree(client_dir)
        os.makedirs(client_dir)
        os.makedirs(os.path.join(client_dir, 'model'))
        os.makedirs(os.path.join(client_dir, 'info'))
        for j in range(t_num):
            client_train_dir = os.path.join(client_dir, f'train_{j+1}')
            os.makedirs(client_train_dir)
            client_train_masks_dir = os.path.join(client_train_dir, 'masks')
            os.makedirs(client_train_masks_dir)
            client_train_images_dir = os.path.join(client_train_dir, 'images')
            os.makedirs(client_train_images_dir)
            random.shuffle(src_images)
            for idx in remain_id_list[:data_npc]:
                src_image = os.path.join(src_images_dir, idx+'.png')
                des_image = os.path.join(client_train_images_dir, idx+'.png')
                src_mask = os.path.join(src_masks_dir, idx+'_mask.png')
                des_mask = os.path.join(client_train_masks_dir, idx + '_mask.png')
                shutil.copy(src_image, des_image)
                shutil.copy(src_mask, des_mask)
                remain_id_list.remove(idx)

    print(f'After dividing basic dataset to client:')
    for i in range(client_num):
        print(f'----- Client {i+1} -----')
        for j in range(t_num):
            data_dir = f'./client_{i+1}/train_{j+1}/images'
            id_list = [item.split('.')[0] for item in os.listdir(data_dir)]
            save_list_as_txt(f'./client_{i+1}', f'client_{i+1}_train_{j+1}_id_list.txt', id_list)
            print(f'train_{j+1} -> {len(os.listdir(data_dir))}')

    client_basic_dir = 'client_basic'
    if os.path.exists(client_basic_dir):
        shutil.rmtree(client_basic_dir)
    os.makedirs(client_basic_dir)
    os.makedirs(os.path.join(client_basic_dir, 'model'))
    os.makedirs(os.path.join(client_basic_dir, 'info'))
    print(f'----- Client basic -----')
    for j in range(t_num):
        id_list = []
        client_train_dir = os.path.join(client_basic_dir, f'train_{j + 1}')
        os.makedirs(client_train_dir)
        client_train_masks_dir = os.path.join(client_train_dir, 'masks')
        os.makedirs(client_train_masks_dir)
        client_train_images_dir = os.path.join(client_train_dir, 'images')
        os.makedirs(client_train_images_dir)
        for i in range(client_num):
            id_list.extend(read_txt_as_list(f'./client_{i+1}/client_{i+1}_train_{j+1}_id_list.txt'))
        save_list_as_txt(client_basic_dir, f'client_basic_train_{j + 1}_id_list.txt', id_list)
        for idx in id_list:
            src_image = os.path.join(src_images_dir, idx + '.png')
            des_image = os.path.join(client_train_images_dir, idx + '.png')
            src_mask = os.path.join(src_masks_dir, idx + '_mask.png')
            des_mask = os.path.join(client_train_masks_dir, idx + '_mask.png')
            shutil.copy(src_image, des_image)
            shutil.copy(src_mask, des_mask)
        print(f'train_{j + 1} -> {len(id_list)}')


def divide_dataset_to_val(src_images_dir: str, src_masks_dir: str, factor: float = 0.1, fixed: bool = False) -> None:
    val_dir = './val/val_clean'
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)
    val_masks_dir = os.path.join(val_dir, 'masks')
    val_images_dir = os.path.join(val_dir, 'images')
    os.makedirs(val_images_dir)
    os.makedirs(val_masks_dir)
    if fixed:
        val_id_list = read_txt_as_list('./val_id_list.txt')
        print(f'Validation dataset is fixed.')
        for idx in val_id_list:
            src_image = os.path.join(src_images_dir, idx+'.png')
            des_image = os.path.join(val_images_dir, idx+'.png')
            src_mask = os.path.join(src_masks_dir, idx+'_mask.png')
            des_mask = os.path.join(val_masks_dir, idx+'_mask.png')
            shutil.copy(src_image, des_image)
            shutil.copy(src_mask, des_mask)
        shutil.copy('./val_id_list.txt', './val/val_id_list.txt')
        print(f'Loading val from <val_id_list.txt> finished.')
    else:
        src_images = os.listdir(src_images_dir)
        src_images_num = len(src_images)
        data_nv = int(src_images_num * factor)
        random.shuffle(src_images)
        for file in src_images[:data_nv]:
            src_image = os.path.join(src_images_dir, file)
            des_image = os.path.join(val_images_dir, file)
            src_mask = os.path.join(src_masks_dir, file.split('.')[0]+'_mask.png')
            des_mask = os.path.join(val_masks_dir, file.split('.')[0]+'_mask.png')
            shutil.copy(src_image, des_image)
            shutil.copy(src_mask, des_mask)
        val_data_list = os.listdir(val_images_dir)
        val_id_list = [item.split('.')[0] for item in val_data_list]
        save_list_as_txt('./val', 'val_id_list.txt', val_id_list)
        print(f'Validation dataset is randomly generated.')
    print(f'After dividing basic dataset to validation dataset: \n'
          f'val -> {len(os.listdir(val_images_dir))}')


def get_args():
    parser = argparse.ArgumentParser(
        description='Build the trainset and valset from images and target masks into client'
    )
    parser.add_argument('--suffix', '-su', type=str, default='tif', help='Image Suffix (png or tif)')
    parser.add_argument('--size', '-s', type=int, default=256, help='Size of images and masks')
    parser.add_argument('--target', '-t', type=float, default=0.2, help='Percent of the target region')
    parser.add_argument('--validation', '-v', type=float, default=0.1,
                        help='Percent of the total dataset that is used as validation (0-1)')
    parser.add_argument('--fixed', '-f', type=bool, default=True, help='Whether the validation set is fixed or not')
    parser.add_argument('--padded', '-p', type=bool, default=False, help='Cut with or without padding')
    parser.add_argument('--filtrate', '-ft', type=bool, default=False, help='Manual filtering or not')
    parser.add_argument('--num_client', '-nc', type=int, default=3, help='Number of clients to build')
    parser.add_argument('--num_trainset', '-nt', type=int, default=2, help='Number of trainset per client')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    grid_size = args.size
    area_min = int(args.target * grid_size * grid_size)
    area_max = int((1 - args.target) * grid_size * grid_size)

    dataset_basic_dir = './dataset'
    dataset_basic_images_dir = os.path.join(dataset_basic_dir, 'images')
    dataset_basic_masks_dir = os.path.join(dataset_basic_dir, 'masks')
    if os.path.exists(dataset_basic_dir):
        shutil.rmtree(dataset_basic_dir)
    os.makedirs(dataset_basic_dir, exist_ok=True)
    os.makedirs(dataset_basic_masks_dir, exist_ok=True)
    os.makedirs(dataset_basic_images_dir, exist_ok=True)

    if args.suffix == 'png':
        bcss_images_dir = 'E:/BCSS/rgbs'
        bcss_masks_dir = 'E:/BCSS/masks'
    elif args.suffix == 'tif':
        bcss_images_dir = '/local/scratch/BCSS/BCSS_all/images'
        bcss_masks_dir = '/local/scratch/BCSS/BCSS_all/masks'
    else:
        raise ValueError('Image suffix does not exist!')

    build_dataset_basic(
        src_dir=bcss_images_dir, des_dir=dataset_basic_images_dir,
        is_mask=False, size=grid_size, is_pad=args.padded,
        is_filtrate=args.filtrate, suffix=args.suffix
    )
    build_dataset_basic(
        src_dir=bcss_masks_dir, des_dir=dataset_basic_masks_dir,
        is_mask=True, size=grid_size, is_pad=args.padded,
        is_filtrate=args.filtrate, suffix=args.suffix
    )

    # recover_trash(
    #     src_dir=dataset_basic_dir,
    #     des_images_dir=dataset_basic_images_dir,
    #     des_masks_dir=dataset_basic_masks_dir
    # )

    data_into_trash(
        src_dataset_dir=dataset_basic_dir,
        src_images_dir=dataset_basic_images_dir,
        src_masks_dir=dataset_basic_masks_dir,
        low=area_min,
        high=area_max
    )
    divide_dataset_to_val(
        src_images_dir=dataset_basic_images_dir,
        src_masks_dir=dataset_basic_masks_dir,
        factor=args.validation,
        fixed=args.fixed
    )
    divide_dataset_to_client(
        src_images_dir=dataset_basic_images_dir,
        src_masks_dir=dataset_basic_masks_dir,
        client_num=args.num_client,
        t_num=args.num_trainset
    )


