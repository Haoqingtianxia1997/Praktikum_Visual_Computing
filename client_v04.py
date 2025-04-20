import argparse
from logging import DEBUG, ERROR, INFO, WARN
from flwr.common.logger import log
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from tqdm import tqdm
import wandb
from evaluate_v01 import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.ewc import compute_fisher_information, save_information
import flwr as fl
from collections import OrderedDict
from utils.ewc import compute_fisher_information, save_information, load_information, ewc_loss
import pickle


def load_data(
        nc: str,
        nt: str,
        base_dir: str,
        batch_size: int = 1,
        img_scale: float = 0.5,
        continual: bool = False,
        reheasal_factor: float = 0.1,
        aug: bool = False,
):
    # 1. Create dataset
    img_t_dir = os.path.join(base_dir, f'./client_{nc}/train_{nt}/images/')
    augimg_t_dir = os.path.join(base_dir, f'./client_{nc}/train_{nt}_aug/images/')
    mask_t_dir = os.path.join(base_dir, f'./client_{nc}/train_{nt}/masks/')
    augmask_t_dir = os.path.join(base_dir, f'./client_{nc}/train_{nt}_aug/masks/')
    img_v_dir = os.path.join(base_dir, './val/val_clean/images/')
    augimg_v_dir = os.path.join(base_dir, './val/val_aug/images/')
    mask_v_dir = os.path.join(base_dir, './val/val_clean/masks/')
    augmask_v_dir = os.path.join(base_dir, './val/val_aug/masks/')

    # 2. Split into train / validation partitions
    if aug:
        train_set = CarvanaDataset(augimg_t_dir, augmask_t_dir, img_scale)
    else:
        train_set = CarvanaDataset(img_t_dir, mask_t_dir, img_scale)
    val_set_cl = CarvanaDataset(img_v_dir, mask_v_dir, img_scale)
    val_set_aug = CarvanaDataset(augimg_v_dir, augmask_v_dir, img_scale)

    n_train = len(train_set)
    n_val = len(val_set_cl)
    num_examples = {"trainset": n_train, "valset": n_val}

    # 3. Create loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader_cl = DataLoader(val_set_cl, shuffle=False, drop_last=True, **loader_args)
    val_loader_aug = DataLoader(val_set_aug, shuffle=False, drop_last=True, **loader_args)

    if continual == 're':
        # take 10% of the 1st trainset (always clean)
        train_set_1st = CarvanaDataset(img_t_dir, mask_t_dir, img_scale)
        train_samples_1st = int(len(train_set_1st) * reheasal_factor)
        train_indices_1st = random.sample(range(len(train_set_1st)), train_samples_1st)
        train_subset_1st = Subset(train_set_1st, train_indices_1st)
        # take 90% of the 2st trainset (clean or aug)
        train_samples_2nd = int(len(train_set) - train_samples_1st)
        train_indices_2nd = random.sample(range(len(train_set)), train_samples_2nd)
        train_subset_2nd = Subset(train_set, train_indices_2nd)
        train_set_combi = ConcatDataset([train_subset_1st, train_subset_2nd])
        train_loader = DataLoader(train_set_combi, shuffle=True, **loader_args)
    elif continual == 'gem':
        # take 10% of the 1st trainset as memory_loader (clean)
        train_set_1st = CarvanaDataset(img_t_dir, mask_t_dir, img_scale)
        train_samples_1st = int(len(train_set_1st) * reheasal_factor)
        train_indices_1st = random.sample(range(len(train_set_1st)), train_samples_1st)
        train_subset_1st = Subset(train_set_1st, train_indices_1st)
        memory_loader = DataLoader(train_subset_1st, shuffle=True, **loader_args)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    return train_loader, val_loader_cl, val_loader_aug, num_examples, memory_loader


def project_gradients(task_grad, mem_grad):
    """将 task_grad 投影到 mem_grad 的空间"""
    mem_flat = torch.cat([g.flatten() for g in mem_grad])
    task_flat = torch.cat([g.flatten() for g in task_grad])
    proj_grad = task_flat - (torch.dot(task_flat, mem_flat) / torch.dot(mem_flat, mem_flat)) * mem_flat
    offset = 0
    proj_grad_list = []
    for g in task_grad:
        proj_grad_list.append(proj_grad[offset:offset + g.numel()].view(g.size()))
        offset += g.numel()
    return proj_grad_list


def train_model(
        model,
        device,
        train_loader,
        val_loader,
        val_loader_aug,
        num_examples,
        optimizer,
        scheduler,
        grad_scaler,
        criterion,
        memory_loader,
        epochs: int = 5,
        batch_size: int = 1,
        amp: bool = True,
        gradient_clipping: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5
):
    n_train = num_examples['trainset']
    global_step = 0
    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        client_info = {
            'loss': 0.,
            'lr': 0.,
            'dice': 0.
        }
        for batch in train_loader:
            images, true_masks = batch['image'], batch['mask']
            assert images.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
                if model.n_classes == 1:
                    loss = alpha * criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += beta * dice_loss((F.sigmoid(masks_pred.squeeze(1)) > 0.5).float(), true_masks.float(), multiclass=False)
                else:
                    loss = alpha * criterion(masks_pred, true_masks)
                    loss += beta * dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)

            if memory_loader:
                # step 1
                task_grad = [param.grad.clone() for param in model.parameters()]
                # step 2
                memory_grads = []
                for mem_batch in memory_loader:
                    mem_images, mem_labels = mem_batch['image'], mem_batch['mask']
                    mem_images = mem_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    mem_labels = mem_labels.to(device=device, dtype=torch.long)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        mem_pred = model(mem_images)
                        if model.n_classes == 1:
                            mem_loss = alpha * criterion(mem_pred.squeeze(1), mem_labels.float())
                            mem_loss += beta * dice_loss((F.sigmoid(mem_pred.squeeze(1)) > 0.5).float(), mem_labels.float(), multiclass=False)
                        else:
                            mem_loss = alpha * criterion(mem_pred, mem_labels)
                            mem_loss += beta * dice_loss(
                                F.softmax(mem_pred, dim=1).float(),
                                F.one_hot(mem_labels, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                    mem_grad = torch.autograd.grad(mem_loss, model.parameters(), retain_graph=True)
                    memory_grads.append(mem_grad)
                # step 3
                for mem_grad in memory_grads:
                    dot_product = sum(torch.dot(t.flatten(), m.flatten()) for t, m in zip(task_grad, mem_grad))
                if dot_product < 0:  # 如果内积为负，则投影
                    task_grad = project_gradients(task_grad, mem_grad)
                # step 4
                with torch.no_grad():
                    for param, g in zip(model.parameters(), task_grad):
                        param.grad = g.clone()

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            global_step += 1
            # Evaluation round
            division_step = (n_train // (2 * batch_size))
            if division_step > 0:
                if global_step % division_step == 0:
                    val_score_cl = evaluate(model, val_loader, device, amp)
                    # val_score_aug = evaluate(model, val_loader_aug, device, amp)
                    scheduler.step(val_score_cl)
                    log(INFO, f'Epoch {epoch}/{epochs} -> Loss (batch): {loss.item()}')
                    log(INFO, f'Epoch {epoch}/{epochs} -> Validation Dice score: {val_score_cl}')
                    client_info['dice'] = val_score_cl.item()
                    # client_info['dice_aug'] = val_score_aug.item()
                    client_info['loss'] = loss.item()
                    client_info['lr'] = optimizer.param_groups[0]['lr']
    return client_info


def train_model_ewc(
        fisher_information, 
        optimal_params, 
        model,
        device,
        train_loader,
        val_loader,
        val_loader_aug,
        num_examples,
        optimizer,
        scheduler,
        grad_scaler,
        criterion,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        ewc_lambda: int = 3,
    ):
    n_train = num_examples['trainset']
    n_val = num_examples['valset']

    global_step = 0

    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        client_info = {
            'loss': 0.,
            'lr': 0.,
            'dice': 0.
        }
        epoch_loss = 0
        # with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
            images, true_masks = batch['image'], batch['mask']
            assert images.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
                if model.n_classes == 1:
                    base_loss = alpha * criterion(masks_pred.squeeze(1), true_masks.float())
                    base_loss += beta * dice_loss((F.sigmoid(masks_pred.squeeze(1)) > 0.5).float(), true_masks.float(), multiclass=False)
                else:
                    base_loss = alpha * criterion(masks_pred, true_masks)
                    base_loss += beta * dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )
                loss = ewc_loss(model, fisher_information, optimal_params, ewc_lambda, base_loss)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            global_step += 1
            epoch_loss += loss.item()

            # Evaluation round
            division_step = (n_train // (2 * batch_size))
            if division_step > 0:
                if global_step % division_step == 0:
                    val_score_cl = evaluate(model, val_loader, device, amp)
                    # val_score_aug = evaluate(model, val_loader_aug, device, amp)
                    scheduler.step(val_score_cl)
                    log(INFO, f'Epoch {epoch}/{epochs} -> Loss (batch): {loss.item()}')
                    log(INFO, f'Epoch {epoch}/{epochs} -> Validation Dice score: {val_score_cl}')
                    client_info['dice'] = val_score_cl.item()
                    # client_info['dice_aug'] = val_score_aug.item()
                    client_info['loss'] = loss.item()
                    client_info['lr'] = optimizer.param_groups[0]['lr']
    return client_info


def test(
        model,
        device,
        val_loader,
        val_loader_aug,
        amp: bool = True,
):
    loss = 0.0
    val_score = evaluate(model, val_loader, device, amp)
    val_score_aug = evaluate(model, val_loader_aug, device, amp)
    log(INFO, 'Validation Dice score: {}'.format(val_score))

    return loss, val_score, val_score_aug


class CustomClient(fl.client.NumPyClient):
    def __init__(
        self, 
        model, 
        train, 
        test, 
        test_aug,
        lr, 
        num_examples, 
        optimizer, 
        scheduler, 
        grad_scaler, 
        criterion,
        save_info,
        save_model,
        continual,
        base_dir,
        memoryloader,
    ) -> None:
        super().__init__()
        self.model = model
        self.train = train
        self.test = test
        self.test_aug = test_aug
        self.learning_rate = lr
        self.num_examples = num_examples
        self.run_id = 0
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.criterion = criterion
        self.save_info = save_info
        self.save_model = save_model
        self.continual = continual
        self.base_dir = base_dir
        self.memoryloader = memoryloader
        self.model_dir = os.path.join(base_dir, f'./client_{args.num_client}/checkpoints')
        self.info_dir = os.path.join(base_dir, f'./client_{args.num_client}/info')

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        log(INFO, "")
        self.run_id = config.get("run_id", 0)
        log(INFO, f'[Round {self.run_id}]')
        updated_learning_rate = config.get("learning_rate", self.learning_rate)
        log(INFO, f"Received updated learning rate: {updated_learning_rate}")
        if self.continual == 'ewc':
            info2server = train_model_ewc(
                fisher_information=fisher_information,
                optimal_params=optimal_params,
                model=self.model,
                epochs=args.epochs,
                device=device,
                amp=args.amp,
                alpha=args.alpha,
                beta=args.beta,
                num_examples=self.num_examples,
                train_loader=self.train,
                val_loader=self.test,
                batch_size=args.batch_size,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                grad_scaler=self.grad_scaler,
                criterion=self.criterion,
                val_loader_aug=self.test_aug,
            )
        else:
            info2server = train_model(
                model=self.model,
                epochs=args.epochs,
                device=device,
                amp=args.amp,
                alpha=args.alpha,
                beta=args.beta,
                num_examples=self.num_examples,
                train_loader=self.train,
                val_loader=self.test,
                batch_size=args.batch_size,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                grad_scaler=self.grad_scaler,
                criterion=self.criterion,
                val_loader_aug=self.test_aug,
                memory_loader=self.memoryloader,
            )
        return self.get_parameters(config={}), self.num_examples["trainset"], info2server

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, dice_score, dice_score_aug = test(
            model=self.model,
            device=device,
            val_loader=self.test,
            val_loader_aug=self.test_aug,
            amp=args.amp,
        )
        if (self.run_id % 10) == 0:
            if self.save_model:
                state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(self.model_dir, f'checkpoint_t{args.num_train}.pth'))
                log(INFO, f"Model's weights saved at round {self.run_id}")
            if self.save_info:
                fisher_information = compute_fisher_information(self.model, self.train, self.criterion, device=device)
                save_information(fisher_information, os.path.join(self.info_dir, f'fisher_t{args.num_train}.pkl'))
                log(INFO, f'Fisher information saved at round {self.run_id}')
                optimal_params = {name: param.clone() for name, param in self.model.named_parameters()}
                save_information(optimal_params, os.path.join(self.info_dir, f'opt_t{args.num_train}.pkl'))
                log(INFO, f'Optimal params saved at round {self.run_id}')
        return float(loss), self.num_examples["valset"], {"Dice Score": float(dice_score), "Dice Score aug": float(dice_score_aug)}


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks -> Client Side')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-6,
                        help='Initial Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data512x512 that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--device', '-d', type=int, default=0, help='Which GPU is used')
    parser.add_argument('--alpha', '-al', type=float, default=1, help='Coefficients of cross-entropy loss')
    parser.add_argument('--beta', '-bt', type=float, default=1, help='Coefficients of dice loss')
    parser.add_argument('--num_client', '-nc', type=str, default='1', help='Train which client (1/2/3)')
    parser.add_argument('--num_train', '-nt', type=str, default='1', help='Use which train set(1/2)')
    parser.add_argument('--save-info', '-si', dest='save_info', type=bool, default=False, help='Whether to save the information? (0/1)')
    parser.add_argument('--save-model', '-sm', dest='save_model', type=bool, default=False, help='Whether to save the model? (0/1)')
    parser.add_argument('--continual', '-ct', type=str, default=False, help='Whether it is continuous training? (ewc/re)')
    parser.add_argument('--aug', '-aug', type=bool, default=False, help='Whether to use augmented data? (0/1)')
    parser.add_argument('--patience', '-pt', type=int, default=10, help='Parameters for adaptive decrease of learning-rate')
    parser.add_argument('--cooldown', '-cd', type=int, default=10, help='Parameters for adaptive decrease of learning-rate')

    return parser.parse_args()
    

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    log(INFO, 
        'Using device: %s',
        device)
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    log(INFO,
        f'Network:\n'
        f'\t{model.n_channels} input channels\n'
        f'\t{model.n_classes} output channels (classes)\n'
        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
        )
    
    # TODO: 需要修改目录到自己的数据集！
    data_base_dir = '/gris/gris-f/homelv/zshi/project/build_dataset/s256_t20_v10_fff'
    
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        log(INFO, f'Model loaded from {args.load}')
        if args.continual == 'ewc':
            fisher_information = load_information(os.path.join(data_base_dir, f'client_{args.num_client}/info/fisher_t1.pkl'))
            log(INFO, f'fisher information loaded.')
            optimal_params = load_information(os.path.join(data_base_dir, f'client_{args.num_client}/info/opt_t1.pkl'))
            log(INFO, f'optimal params loaded.')

    model.to(device=device)

    trainloader, testloader, testloader_aug, num_examples, memoryloader = load_data(
        nc=args.num_client,
        nt=args.num_train,
        batch_size=args.batch_size,
        img_scale=args.scale,
        continual=args.continual,
        reheasal_factor=0.1,
        aug=args.aug,
        base_dir=data_base_dir,
    )

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-8, 
        momentum=0.999, 
        foreach=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', 
        factor=0.8, 
        patience=args.patience, 
        cooldown=args.cooldown
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    client = CustomClient(
        model, 
        trainloader, 
        testloader, 
        testloader_aug,
        args.lr, 
        num_examples,
        optimizer,
        scheduler,
        grad_scaler,
        criterion,
        args.save_info,
        args.save_model,
        args.continual,
        data_base_dir,
        memoryloader,
    ).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)

