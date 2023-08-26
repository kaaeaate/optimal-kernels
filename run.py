import argparse
import logging
from pathlib import Path
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from train import train_model
from utils.common import get_model, get_dataloader


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-data_name', '--dataset_name', type=str, default='',
                        help='Dataset name')
    parser.add_argument('-data_path', '--dataset_path', type=str, default='',
                        help='Dataset directory path')
    
    parser.add_argument('-device', '--device', type=str, default='cuda',
                        help='Cuda device number')
    parser.add_argument('-model', '--model', type=str, default='unet',
                        help='Model name')
    parser.add_argument('-optimizer', '--optimizer', type=str, default='Adam',
                        help='Optimizer name')
    parser.add_argument('-scheduler', '--scheduler', type=str, default='LR',
                        help='Scheduler name')
    parser.add_argument('-exp_name', '--experiment_name', type=str, default='test_run',
                        help='Experiment name')
    parser.add_argument('-batch', '--batch_size', type=int, default=4,
                        help='Train batch size')
    parser.add_argument('-epochs', '--num_epochs', type=int, default=50,
                        help='Train epochs number')
    
    
    parser.add_argument('-tb', '--tensorboard', type=bool, default=False,
                        help='Tensorboard option')
    parser.add_argument('-logs', '--logs_dir', type=str, default='logs',
                        help='Logs directory path')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataloaders = get_dataloader(args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    exp_name = f'{args.experiment_name}_' + datetime.now().isoformat(timespec='minutes')
    
    model = train_model(model=args.model, 
                        dataloaders=dataloaders, 
                        optimizer=optimizer, 
                        scheduler=exp_lr_scheduler, 
                        experiment_name=exp_name, 
                        device=device,
                        tensorboard=args.tensorboard, 
                        logs_base_dir=args.logs_dir, 
                        save_weights=True, weights_path=f'./weights/{exp_name}',
                        save_metrics=True, metrics_path=f'./metrics_txt/{file_name}',
                        add_loss_to_kernel=True,
                        num_epochs=args.num_epochs
                       )
