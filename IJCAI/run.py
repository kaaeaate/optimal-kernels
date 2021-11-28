import argparse
import logging
from train import train_model

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-device', '--device', type=str, default='cuda',
                        help='Cuda device number')
    parser.add_argument('-model', '--model', type=str, default='U-Net',
                        help='Model name')
    parser.add_argument('-optimizer', '--optimizer', type=str, default='Adam',
                        help='Optimizer name')
    parser.add_argument('-scheduler', '--scheduler', type=str, default='LR',
                        help='Scheduler name')
    parser.add_argument('-exp_name', '--experiment_name', type=str, default='test_run',
                        help='Experiment name')
    
    parser.add_argument('-tb', '--tensorboard', type=bool, default=False,
                        help='Tensorboard option')
    parser.add_argument('-logs', '--logs_dir', type=str, default='logs',
                        help='Logs directory path')
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()
    train_model(model=args.model,
                optimizer=args.optimizer,
                scheduler=args.scheduler,
                experiment_name=args.experiment_name
               )
