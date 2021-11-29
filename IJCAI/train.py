import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from F import binary_cross_entropy_with_logits as bce_with_logits
from torch.utils.tensorboard import SummaryWriter
from utils.loss import dice_loss, calc_loss, print_metrics
from tqdm.notebook import tqdm
import time
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from collections import defaultdict


def train_model(model, dataloaders, optimizer, scheduler, experiment_name, 
                tensorboard=True, logs_base_dir='logs', 
                save_weights=False, weights_folder='',
                save_metrics=False, metrics_path='metrics_txt/model',
                num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_dice = 0
    
    if tensorboard:
        writer = SummaryWriter(logs_base_dir / experiment_name)

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()
        i=0
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train': 
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.float()
                labels = labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)    

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    if not add_loss_to_kernel:
                        loss = calc_loss(outputs, labels, metrics)
                    else:
                        loss_kernel = 0
                        kernel = model.dconv_down1[0].weight.data
                        ups = nn.Upsample(size=kernel.shape[-1])
                        labels_kernel = ups(labels)
                        labels_kernel = (labels_kernel - labels_kernel.mean()) / labels_kernel.std()
                        for i in range(0, kernel.shape[0]-batch_size_train+1, batch_size_train):
                            loss_kernel += bce_with_logits(kernel[i:i+labels_kernel.shape[0]], 
                                                           labels_kernel)
                        loss_target = calc_loss(outputs, labels, metrics)
                        loss = loss_target + loss_kernel                                

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                epoch_samples += inputs.size(0)
                

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples            
            dice_epoch = metrics['dice'] / epoch_samples
            
            if tensorboard:
                writer.add_scalar(f'Loss_{phase}', epoch_loss.item(), global_step=epoch)
                writer.add_scalar(f'DICE_{phase}', dice_epoch.item(), global_step=epoch)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best loss")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_weights:
                    Path(weights_folder).mkdir(exist_ok=True)
                    torch.save(model.state_dict(), f'./{weights_folder}/{experiment_name}.pth')
            
            if phase == 'val' and dice_epoch > best_dice:
                print("saving best DICE")
                best_dice = dice_epoch

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best val DICE: {:4f}'.format(best_dice))
    
    if save_metrics:
        with open(f"./{metrics_folder}.txt","a") as the_file:
                the_file.write('best DICE: {}\n'.format(best_dice))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    