import click
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import json, statistics

from model import Model
from data import get_VOC_dataset, get_ImageNet_dataset
from loss import YoloLoss
from checkpoint import save_checkpoint, load_checkpoint

class WrappedChainScheduler(torch.optim.lr_scheduler.ChainedScheduler):
    """For some reason, PyTorch's ChainedScheduler class does not allow you to pass keyword arguments
    to the schedulers in the list, preventing you from using ReduceLROnPlateau in a chain. This is a 
    simple wrapper class which fixes that issue.
    """

    def __init__(self, schedulers):
        super().__init__(schedulers)
    
    def step(self, metrics = None):
        for scheduler in self._schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics)
            else:
                scheduler.step()
        self._last_lr = [group['lr'] for group in self._schedulers[-1].optimizer.param_groups]

def train(device, model, optimizer, loss_function, data_loader, scheduler, params, model_name):
    for epoch in range(params['epochs']):
        loop = tqdm(data_loader, leave=True)

        losses = []
        for batchIndex, (x, y) in enumerate(loop):
            # run model on batch
            x, y = x.to(device), y.to(device)
            loss = loss_function(model(x), y)

            # run autograd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            losses.append(loss.item())
            loop.set_postfix(loss=statistics.mean(losses), epoch = epoch, lr = scheduler.get_last_lr()[0])

        scheduler.step(metrics = statistics.mean(losses))

        if epoch % params['num_epochs_between_checkpoints'] == 0 and epoch:
            save_checkpoint(model, optimizer, scheduler, model_name)

    save_checkpoint(model, optimizer, scheduler, model_name)

def init(device, model_name, params, features):
    # If instructed, load a feature dectection block from a specified checkpoint
    featureDetector = None
    if features:
        parent_model = Model(classifier_name = f"{features}_classifier")
        load_checkpoint(parent_model, None, None, features)
        featureDetector = parent_model.featureDetector
        
    # Create the appropriate model, using a pre-loaded feature detector if necessary.
    model = Model(classifier_name = f"{model_name}_classifier", featureDetector = featureDetector).to(device)
    
    optimizer = torch.optim.SGD(
            model.parameters(), 
            momentum = params['momentum'],
            weight_decay = params['weight_decay'], 
            lr = params['lr']
            )
    
    if model_name == 'yolo':
        scheduler = WrappedChainScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, end_factor = 1.0, total_iters = 10),
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [75, 105, 135])
            #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        ])

    if model_name == 'imagenet':
        scheduler = WrappedChainScheduler([
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        ])

    loss_function = YoloLoss() if model_name == 'yolo' else torch.nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
    
    dataset = get_VOC_dataset(params, augment=True) if model_name == 'yolo' else get_ImageNet_dataset(params)

    data_loader = DataLoader(
            dataset = dataset,
            batch_size = params['batch_size'],
            num_workers = params['num_workers'],
            pin_memory = params['pin_memory'],
            shuffle = True,
            drop_last = True
            )

    return (model, optimizer, loss_function, data_loader, scheduler)

models = {
        'yolo': 'YOLO Version 1 Object Detector.',
        'imagenet': 'Straightforward image classifier built off the same feature detector.'
        }

@click.command()
@click.option('-m', '--model', 'model_name', type=click.Choice(models.keys()), prompt = 'Enter model name', help = 'The model to train.')
@click.option('-n', '--new', is_flag = True, help = 'Generate new model rather than loading from a checkpoint.')
@click.option('-f', '--features', help = 'Parent model checkpoint to extract a feature detector from.')
@click.option('-s', '--seed', type=int, help = 'Manual seed for deterministic behavior.')
@click.option('-p', '--params', 'param_filename', help = 'Specify a different parameter file.')
def main(model_name, new, features, seed, param_filename): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not seed is None: torch.manual_seed(seed)
    
    param_filename = param_filename if param_filename else "config/" + model_name + ".json"
    with open(param_filename, "r") as FILE: params = json.load(FILE)
    model, optimizer, loss_function, data_loader, scheduler = init(device, model_name, params, features)
    
    if not new: load_checkpoint(model, optimizer, scheduler, model_name)

    train(device, model, optimizer, loss_function, data_loader, scheduler, params, model_name)

if __name__ == "__main__":
    main()
