import click
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import json, sys

from model import Model
from data import VOCDataset, Compose
from loss import YoloLoss
from checkpoint import *

def train(device, model, optimizer, loss_function, data_loader, params, model_name):
    for epoch in range(params['epochs']):
        loop = tqdm(data_loader, leave=True)

        for batchIndex, (x, y) in enumerate(loop):
            # run model on batch
            x, y = x.to(device), y.to(device)
            loss = loss_function(model(x), y)

            # run autograd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_postfix(loss=loss.item(), epoch = epoch)

        if epoch % params['num_epochs_between_checkpoints'] == 0 and epoch:
            save_checkpoint(model, optimizer, model_name)

    save_checkpoint(model, optimizer, model_name)

def init(device, model_name, params):
    model = Model(classifier_name = f"{model_name}_classifier").to(device)
    optimizer = optim.Adam(model.parameters(), lr = params['lr'])
    #optim.SGD(model.parameters(), momentum = 0.9, weight_decay = 5e-4, lr = 1e-3)
    loss_function = YoloLoss() if model_name == 'yolo' else nn.CrossEntropyLoss()

    dataset = VOCDataset(
            params['training_csv'],
            transform = Compose([
                transforms.Resize((448, 448)), 
                transforms.ToTensor()
                ]),
            img_dir = params['img_dir'],
            label_dir = params['label_dir']
            )

    data_loader = DataLoader(
            dataset = dataset,
            batch_size = params['batch_size'],
            num_workers = params['num_workers'],
            pin_memory = params['pin_memory'],
            shuffle = True,
            drop_last = True
            )

    return (model, optimizer, loss_function, data_loader)

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
    
    param_filename = param_filename + ".json" if param_filename else model_name + ".json"
    with open(param_filename, "r") as FILE: params = json.load(FILE)
    model, optimizer, loss_function, data_loader = init(device, model_name, params)
    if not new: load_checkpoint(model, optimizer, model_name)
    train(device, model, optimizer, loss_function, data_loader, params, model_name)

if __name__ == "__main__":
    main()
