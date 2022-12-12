import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import json, sys

from model import Model
from data import VOCDataset, Compose
from loss import YoloLoss 

def save_checkpoint(model, optimizer, filename = "checkpoint.pth.tar"):
    state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
    print("Saving Checkpoint")
    torch.save(state, filename)

if __name__ == "__main__":
    # Initialize general parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(6969)
    with open("params.json", "r") as FILE:
        params = json.load(FILE)

    print(params)

    # Initialize model & optimizer and so on
    model = Model(classifier_name = "yolo_classifier").to(device)
    optimizer = optim.Adam(model.parameters(), lr = params['lr'])
    #optim.SGD(model.parameters(), momentum = 0.9, weight_decay = 5e-4, lr = 1e-3)
    lossFunction = YoloLoss()

    # Load model from checkpoint if instructed
    if '--new' not in sys.argv:
        print("Loading model from checkpoint.")
        checkpoint = torch.load('checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Generating new model.")

    # Initialize data loader
    trainDataSet = VOCDataset(
            params['training_csv'],
            transform = Compose([
                transforms.Resize((448, 448)), 
                transforms.ToTensor()
                ]),
            img_dir = params['img_dir'],
            label_dir = params['label_dir']
            )

    trainDataLoader = DataLoader(
            dataset = trainDataSet,
            batch_size = params['batch_size'],
            num_workers = params['num_workers'],
            pin_memory = params['pin_memory'],
            shuffle = True,
            drop_last = True
            )

    # Training loop
    for epoch in range(params['epochs']):
        loop = tqdm(trainDataLoader, leave=True)

        for batchIndex, (x, y) in enumerate(loop):
            # run model on batch
            x, y = x.to(device), y.to(device)
            loss = lossFunction(model(x), y)

            # run autograd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_postfix(loss=loss.item(), epoch = epoch)

        if epoch % params['num_epochs_between_checkpoints'] == 0 and epoch:
            save_checkpoint(model, optimizer)

    save_checkpoint(model, optimizer)

