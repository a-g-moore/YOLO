import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Yolo
from data import VOCDataset, Compose
from loss import YoloLoss 

torch.manual_seed(6969)

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

def train(dataLoader, model, optimizer, lossFunction):
    loop = tqdm(dataLoader, leave=True)

    for batchIndex, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = lossFunction(model(x), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

def saveCheckpoint(model, optimizer, filename = "checkpoint.pth.tar"):
    state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
    print("Saving Checkpoint")
    torch.save(state, filename)

def main():
    model = Yolo().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    #optim.SGD(model.parameters(), momentum = 0.9, weight_decay = 5e-4, lr = 1e-3)
    lossFunction = YoloLoss()

    load_model = False
    
    if load_model:
        checkpoint = torch.load('checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])



    trainDataSet = VOCDataset(
            "data/8examples.csv",
            transform = Compose([
                transforms.Resize((448, 448)), 
                transforms.ToTensor()
                ]),
            img_dir = IMG_DIR,
            label_dir = LABEL_DIR
            )
    
    trainDataLoader = DataLoader(
            dataset = trainDataSet,
            batch_size = BATCH_SIZE,
            num_workers = NUM_WORKERS,
            pin_memory = PIN_MEMORY,
            shuffle = True,
            drop_last = False
            )

    schedule = torch.linspace(1e-3, 1e-2, 10).tolist() + [1e-2]*65 + [1e-3]*30 + [1e-4]*30
    #print(schedule)

    for epoch in range(EPOCHS):
        train(trainDataLoader, model, optimizer, lossFunction)
        #optimizer.param_groups[0]['lr'] = schedule[epoch]

        if epoch % 10 == 0:
            saveCheckpoint(model, optimizer)

    saveCheckpoint(model, optimizer)
    return model

if __name__ == "__main__":
    main()
