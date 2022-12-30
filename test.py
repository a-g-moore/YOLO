from model import Model
from data import get_ImageNet_dataset
from checkpoint import load_checkpoint
from torch.utils.data.dataloader import DataLoader
import torch, json

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(classifier_name = "imagenet_classifier").to(device)
    load_checkpoint(model, None, None, "imagenet")

    params = {"data_folder": "imagenet-mini/val"}
    dataset = get_ImageNet_dataset(params, augment = False)

    num_samples = 128

    data_loader = DataLoader(
            dataset = dataset,
            batch_size = num_samples,
            num_workers = 8,
            pin_memory = True,
            shuffle = True,
            drop_last = True
            )
        
    x, y = next(iter(data_loader))
    x = x.to(device)
    out = model(x)

    total_correct = 0
    for i in range(num_samples):
        total_correct = total_correct+1 if torch.argmax(out[i]) == y[i] else total_correct

    print(total_correct/num_samples)

if __name__ == "__main__":
    main()