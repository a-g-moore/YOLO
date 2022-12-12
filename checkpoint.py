import torch

def save_checkpoint(model, optimizer, filename):
    filename += ".pth.tar"
    print(f"Saving model to {filename}")
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename): 
    filename += ".pth.tar"
    print(f"Loading model from {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
