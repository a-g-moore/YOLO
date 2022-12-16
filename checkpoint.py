import torch

def save_checkpoint(model, optimizer, scheduler, filename):
    filename = "checkpoints/" + filename + ".pth.tar"
    print(f"Saving model to {filename}")
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    torch.save(state, filename)

def load_checkpoint(model, optimizer, scheduler, filename): 
    filename = "checkpoints/" + filename + ".pth.tar"
    print(f"Loading model from {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
