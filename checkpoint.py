import torch, os, stat

def save_checkpoint(model, optimizer, scheduler, filename):
    filename = "checkpoints/" + filename + ".pth.tar"
    print(f"Saving model to {filename}")
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    if not os.path.exists('checkpoints'): 
        os.makedirs('checkpoints')
        os.chmod('checkpoints', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    
    torch.save(state, filename)
    os.chmod(filename, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # TODO: Remove chmod commands for deployment. These are specifically for working with the shared folder
    # and are only for purposes of collaboration in development. They may present a security risk if the 
    # function is not used in the intended manner.

def load_checkpoint(model, optimizer, scheduler, filename): 
    filename = "checkpoints/" + filename + ".pth.tar"
    if not os.path.isfile(filename): 
        print("Cannot find checkpoint file! Generating new model...")
        return
    
    print(f"Loading model from {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
