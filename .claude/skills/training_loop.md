
Skill: Training Loop + Checkpointing
Standard pattern for this project:

python
import torch
import wandb
from pathlib import Path

CHECKPOINT_DIR = Path("/blue/smin.fgcu/rkale.fgcu/cdm/checkpoints")

def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def save_checkpoint(model, optimizer, epoch, metrics, config):
    if epoch % 5 == 0:  # Every 5 epochs
        path = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        }, path)
        wandb.save(str(path))

# wandb init at start of every training script
wandb.init(project="conflict-damage-monitor", config=config)
Rules:

Always log to wandb (project: conflict-damage-monitor)

Save checkpoints every 5 epochs to /blue/.../checkpoints/

Always save optimizer state alongside model state

Use F1 score per damage class as primary metric, not just accuracy
