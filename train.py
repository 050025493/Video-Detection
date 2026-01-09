import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import model
import config
import wandb # Weights & Biases

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Initialize W&B 
if config.USE_WANDB:
    wandb.init(
        project=config.WANDB_PROJECT,
        config={
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LEARNING_RATE,
            "architecture": "EfficientNetB0"
        }
    )

# 3. Prepare Data & Model
train_loader, val_loader = dataset.get_data_loaders()
net = model.build_model().to(device)

# 4. Optimizer & Loss
criterion = nn.BCELoss() # Binary Cross Entropy
optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

# 5. Training Variables
best_val_loss = float('inf')
patience_counter = 0

print("\n--- Starting Training ---")
# train loop
for epoch in range(config.EPOCHS):
    
    net.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.unsqueeze(1) # Fix shape for BCELoss
        
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
   #validtion loop
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Accuracy Calc
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    
    # login wandb
    print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    if config.USE_WANDB:
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_acc": val_acc})

    # preventing overfitting using early stopping
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0 # Reset counter
        torch.save(net.state_dict(), config.MODEL_PATH)
        print("  -> Best model saved!")
    else:
        patience_counter += 1
        print(f"  -> No improvement. Patience: {patience_counter}/{config.PATIENCE}")
        
    # Stop if patience runs out
    if patience_counter >= config.PATIENCE:
        print("\nEarly Stopping triggered. Training finished.")
        break

if config.USE_WANDB:
    wandb.finish()