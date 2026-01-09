import torch
import torch.nn as nn
from torchvision import models

def initialize_weights(m):
    """
    He Initialization (Kaiming Normal) for ReLU layers.
    This helps the model converge faster.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def build_model():
    # 1. Load Pre-trained EfficientNetB0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # 2. Freeze the base layers (Transfer Learning)
    for param in model.features.parameters():
        param.requires_grad = False
        
    # 3. Replace the Classifier Head
    # Input size for B0 classifier is 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 512),
        nn.ReLU(),
        
        nn.Dropout(p=0.3),
        nn.Linear(512, 1),
        nn.Sigmoid() # Output probability 0-1
    )
    
    # 4. Apply He Initialization to the new layers
    model.classifier.apply(initialize_weights)
    
    return model