import torch
import pandas as pd
import numpy as np
import os
import dataset
import model
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Model Architecture & Weights
print(f"Loading best model from {config.MODEL_PATH}...")
net = model.build_model()
net.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
net.to(device)
net.eval() 

# 2. Load Test List
test_df = pd.read_csv(config.TEST_CSV)
results = []

print(f"Starting inference on {len(test_df)} videos...")

with torch.no_grad():
    for index, row in test_df.iterrows():
        video_name = row['filename']
        full_path = os.path.join(config.TEST_FOLDER, video_name)
        
        # A. Get Frames (Now returns a pre-processed Tensor)
        video_tensor = dataset.load_video_frames(full_path)
        
        # Check if video was empty/broken
        # If shape is [10, 3, 224, 224], it's good. 
       
        if video_tensor.sum() == 0 and video_tensor.shape[0] > 0:
            print(f"Warning: Video {video_name} is empty or broken.")
            pass

        # B. Send directly to GPU (No extra transforms needed!)
        batch_input = video_tensor.to(device)
        
        # C. Predict
        outputs = net(batch_input)
        
        # D. Aggregate
        avg_prob = torch.mean(outputs).item()
        pred_label = 1 if avg_prob > 0.5 else 0
        
        results.append([video_name, pred_label, avg_prob])

# 3. Save
submission = pd.DataFrame(results, columns=['filename', 'Prediction', 'Probability'])  # column names can be changed like gilename to video_name
submission.to_csv('submission.csv', index=False)
print("Success! 'submission.csv' generated.")