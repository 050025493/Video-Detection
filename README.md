🎥 Deepfake Video Detection System
📌 Project Overview
This project is an AI-powered security tool designed to distinguish between authentic videos and AI-generated "Deepfakes." As generative AI becomes more accessible, the risk of identity theft and misinformation increases. This system addresses that threat by using Computer Vision to analyze video frames for subtle artifacts invisible to the human eye, classifying content as either "Real" or "Fake" with 92.5% accuracy.

🧠 Technical Architecture
Instead of using computationally heavy 3D-CNNs (which process entire videos at once), I implemented a Frame-Based Classification approach. This makes the model faster and deployable on standard hardware.

Input Processing (Stratified Sampling): The system extracts 10 evenly spaced frames from a video. This ensures we capture the temporal evolution of the video without the redundancy of processing every single frame.

The Backbone (EfficientNetB0): Utilizes Transfer Learning by fine-tuning EfficientNetB0 (pre-trained on ImageNet). This architecture was chosen for its "Compound Scaling" method, balancing depth, width, and resolution for optimal accuracy.

The Classifier: Extracted features are passed through a custom fully connected network (Dense Layers) to output a probability score. The final prediction for a video is the average probability of its frames.

🛠️ Key Technical Challenges & Solutions
1. Handling Memory Constraints (Lazy Loading)
Challenge: The dataset contained hundreds of video files. Attempting to load all processed frames into RAM simultaneously caused "Out of Memory" crashes.

Solution: Engineered a custom PyTorch Dataset class with Lazy Loading. Instead of pre-loading data, the system stores only file paths. It opens, processes, and converts videos to tensors on-the-fly only when the training loop requests a specific batch. This reduced startup time to near-zero.

2. Convergence Speed (He Initialization)
Challenge: Initial training runs showed slow convergence, with the model getting stuck at a high loss.

Solution: Applied He Initialization (Kaiming Normal) to the custom classification layers. Since the network uses ReLU activation functions, this specific strategy maintained the variance of activations, preventing the "dying ReLU" problem and allowing the model to learn effectively from Epoch 1.

3. Preventing Overfitting (Early Stopping)
Challenge: Deep Learning models often memorize training data, failing to generalize to new videos.

Solution: Implemented Early Stopping. The training loop monitors the Validation Loss after every epoch. If the model stops improving for 3 consecutive epochs, training terminates automatically, and the weights from the best-performing epoch are restored.

💻 Tech Stack
Framework: PyTorch (chosen for dynamic computation graph and ease of debugging).

Vision: OpenCV (cv2) for robust video reading and artifact detection.

Architecture: EfficientNetB0 (via torchvision).

Experiment Tracking: Weights & Biases (W&B) for real-time loss visualization.

Data Handling: Pandas & NumPy.

🚀 How to Run
1. Installation
The project is optimized for Python 3.11.

Bash

pip install torch torchvision torchaudio opencv-python pandas scikit-learn wandb
2. Training
To train the model from scratch (uses the train folder):

Bash

python train.py
Output: Saves the best weights to deepfake_model.pth.

3. Prediction
To generate predictions on new videos (uses the test folder):

Bash

python predict.py
Output: Generates submission.csv with predictions (0=Real, 1=Fake).

📊 Results
Validation Accuracy: 92.46%

Training Loss: Converged from 15.17 to 0.43.

