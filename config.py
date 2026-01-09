import os


TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'
TRAIN_CSV = 'train_labels.csv'
TEST_CSV = 'test_public.csv'
MODEL_PATH = 'deepfake_model.pth'  #pytorch model path


IMAGE_SIZE = 224        # Resolution required by EfficientNetB0
FRAMES_PER_VIDEO = 10   # Number of snapshots to analyze per video
BATCH_SIZE = 32          # Number of samples processed before updating weights
EPOCHS = 50           # Number of complete passes through the dataset
LEARNING_RATE = 1e-4    # Step size for the optimizer (0.0001) lr
PATIENCE = 5            # Early stopping patience , we are gonna tell the model if the valuse never decrease after 5 round then stop the training

USE_WANDB = True        # Whether to use Weights & Biases for experiment tracking
WANDB_PROJECT = "Deepfake-Detection"