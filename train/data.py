import pandas as pd
import os

train_df = pd.read_csv('train_labels.csv')
test_df = pd.read_csv('test_public.csv')
print("Training Data:")
print(train_df.head())
print("\nTesting Data:")
print(test_df.head())

#total videos
print(f"\nTotal training videos: {len(train_df)}")
print(f"Total test videos: {len(test_df)}")

#class distribution in training set
print("\nClass distribution:")
print(train_df.iloc[:, 1].value_counts())
