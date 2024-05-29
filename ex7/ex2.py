import numpy as np


training_images = np.load('data/training_images.npy')
training_boxes  = np.load('data/training_boxes.npy')
training_labels = np.load('data/training_labels.npy')

print(f'Training images {training_images.shape}')
print(f'Training boxes  {training_boxes.shape}')
print(f'Training labels {training_labels.shape}')