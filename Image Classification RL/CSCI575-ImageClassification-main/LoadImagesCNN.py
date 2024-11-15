import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    directory='archive/seg_train/seg_train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(150,150,3))