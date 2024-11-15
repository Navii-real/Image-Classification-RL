import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import CSVLogger

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    df_train = pd.read_csv("training.csv")
    df_val = pd.read_csv("validation.csv")

    # Create the ImageDataGenerator object
    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    # Generate batches and augment the images
    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        directory='seg_train/seg_train/',
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        color_mode='rgb',
    )

    val_generator = train_datagen.flow_from_dataframe(
        df_val,
        directory='seg_test/seg_test/',
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        color_mode='rgb',
    )

    # Initialize the Pretrained Model
    feature_extractor = ResNet50(weights='imagenet',
                                 input_shape=(224, 224, 3),
                                 include_top=False) # 175 layers

    # Set this parameter to make sure it's not being trained
    feature_extractor.trainable = False

    # Set the input layer
    input_ = tf.keras.Input(shape=(224, 224, 3))

    # Set the feature extractor layer
    x = feature_extractor(input_, training=False)
    print("Number of layers in the base model: ", len(feature_extractor.layers))

    # Set the pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Set the final layer with sigmoid activation function
    output_ = tf.keras.layers.Dense(6, activation='softmax')(x)

    # Create the new model object
    model = tf.keras.Model(input_, output_)

    # Compile it
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=['accuracy'])

    # Print The Summary of The Model
    model.summary()

    csv_logger = CSVLogger('training_log1.csv', append=True)
    history = model.fit(train_generator, epochs=15, validation_data=val_generator, callbacks=[csv_logger])


    #####################
    #    Fine Tuning    #
    #####################
    feature_extractor.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 150

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in feature_extractor.layers[:fine_tune_at]:
      layer.trainable = False

    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=['accuracy'])
    model.summary()

    fine_tune_epochs = 10
    total_epochs = 15 + fine_tune_epochs

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history_fine = model.fit(train_generator,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=val_generator,
                             callbacks=[csv_logger, cp_callback])

    model.save("resnet50_tuned")

if __name__ == "__main__":
    main()


