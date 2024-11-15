from sklearn.metrics import plot_confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import layers 
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory, plot_model
from tensorflow.keras.models import Sequential
from pathlib import Path

np.random.seed(0)
tf.random.set_seed(0)




img_height = 150
img_width = 150

def getCNNData(dataset = "train",subset = "training",validation_split=0.2):
        dset = "seg_"+dataset
        rootPath = Path.cwd()
        data_dir = rootPath / "archive" / dset /dset
        ds = image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset=subset,
        seed=123,
        labels='inferred',
        image_size=(150, 150),
        batch_size=32)
        # AUTOTUNE = tf.data.AUTOTUNE
        # ds = ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        return ds

class doCNN:
    img_height = 150
    img_width = 150
    batch_size = 32 

    def __init__(self) -> None:
        pass

    def doCNN(trainds,valds,activation='relu',epochs =10,dropout=0.25):
        """
        build model, compile model
        """
        num_classes = len(trainds.class_names)
        data_augmentation = data_augmentation = Sequential([layers.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),layers.RandomRotation(0.1),layers.RandomZoom(0.1),])

  

        model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation=activation),
        #layers.BatchNormalization(),
        layers.MaxPooling2D(),
        #layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation=activation),
        #layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation=activation),
        #layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,padding='same',activation=activation),
        #layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(256,3,padding='same',activation=activation),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(dropout),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        #layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(128,activation=activation),
        #layers.BatchNormalization(),
        #layers.Dropout(dropout),
        layers.Dense(64, activation=activation),
        #layers.BatchNormalization(),
        #layers.Dropout(dropout),
        layers.Dense(32,activation=activation),
        #layers.BatchNormalization(),
        #layers.Dropout(dropout),
        layers.Dense(32,activation=activation),
        #layers.BatchNormalization(),
        #layers.Dropout(dropout),
        layers.Dense(16,activation=activation),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(num_classes,activation= "softmax")
        ])
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        history = model.fit(trainds,validation_data =valds,epochs=epochs)

        return history,model

    def plots(history,epochs):
        #TODO: accuracy and loss plots 
        """
        Plot accuracy, and loss plots
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig("acc_valPlots.png")
        plt.show()
    
    def testCNN(testData,model):
        results = model.evaluate(testData)
        preds = model.predict(testData)        
        y_pred = []  # store predicted labels
        y_true = []  # store true labels

        # iterate over the dataset
        for image_batch, label_batch in testData:   # use dataset.unbatch() with repeat
        # append true labels
            y_true.append(label_batch)
            # compute predictions
            preds = model.predict(image_batch)
            # append predicted labels
            y_pred.append(np.argmax(preds, axis = - 1))

# convert the true and predicted labels into tensors
        correct_labels = tf.concat([item for item in y_true], axis = 0)
        predicted_labels = tf.concat([item for item in y_pred], axis = 0)
        confusionMat = tf.math.confusion_matrix(correct_labels,predicted_labels,num_classes=6)

        return confusionMat



        




    #def doCNN(ds):



if __name__ == "__main__":
    epochs = 50
    train_ds = getCNNData()
    val_ds = getCNNData(dataset="train",subset="validation")
    history,model  = doCNN.doCNN(train_ds,val_ds,epochs=epochs)
    plot_model(model,to_file='cnn_model.png',show_shapes = True,show_layer_activations=True)
    doCNN.plots(history=history,epochs=epochs)
    testData = getCNNData("test",None,None)
    confusion = testCNN(testData,model)

    


