import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras import layers

DATA_PATH = "./Data/data.json"


with open(DATA_PATH, "r") as fp:
    data = json.load(fp)
    
features = np.array(data["mfcc"])
labels = np.array(data["labels"])

X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.2)

model = keras.Sequential([
    layers.Flatten(input_shape=(features.shape[1],features.shape[2])),
    layers.Dense(512, "relu"),
    layers.Dense(512, "relu"),
    layers.Dense(512, "relu"),
    layers.Dense(10, "softmax")
])

model.compile(optimizer = "Adam", 
                loss = "sparse_categorical_crossentropy",
                metrics = ["accuracy"])

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="./checkpoints/norm_nn/checkpoint.ckpt",
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True)


history = model.fit(X_train, y_train, 32, 50,
validation_data=(X_test,y_test),
callbacks=[model_checkpoint_callback])






