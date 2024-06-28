import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.preprocessing.image import (  # type: ignore
    ImageDataGenerator,
    apply_affine_transform,
    img_to_array,
    load_img,
)

from helpers import get_ordered_fnames

# Parameters
example_dirs = ["processed_data"]
vsplit = 0.2
batch_size = 16
xforms_per_image = 1
patience = 25

folder_extension = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


# Load config file
with open("config.json") as f:
    config = json.load(f)

SQUARE_NM_SIZE = config["SQUARE_NM_SIZE"]
SQUARE_PIXEL_SIZE = config["SQUARE_PIXEL_SIZE"]


# Allow for user notes to be added to logs
user_notes = input("User Notes: ")
if user_notes:
    user_notes = "\n".join(user_notes.split(r"\n"))

# Get the iteration number for the model training
if os.path.exists("models/iteration.txt"):
    with open("models/iteration.txt", "r") as f:
        iteration = int(f.read())
        print("Iteration: ", iteration)
else:
    iteration = 1
    print("Iteration: ", iteration)


def validation_split_list(in_list, validation_split=0):
    v_list = []
    t_list = []

    split_idx = int(len(in_list) * validation_split)

    print("validation split idx: " + str(split_idx))

    for i in range(len(in_list)):
        if (i + 1) <= split_idx:
            v_list.append(in_list[i])
        else:
            t_list.append(in_list[i])

    return v_list, t_list


# determine the file names of all of the examples
overall_fnames = []
for example_dir in example_dirs:
    fnames = get_ordered_fnames(example_dir)
    overall_fnames.extend(fnames)

img_data = []

for fname in overall_fnames:
    img = load_img(
        fname,
        target_size=(SQUARE_PIXEL_SIZE, SQUARE_PIXEL_SIZE),
        color_mode="grayscale",
    )
    img_data.append(img_to_array(img))  # numpy array


img_data = np.array(img_data)
print(f"data shape: {img_data.shape}")

# read in the features/labels which are ordered the same as img_data
features = pd.read_csv(os.path.join(example_dirs[0], "features.csv"), sep=",")
# merge the features from the remaining example directories
for example_dir in example_dirs[1:]:
    features = pd.concat(
        [features, pd.read_csv(os.path.join(example_dir, "features.csv"), sep=",")],
        ignore_index=True,
    )

db_features = features.loc[features["defectType"] == "DB"]
print("size: " + str(db_features.size))
db_features = db_features.loc[db_features["sampleBias"] > 0]
print("size: " + str(db_features.size))
db_dull_features = db_features.loc[db_features["tipQuality"] == "dull"]
db_sharp_features = db_features.loc[db_features["tipQuality"] == "sharp"]

db_dull_indexes = db_dull_features.index
db_sharp_indexes = db_sharp_features.index

dull_indexes = db_dull_indexes.tolist()
sharp_indexes = db_sharp_indexes.tolist()

# --------------------------------------------------------------------------------------------------------
# "standard" ML processing starts here
# "standard" ML processing starts here
random.shuffle(dull_indexes)
random.shuffle(sharp_indexes)


# validation split
dull_indexes_v, dull_indexes_t = validation_split_list(
    dull_indexes, validation_split=vsplit
)
sharp_indexes_v, sharp_indexes_t = validation_split_list(
    sharp_indexes, validation_split=vsplit
)

indexes_v = dull_indexes_v + sharp_indexes_v
indexes_t = dull_indexes_t + sharp_indexes_t


# create the final validation and test arrays
validation_images = []
for i in indexes_v:
    validation_images.append(img_data[i])

validation_images = np.array(validation_images)

# training images can be manually augmented here
train_images = []
for i in indexes_t:
    # 8 transforms per image (xforms_per_image):
    # 4x 90 degree rotations (incl. 0 deg), and horizontal flips of each

    img = img_data[i]
    train_images.append(img)

    if xforms_per_image == 8:
        flip_img = apply_affine_transform(img, zx=-1)
        train_images.append(flip_img)

        img = apply_affine_transform(img, theta=90)
        train_images.append(img)
        flip_img = apply_affine_transform(img, zx=-1)
        train_images.append(flip_img)

        img = apply_affine_transform(img, theta=90)
        train_images.append(img)
        flip_img = apply_affine_transform(img, zx=-1)
        train_images.append(flip_img)

        img = apply_affine_transform(img, theta=90)
        train_images.append(img)
        flip_img = apply_affine_transform(img, zx=-1)
        train_images.append(flip_img)

train_images = np.array(train_images)


validation_labels = np.array([0] * len(dull_indexes_v) + [1] * len(sharp_indexes_v))
train_labels = np.array(
    [0] * len(dull_indexes_t) * xforms_per_image
    + [1] * len(sharp_indexes_t) * xforms_per_image
)

print("validation labels: " + str(validation_labels.shape))
print("validation images: " + str(validation_images.shape))

print("train labels: " + str(train_labels.shape))
print("train images: " + str(train_images.shape))


if xforms_per_image == 1:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=360,
        shear_range=0.15,
        zoom_range=0.05,
        horizontal_flip=True,
    )
else:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=360,
        shear_range=0.15,
        zoom_range=0.05,
        horizontal_flip=True,
    )


val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow(
    train_images, y=train_labels, batch_size=batch_size
)

validation_generator = val_datagen.flow(
    validation_images, y=validation_labels, batch_size=batch_size
)

# Build the convnet
model = Sequential(
    [
        layers.Input(shape=(SQUARE_PIXEL_SIZE, SQUARE_PIXEL_SIZE, 1)),
        layers.Conv2D(30, 5, activation="relu"),
        layers.Conv2D(40, 5, activation="relu"),
        layers.MaxPooling2D((2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["acc"],
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=patience,
    restore_best_weights=True,
    min_delta=0.0001,
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=500,
    verbose=1,
    callbacks=[early_stopping],
)


# Save the iteration number for the model training
if not os.path.exists("models"):
    os.makedirs("models")
with open("models/iteration.txt", "w") as f:
    f.write(str(iteration + 1))
# Create the iteration folder
folder_directory = f"models/{iteration} - {folder_extension}"
if not os.path.exists(folder_directory):
    os.makedirs(folder_directory)


# save the model
model.save(f"{folder_directory}/model.keras")
model.save(f"{folder_directory}/model.h5")

# create logs
with open(f"{folder_directory}/logs.txt", "w") as f:
    f.write(
        f"""
--User Notes--
{user_notes}
        
--Config Info--
Square NM Size: {SQUARE_NM_SIZE}
Pixel Size: {SQUARE_PIXEL_SIZE}
Augmentation: {xforms_per_image}
Batch Size: {batch_size}
Patience: {patience}
        
--Training Info--
Epochs: {len(history.history["acc"])}

Steps per Epoch: {len(train_labels) * xforms_per_image // batch_size}
Validation Steps: {len(validation_labels) // batch_size}

--Model Info--
Model Best/Restored Validation Loss: {history.history["val_loss"][np.argmin(history.history["val_loss"])]}

Model Final Training Accuracy: {history.history["acc"][-1]}
Model Final Validation Accuracy: {history.history["val_acc"][-1]}
Model Final Training Loss: {history.history["loss"][-1]}
Model Final Validation Loss: {history.history["val_loss"][-1]}
    """
    )


# list of accuracy results on training and validation for each epoch
acc = history.history["acc"]
val_acc = history.history["val_acc"]

# list of loss results
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_it = range(len(acc))

# plot training and validation accuracy
plt.figure()
plt.plot(epochs_it, acc)
plt.plot(epochs_it, val_acc)
plt.title("Training and validation accuracy")
plt.savefig(f"{folder_directory}/accuracy.png")

# plot training and validation loss
plt.figure()
plt.plot(epochs_it, loss)
plt.plot(epochs_it, val_loss)
plt.title("Training and validation loss")
plt.savefig(f"{folder_directory}/loss.png")


plt.show()
