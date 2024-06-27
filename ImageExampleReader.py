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
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore

from helpers import get_ordered_fnames

example_dirs = ["processed_data"]
vsplit = 0.2
batch_size = 16
xforms_per_image = 1
enable_augmentation = True
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

    split_idx = (
        len(in_list)
        * validation_split
        * xforms_per_image
        / (validation_split * (xforms_per_image - 1) + 1)
    )

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


# Function to apply the required augmentations
def augment_image(image):
    # Convert image to float32 and scale to [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Generate 8 augmented images: 4 rotations and 4 flips
    # TODO: Zoom and brightness
    augmented_images = []
    for i in range(4):
        rotated_image = tf.image.rot90(image, k=i)
        augmented_images.append(rotated_image)
        augmented_images.append(tf.image.flip_left_right(rotated_image))

    return augmented_images if enable_augmentation else [image]


# Convert numpy arrays to tf.data.Dataset
def create_dataset(images, labels, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=len(images))

    dataset = dataset.map(
        lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if training and enable_augmentation:
        dataset = dataset.flat_map(
            lambda x, y: tf.data.Dataset.from_tensor_slices((augment_image(x), [y] * 8))
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


random.shuffle(dull_indexes)
random.shuffle(sharp_indexes)

# Validation split
dull_indexes_v, dull_indexes_t = validation_split_list(
    dull_indexes, validation_split=vsplit
)
sharp_indexes_v, sharp_indexes_t = validation_split_list(
    sharp_indexes, validation_split=vsplit
)

indexes_v = dull_indexes_v + sharp_indexes_v
indexes_t = dull_indexes_t + sharp_indexes_t

# Create the final validation and test arrays
validation_images = np.array([img_data[i] for i in indexes_v])
train_images = np.array([img_data[i] for i in indexes_t])

validation_labels = np.array([0] * len(dull_indexes_v) + [1] * len(sharp_indexes_v))
train_labels = np.array([0] * len(dull_indexes_t) + [1] * len(sharp_indexes_t))

print("validation labels: " + str(validation_labels.shape))
print("validation images: " + str(validation_images.shape))

print("train labels: " + str(train_labels.shape))
print("train images: " + str(train_images.shape))


# Create tf.data.Dataset instances
train_dataset = create_dataset(train_images, train_labels, batch_size, training=True)
validation_dataset = create_dataset(
    validation_images, validation_labels, batch_size, training=False
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
    patience=25,
    restore_best_weights=True,
    min_delta=0.0001,
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=500,
    verbose=1,
    callbacks=[early_stopping],
    steps_per_epoch=(
        (len(train_labels) * 8 // batch_size)
        if enable_augmentation
        else len(train_labels) // batch_size
    ),
    validation_steps=len(validation_labels) // batch_size,
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
Augmentation Setting: {xforms_per_image}
Batch Size: {batch_size}
        
--Training Info--
Epochs: {len(history.history["acc"])}

Steps per Epoch: {(len(train_labels) * 8 // batch_size) if enable_augmentation else len(train_labels) // batch_size}
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
