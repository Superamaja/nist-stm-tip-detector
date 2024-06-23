import os
import random
import time
from datetime import datetime
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    apply_affine_transform,
    img_to_array,
    load_img,
)

example_dirs = ["exampleTest_test"]
vsplit = 0.2
batch_size = 10
image_px = 75
xforms_per_image = 1
folder_extension = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


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
overall_example_fnames = []
for example_dir in example_dirs:
    example_fnames0 = []
    for file in os.listdir(example_dir):
        if file.endswith(".png"):
            example_fnames0.append(os.path.join(example_dir, file))

    # re-order according to the index specified at the end of each file name: *_[index].png
    # (e.g. example_5.png should correspond to an index of 5)
    # this is important because the .csv file assumes this order
    fname_order = []
    for fname in example_fnames0:
        end = fname.split("_")[-1]
        num_str = end.split(".")[0]
        fname_order.append(int(num_str))

    # example_fnames will hold the correctly ordered set of file names
    example_fnames = [None] * len(example_fnames0)
    old_idx = 0
    for idx in fname_order:
        example_fnames[idx] = example_fnames0[old_idx]
        old_idx += 1
    overall_example_fnames.extend(example_fnames)

img_data = []

for fname in overall_example_fnames:
    img = load_img(fname, target_size=(image_px, image_px), color_mode="grayscale")
    img_data.append(img_to_array(img))  # numpy array


img_data = np.array(img_data)
print(f"data shape: {img_data.shape}")

max = 0
for row in img_data[0]:
    for px in row:
        for val in px:
            if val > max:
                max = val
print("max: " + str(max))


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

max = 0
for row in train_images[0]:
    for px in row:
        for c in px:
            if c > max:
                max = c

print("max: " + str(max))

if xforms_per_image == 1:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=360,
        # width_shift_range = 0.1,
        # height_shift_range = 0.1,
        shear_range=0.15,
        zoom_range=0.05,
        horizontal_flip=True,
    )
else:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.1, zoom_range=0.05
    )


val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow(
    train_images, y=train_labels, batch_size=batch_size
)

validation_generator = val_datagen.flow(
    validation_images, y=validation_labels, batch_size=batch_size
)


# build the convnet
model = Sequential(
    [
        layers.Input(shape=(image_px, image_px, 1)),
        layers.Conv2D(30, 5, activation="relu"),
        layers.Conv2D(40, 5, activation="relu"),
        layers.MaxPooling2D(2, strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)


model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.00005),
    metrics=["acc"],
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    min_delta=0.0001,
)


history = model.fit(
    train_generator,
    # steps_per_epoch=ceil(len(train_labels) / batch_size),
    epochs=500,
    validation_data=validation_generator,
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

# create logs
with open(f"{folder_directory}/logs.txt", "w") as f:
    f.write(
        f"""
        Model Final Accuracy: {history.history["acc"][-1]}
        Model Final Validation Accuracy: {history.history["val_acc"][-1]}
        Model Final Loss: {history.history["loss"][-1]}
        Model Final Validation Loss: {history.history["val_loss"][-1]}
        Loss Metric Type: {model.loss}
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
