import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

image_size = 124

def load_folder(folder, label):
    X, y = [], []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size))
        X.append(img)
        y.append(label)
    return X, y

X_mask, y_mask = load_folder("out/with_mask", 0)
X_nom,  y_nom  = load_folder("out/with_no_mask", 1)

X = np.array(X_mask + X_nom, dtype=np.float32)
y = np.array(y_mask + y_nom, dtype=np.int64)

print("X:", X.shape, "y:", y.shape, "mask:", (y==0).sum(), "no_mask:", (y==1).sum())

# normalize + channel
X = X / 255.0
X = X.reshape(-1, image_size, image_size, 1)
print("X reshaped:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print("Train:", X_train.shape, "Test:", X_test.shape, "Val:", X_val.shape)

# augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# model

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    GlobalAveragePooling2D(),   # <-- Flatten deyil
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=30,
    callbacks=[cb],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

model.save("mask_model_gray.keras")
print("Saved model: mask_model_gray.keras")

