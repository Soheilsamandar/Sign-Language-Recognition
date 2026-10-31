from pathlib import Path
import tensorflow as tf
import numpy as np

# =====================================
# Configuration
# =====================================

DATA_DIR = Path(r"E:\Sign Language Recognition\Data")
OUTPUT_DIR = Path(r"E:\Sign Language Recognition\output")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

# =====================================
# Check dataset
# =====================================

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

class_names = sorted([folder.name for folder in DATA_DIR.iterdir() if folder.is_dir()])

if len(class_names) < 2:
    raise ValueError("At least two class folders are required.")

print("Classes:", class_names)

# =====================================
# Load dataset
# =====================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
)

# =====================================
# Optimize dataset
# =====================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =====================================
# Data augmentation
# =====================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ]
)

# =====================================
# Build CNN model
# =====================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(224, 224, 3)),
        augmentation,
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ]
)

# =====================================
# Compile model
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# =====================================
# Callbacks
# =====================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(OUTPUT_DIR / "keras_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
)

# =====================================
# Train model
# =====================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping],
)

# =====================================
# Save labels
# =====================================

labels_path = OUTPUT_DIR / "labels.txt"

with open(labels_path, "w", encoding="utf-8") as file:
    for index, label in enumerate(class_names):
        file.write(f"{index} {label}\n")

# =====================================
# Evaluate best model
# =====================================

best_model = tf.keras.models.load_model(OUTPUT_DIR / "keras_model.h5")

loss, accuracy = best_model.evaluate(val_ds)

print("\nTraining completed!")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

print("\nSaved files:")
print(OUTPUT_DIR / "keras_model.h5")
print(OUTPUT_DIR / "labels.txt")
