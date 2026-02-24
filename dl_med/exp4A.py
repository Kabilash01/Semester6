import kagglehub
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
base = Path(path)
print("Dataset downloaded to:", base)

def find_dataset_root_with_images_folders(base_path: Path) -> Path:
    for root in [base_path] + [p for p in base_path.rglob("*") if p.is_dir()]:
        images_dirs = list(root.glob("*/images"))
        if len(images_dirs) >= 2:  
            return root
    raise RuntimeError("Could not locate a dataset root containing multiple '<class>/images' folders.")

dataset_root = find_dataset_root_with_images_folders(base)
print("Detected dataset root:", dataset_root)
class_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir() and (p / "images").is_dir()])
if not class_dirs:
    raise RuntimeError(
    )

print("Classes found:", [p.name for p in class_dirs])
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
class_names = [p.name for p in class_dirs]
class_to_idx = {name: i for i, name in enumerate(class_names)}

file_paths = []
labels = []

for cls in class_dirs:
    img_dir = cls / "images"
    for ext in (".png", ".jpg", "*.jpeg"):
        for fp in img_dir.glob(ext):
            file_paths.append(str(fp))
            labels.append(class_to_idx[cls.name])

print("Total images:", len(file_paths))
n = len(file_paths)
idx = tf.random.shuffle(tf.range(n), seed=SEED)
file_paths = tf.gather(file_paths, idx).numpy().tolist()
labels = tf.gather(labels, idx).numpy().tolist()

test_size = int(0.2 * n)
val_size = int(0.2 * (n - test_size))

test_paths, test_labels = file_paths[:test_size], labels[:test_size]
val_paths, val_labels = file_paths[test_size:test_size + val_size], labels[test_size:test_size + val_size]
train_paths, train_labels = file_paths[test_size + val_size:], labels[test_size + val_size:]

print("Train/Val/Test:", len(train_paths), len(val_paths), len(test_paths))

def load_and_preprocess(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  
    return img, label

AUTOTUNE = tf.data.AUTOTUNE

def make_ds(paths, labels, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(2000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds
train_ds = make_ds(train_paths, train_labels, training=True)
val_ds = make_ds(val_paths, val_labels, training=False)
test_ds = make_ds(test_paths, test_labels, training=False)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(train_ds, validation_data=val_ds, epochs=5)
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc*100:.2f}%")
import numpy as np
from sklearn.metrics import classification_report
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)
print(classification_report(y_true, y_pred, target_names=class_names))
import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.legend()
plt.title("Accuracy")
plt.show()
