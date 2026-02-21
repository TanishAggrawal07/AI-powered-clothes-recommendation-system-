import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

tf.get_logger().setLevel('ERROR')  # To suppress warnings

# ─────────────────────────────────────────────────────────
# 1. GPU Info
# ─────────────────────────────────────────────────────────
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# ─────────────────────────────────────────────────────────
# 2. Paths & Hyperparameters
# ─────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTR_CSV = os.path.join(BASE_DIR, 'dataset', 'list_attr_celeba.csv')
PART_CSV = os.path.join(BASE_DIR, 'dataset', 'list_eval_partition.csv')
BASE_IMG_DIR = os.path.join(BASE_DIR, 'dataset', 'img_align_celeba', 'img_align_celeba')

IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 7
PATIENCE = 3
MODEL_PATH = "best_beard_model.h5"

# ─────────────────────────────────────────────────────────
# 3. Load & Process Labels
# ─────────────────────────────────────────────────────────

attr_df = pd.read_csv(ATTR_CSV)[['image_id', 'No_Beard']]
attr_df['label'] = attr_df['No_Beard'].map({1: 0, -1: 1})
part_df = pd.read_csv(PART_CSV)
df = attr_df.merge(part_df, on='image_id')
df['split'] = df['partition'].map({0: 'train', 1: 'val', 2: 'test'})

for split in ['train', 'val', 'test']:
    df.loc[df['split'] == split, 'image_id'] = df[df['split'] == split]['image_id'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

print(f"Train/Val/Test sizes: {len(train_df)}, {len(val_df)}, {len(test_df)}")

# Class weights
y_train = train_df['label'].values
weights = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}
print("Class weights:", class_weights)

# ─────────────────────────────────────────────────────────
# 4. TF Dataset Generator
# ─────────────────────────────────────────────────────────

def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

def create_dataset(dataframe, shuffle=True):
    paths = [os.path.join(BASE_IMG_DIR, fname) for fname in dataframe['image_id'].values]
    labels = dataframe['label'].values.astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_df)
val_dataset = create_dataset(val_df, shuffle=False)
test_dataset = create_dataset(test_df, shuffle=False)

# ─────────────────────────────────────────────────────────
# 5. Build & Compile Model
# ─────────────────────────────────────────────────────────

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ─────────────────────────────────────────────────────────
# 6. Callbacks
# ─────────────────────────────────────────────────────────

checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)

# ─────────────────────────────────────────────────────────
# 7. Train Model
# ─────────────────────────────────────────────────────────

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    class_weight=class_weights,
    callbacks=[checkpoint, earlystop]
)

# ─────────────────────────────────────────────────────────
# 8. Evaluate
# ─────────────────────────────────────────────────────────

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

