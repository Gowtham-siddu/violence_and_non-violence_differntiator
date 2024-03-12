from google.colab import drive
drive.mount('/content/drive')

# Import libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import cv2
import numpy as np

# Define paths (replace with your actual paths)
training_data_path = "/content/data/train"
validation_data_path = "/content/data/validation"
test_data_path = "/content/data/test"

# Function to extract frames at regular intervals
def extract_frames(video_path, frame_rate=1):
  cap = cv2.VideoCapture(video_path)
  frames = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_rate == 0:
      frames.append(frame)
  cap.release()
  return frames

train_frames = []
for video_path in glob.glob(training_data_path + "/*.mp4"):
  train_frames.extend(extract_frames(video_path))

val_frames = []
for video_path in glob.glob(validation_data_path + "/*.mp4"):
  val_frames.  extend(extract_frames(video_path))

datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


train_datagen = datagen.flow_from_directory(
    training_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    save_to_dir='/content/train_frames',
    save_prefix='augmented'
)

# Validation data generator without augmentation
val_datagen = datagen.flow_from_directory(
    validation_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Test data generator (assuming labels are available)
test_datagen = datagen.flow_from_directory(
    test_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Maintain order for evaluation
)

# Normalize pixel values
train_datagen.mean = np.array([123.68, 116.779, 103.939], dtype="float32")
val_datagen.mean = train_datagen.mean
test_datagen.mean = train_datagen.mean

# Pre-trained model selection
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze pre-trained layers
for layer in base_model.layers:
  layer.trainable = False

# Add classifier head
x = base_model.output
x = Flatten()(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Model compilation
model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
learning_rate_scheduler = LearningRateScheduler(lambda epoch: 0.001 * (0.9 ** epoch))

# Train the model
model.fit(
    train_datagen,
    epochs=10,
    validation_data=val_datagen,
    callbacks=[early_stopping, learning_rate_scheduler]
)

#
