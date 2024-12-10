import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

train_dir = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/DatasetResNet/Train'
val_dir = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/DatasetResNet/Validation'

print('Training directory contents:', os.listdir(train_dir))
print('Validation directory contents:', os.listdir(val_dir))

# Constants
NUM_CLASSES = 5  # Example number of classes in your dataset
CLASS_NAMES = ['schraube1','baugruppe2','toleranz3', 'guss4','vonholzeck5']
BATCH_SIZE = 32  # Example batch size, adjust to your needs
IMG_SIZE = (260, 260)  # EfficientNet default image size for B2 is 260x260 pixels

# Load the EfficientNetB0 model pre-trained on ImageNet data, excluding the top fully connected layer
base_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# This is the new model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with loss function and optimizer
model.compile(optimizer=legacy.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# Prepare your data generators for training and validation sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/DatasetResNet/Train',  # This should be the path to your training data
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/DatasetResNet/Validation',  # This should be the path to your validation data
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Train the model on the new data for a few epochs
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=50
)

# Save the model
model.save('/Users/jadeja19/Documents/Hiwi iktd/IKTD ML PROJECT/EfficientNet.h5')  # Save the model for later use or deployment

# Function to predict class name
def predict_class(model, img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Model expects images to be normalized

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    return predicted_class
