import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

train_dir = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/DatasetResNet/Train'
val_dir = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/DatasetResNet/Validation'

print('Training directory contents:', os.listdir(train_dir))
print('Validation directory contents:', os.listdir(val_dir))


#constants

Num_classes = 5
Batch_size = 32
Img_size = (224, 224)

#Define classes

class_name = ['class1', 'class2','class3','class4','class5']

#load Resnet50 model pre-trained on Image-Net data

base_model = ResNet50(weights='imagenet',include_top = False, input_shape=(Img_size + (3,)))

#freeze the layer of the base_model

for layer in base_model.layers:
  layer.trainable = False

# add custom layer on top of base layer

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation= 'relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(Num_classes, activation='softmax')(x)

# this is the new model we will train

model = Model(inputs = base_model.input, outputs = predictions)

# compile the loss function with loss function and optimizer

model.compile(optimizer=legacy.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#prepare datagenerators for training and validation sets
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/DatasetResNet/Train",
    target_size = Img_size,
    batch_size = Batch_size,
    class_mode = 'categorical'
)

validation_generator = test_datagen.flow_from_directory(
    "/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/DatasetResNet/Validation",
    target_size = Img_size,
    batch_size = Batch_size,
    class_mode = 'categorical'
)

# train the model on new dataset for 10 epochs

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples// Batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples// Batch_size,
    epochs = 10

)
#save the model

model.save("/Users/jadeja19/Documents/Hiwi iktd/IKTD ML PROJECT/EfficentNet.h5")


