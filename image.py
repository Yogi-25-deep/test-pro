
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global spatial average pooling layer
x = keras.layers.GlobalAveragePooling2D()(base_model.output)

# Add a few fully connected layers
x = keras.layers.Dense(256, activation='relu')(x)
predictions = keras.layers.Dense(2, activation='softmax')(x)

# Create the final model
model = keras.models.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'diary_products/',
    batch_size=32,
    class_mode='sparse',
    shuffle=True)

test_generator = test_datagen.flow_from_directory(
    'diary_products/',
    batch_size=32,
    class_mode='sparse',
    shuffle=True)
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)
model.evaluate(test_generator)
