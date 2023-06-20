# Redes neuronales convolucionales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# PARTE 1 - Construir el modelo CNN

# Importar las librerias
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Iniciar la CNN
classifier = Sequential()

# Paso 1 - Crear la capa de convolucion
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3),
                             input_shape=(64, 64, 3), activation="relu"))

# Paso 2 - Max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Paso extra 1 - Crear una segunda capa de convolucion y max poolin
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Paso 3 - Flattening
classifier.add(Flatten())

# Paso 4 - Full conection
classifier.add(Dense(units = 128, activation="relu"))
classifier.add(Dense(units = 1, activation="sigmoid"))

# Paso 5 - Compilar la CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# PARTE 2 - Ajustar la CNN a las imagenes para entrenar
from keras.preprocessing.image import ImageDataGenerator

# El train datagen sirve para modificar las imagenes aleatoriamente para reducir el sobreajuste
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#Se extraen las imagenes que se necesitan
train_dataset = train_datagen.flow_from_directory('dataset/training_set',
                        # El tamaño de las imagenes debe coincidir con el classifier
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_dataset = train_datagen.flow_from_directory('dataset/test_set',
                        # El tamaño de las imagenes debe coincidir con el classifier
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='binary')

classifier.fit_generator(train_dataset,
                        steps_per_epoch = train_dataset.n//32,
                        epochs=25,
                        validation_data=test_dataset,
                        validation_steps=2000)


# PARTE 3 - Cómo hacer nuevas predicciones
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
train_dataset.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
