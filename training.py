import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()



#Plotting some pictures from the dataset(optional), you can remove them if you want

#--------------
plt.figure(figsize = (10,10))
x, y = 1, 1

for i in range(x*y):
    plt.subplot(x, y, i+1)
    plt.imshow(x_train[i])
plt.show()

#---------------

batch_size = 128
num_classes = 10
epochs = 15


img_rows, img_cols = 28, 28
input_shape = 0
#Reshaping x values

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#Converting binary matrix according to num_classes
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Creating model

model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3 ), 
                 activation='relu', 
                 input_shape = input_shape))


model.add(Conv2D(64, kernel_size = (3, 3), 
                 activation = 'relu', 
                 input_shape = input_shape))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.50))

model.add(Dense(num_classes, activation = 'softmax'))

#model.summary()

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])


#If you want to train the model again, just change it to True
training = False

if training:

    model.fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epochs,
              verbose = 1,
              validation_data=(x_test, y_test))
    
    model.save('mnist_model.h5')


    score = model.evaluate(x_test, y_test, verbose = 1)
    print('Test Loss = ', score[0])
    print('Test Score = ', score[1])