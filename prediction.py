from PIL import ImageGrab
import numpy as np
import cv2
from keras.models import load_model

#loading the model
model_test = load_model('mnist_model.h5')

left = 15
top = 180
right = 620
buttom = 800

#You can check that you grab the ms paint screen correctly
#you can set this up by changing top,left,right and buttom params
#Before setting up change while condition to False
'''
screen =  np.array(ImageGrab.grab(bbox=(left,top,right,buttom)))
    
screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
cv2.imshow('window',screen)
'''

while(True):
    
    screen =  np.array(ImageGrab.grab(bbox=(left,top,right,buttom)))
    
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
    screen = cv2.resize(screen,(28,28))
    
    screen = screen.reshape(1,28,28,1)
    pre = model_test.predict(screen, batch_size = 1)
    
    for i in range(len(pre[0])):
        if (pre[0,i] > 0.4):
            print('It\'s {} probably: {}%'.format(i,pre[0,i]*100))
