import os
import numpy as np
from PIL import Image

path = "/home/julius/Project/digit_clasifier/MNIST Dataset JPG format/MNIST - JPG - training/"
images = []
awnsers = []


for i in range(10):
    for f in os.listdir(path+str(i)):
        print(f, i)
        image = Image.open(path+str(i)+'/'+f)
        images.append(np.array(image))
        awnsers.append(i)

def int_to_arr(n):
    arr = np.zeros(10)
    arr[n] = 1.0
    return arr

from nn import NeuralNet 

#m.reshape(-1)
nn = NeuralNet(28*28, 10, 10) 

while True:
    for i in range(len(images)):
        print("The awnser is:", awnsers[i])
        print("The network guessed:")
        print(nn.backprop(images[i].reshape(-1), int_to_arr(awnsers[i])))
        print()
        print("--------------------------------")
        print("--------------------------------")
        print()
        print()
