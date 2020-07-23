import csv
import cv2
import h5py
from keras.models import load_model


model = load_model("/home/ashre/thesis/speed_model.h5")
image = cv2.imread('/home/ashre/carla_images/IMG/0_204.png')


print("Network prediction: ", model.predict(image[None,:,:,:], batch_size=1))
