# import pickle 
# from app import model_pkl_file, probability_model

# # load model from pickle file
# with open(model_pkl_file, 'rb') as file:  
#     model = pickle.load(file)

# # evaluate model
# from imp import load_module

import tensorflow as tf 
import keras
from tenserflow.keras.models import load_model
from app import probability_model


savedModel=load_model("model_mnist.h5")
print(savedModel)


