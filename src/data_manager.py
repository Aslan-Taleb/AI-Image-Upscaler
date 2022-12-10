import pickle
import numpy as np
import tensorflow as tf
import glob
from PIL import Image


import matplotlib.pyplot as plt


def standardize_dataset(x, axis=None):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.sqrt(((x - mean)**2).mean(axis=axis, keepdims=True))
    return (x - mean) / std

def add_gaussian_noise(X, mean=0, std=1):
    """Returns a copy of X with Gaussian noise."""
    return X.copy() + std * np.random.standard_normal(X.shape) + mean

class DataManager:
    def __init__(self):
        self.X = None
        self.Y = None
        self.training_set_size = None
        self.load_data()
        
    def load_data(self):

        # Mise en place de une image , converti en 64,64 et en gris
        imageOrigine = Image.open(r"C:\Users\AslaN\Documents\AI-Image-Upscaler\src\images\000001.jpg")
        imageOrigineRedim = imageOrigine.resize((64,64))
        imageOrigineGris = imageOrigineRedim.convert('L')
        Tableau_imageOR = np.array([np.array(imageOrigineGris)])

        # Mise en place de une image , converti en 32,32 et en gris
        imageBis = Image.open(r"C:\Users\AslaN\Documents\AI-Image-Upscaler\src\images\000001.jpg")
        imageBisRedim = imageBis.resize((32,32))
        imageBisGris = imageBisRedim.convert('L')
        Tableau_imgeBIS = np.array([np.array(imageBisGris)])


        # Mise en place de tout nos image dans une liste
        liste_chemin = glob.glob("C:/Users/AslaN/Documents/AI-Image-Upscaler/src/images/*.jpg")
        np.random.shuffle(liste_chemin)

        for i in liste_chemin :

            image_originale = ((Image.open(i)).resize((64,64))).convert('L')
            Tableau_image_originale = np.array([np.array(image_originale)])
            Tableau_imageOR = np.append(Tableau_image_originale,Tableau_imageOR,axis=0)

            image_changer = ((Image.open(i)).resize((32,32))).convert('L')
            Tableau_image_changer = np.array([np.array(image_changer)])
            Tableau_imgeBIS = np.append(Tableau_image_changer,Tableau_imgeBIS,axis=0)
            

        data = Tableau_imageOR
        data1 = Tableau_imgeBIS

        
        """uint8 -> float32"""
        data = data.astype(np.float32)
        data1 = data1.astype(np.float32)


        """(60000, 28, 28) -> (60000, 28, 28, 1)"""
        data = data.reshape(data.shape[0], 64, 64, 1)
        data1 = data1.reshape(data1.shape[0], 32, 32, 1)
        

        """Standardizes images."""
        data = standardize_dataset(data, axis=(1,2))
        data1 = standardize_dataset(data1, axis=(1,2))

        
        self.X = data
        self.Y = data1
        self.training_set_size = data.shape[0]
        

    def get_batch(self, batch_size, use_noise=False):
        indexes = np.random.randint(self.X.shape[0], size=batch_size)
        if use_noise:
            return self.X[indexes,:], (self.Y[indexes,:])
        return self.X[indexes,:], self.X[indexes,:]

    


