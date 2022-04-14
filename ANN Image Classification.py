import os
import pathlib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
#from IPython.display import display
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

src_img = Image.open('/home/kingkonzy/Downloads/archive/PetImages/Cat/1.jpg')
#src_img.show()

#im = np.array(src_img)
#print(im.shape)

gray_img = ImageOps.grayscale(src_img)
#display(gray_img)
#gray_img.show()

#g_im = np.array(gray_img)
#print(g_im.shape)

gray_resized_img = gray_img.resize(size=(96, 96))
#gray_resized_img.show()

print(np.ravel(gray_resized_img))

img_final = np.ravel(gray_resized_img) / 255.0
print(img_final)

def process_image(img_path: str) -> np.array:
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img = img.resize(size=(96, 96))
    img = np.ravel(img) / 255.0
    return img

tst_img = process_image(img_path='/home/kingkonzy/Downloads/archive/PetImages/Dog/10012.jpg')
Result = Image.fromarray(np.uint8(tst_img * 255).reshape((96, 96)))
#Result.show()

def process_folder(folder: pathlib.PosixPath) -> pd.DataFrame:
    # We'll store the images here
    processed = []
    
    # For every image in the directory
    for img in folder.iterdir():
        # Ensure JPG
        if img.suffix == '.jpg':
            # Two images failed for whatever reason, so let's just ignore them
            try:
                processed.append(process_image(img_path=str(img)))
            except Exception as _:
                continue
           
    # Convert to pd.DataFrame
    processed = pd.DataFrame(processed)
    # Add a class column - dog or a cat
    processed['class'] = folder.parts[-1]
    
    return processed

train_cat = process_folder(folder=pathlib.Path.cwd().joinpath('data/train/cat'))
train_dog = process_folder(folder=pathlib.Path.cwd().joinpath('data/train/dog'))
train_set = pd.concat([train_cat, train_dog], axis=0)
#test
#hello

