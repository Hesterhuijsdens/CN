import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt


# function to generate the data:
def generate_data(data):
    if data == "pixels":
        print("pixels")
    elif data == "natural":

        # load and transform into gray scale:
        I = rgb2gray(img.imread('sampleMerry_0011_Lasalle.jpg'))

        # extract image patches:
        N = (np.shape(I)[0] * np.shape(I)[1]) / 64 # N = 64?
        x = np.zeros((int(N), 8, 8))
        patch_nr = 0
        for i in range(1, int((np.shape(I)[0] / 8) + 1)):
            for j in range(1, int((np.shape(I)[1] / 8) + 1)):
                x[patch_nr, :, :] = I[(i-1)*8:i*8, (j-1)*8:j*8]
                patch_nr += 1
        x = x / 64.0
    else: # data == gratings
        print("gratings")

    # reshape:
    X = np.zeros((int(N), 64))
    for k in range(0, int(N)):
        X[k, :] = np.reshape(np.squeeze(x[k, :, :]), (1, 64))
    return X


# function to transform color scale into gray scale:
def rgb2gray(image):
    return np.dot(image[..., :3], [0.2989, 0.587, 0.114])





