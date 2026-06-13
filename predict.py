import cv2
import joblib
import numpy as np

from skimage.feature import hog

IMG_SIZE = 128

model = joblib.load("model/hog_svm_model.pkl")


def extract_hog(image):
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2-Hys"
    )


image_path = input("Enter Image Path: ")

img = cv2.imread(image_path)

if img is None:
    print("Invalid Image Path")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

feature = extract_hog(img)

feature = np.array(feature).reshape(1, -1)

prediction = model.predict(feature)

if prediction[0] == 0:
    print("Prediction: Cat")
else:
    print("Prediction: Dog")