import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

DATASET_PATH = r"SCT_ML_3\cat_vs_dog\dogs-vs-cats-classification\train"

IMG_SIZE = 128
LIMIT = 1000

features = []
labels = []


def extract_hog(image):
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2-Hys"
    )


def load_images(folder_path, label):

    count = 0

    for image_name in os.listdir(folder_path):

        if count >= LIMIT:
            break

        image_path = os.path.join(folder_path, image_name)

        try:

            img = cv2.imread(image_path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

            feature = extract_hog(gray)

            features.append(feature)
            labels.append(label)

            count += 1

        except:
            continue


print("Loading Cat Images...")
load_images(os.path.join(DATASET_PATH, "cats"), 0)

print("Loading Dog Images...")
load_images(os.path.join(DATASET_PATH, "dogs"), 1)

print("Total Images:", len(features))

cat_count = labels.count(0)
dog_count = labels.count(1)

plt.figure(figsize=(5, 4))
plt.bar(["Cats", "Dogs"], [cat_count, dog_count])
plt.title("Dataset Distribution")
plt.ylabel("Number of Images")
plt.show()

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training HOG + SVM Model...")

model = LinearSVC(
    C=1.0,
    max_iter=10000
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/hog_svm_model.pkl")

print("Model Saved Successfully")

cat_folder = os.path.join(DATASET_PATH, "cats")
dog_folder = os.path.join(DATASET_PATH, "dogs")

cat_img = cv2.imread(
    os.path.join(cat_folder, os.listdir(cat_folder)[0])
)

dog_img = cv2.imread(
    os.path.join(dog_folder, os.listdir(dog_folder)[0])
)

cat_img = cv2.cvtColor(cat_img, cv2.COLOR_BGR2RGB)
dog_img = cv2.cvtColor(dog_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(cat_img)
plt.title("Sample Cat Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(dog_img)
plt.title("Sample Dog Image")
plt.axis("off")

plt.show()