from PIL import Image
import pandas as pd
import numpy as np

def load_data(csv_path, base_path):
    images = []
    labels = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        full_path = f"{base_path}/{row['Path']}"
        img = Image.open(full_path).resize((32, 32))
        images.append(np.array(img))
        labels.append(row["ClassId"])
    return np.array(images), np.array(labels)

def get_data():
    X_train, y_train = load_data("data/Train.csv", "data")
    X_test, y_test = load_data("data/Test.csv", "data")
    X_train = X_train / 255.0
    X_test = X_test /  255.0
    return X_train, y_train, X_test, y_test