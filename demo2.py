import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_and_process_image(image_path, target_size=(64, 64)):
    try:
        image = Image.open(image_path)
        image = image.resize(target_size)
        image_array = np.array(image)
        # Flatten the image into a 1D array
        image_array = image_array.flatten()
        return image_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros(target_size[0] * target_size[1] * 3)

# Load the data
data = pd.read_csv(r"C:\Users\kalam\OneDrive\Documents\DOG.csv")

# Process image paths and load images
image_features = np.array([load_and_process_image(path) for path in data['image_path']])
data.drop(columns=['image_path'], inplace=True)  # Drop the image path column after processing

# Concatenate image features with the remaining data
image_features_df = pd.DataFrame(image_features)
data = pd.concat([data, image_features_df], axis=1)

# Encode the 'GENDER' column if it exists
if 'GENDER' in data.columns:
    le = LabelEncoder()
    data['GENDER'] = le.fit_transform(data['GENDER'])

# Split data into features and target
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=800)

# Initialize and fit the decision tree classifier
clf = DecisionTreeClassifier(max_depth=5)  # Limiting depth to avoid overfitting and improve accuracy
clf = clf.fit(xtrain, ytrain)

# Make predictions on the test set
y_pred = clf.predict(xtest)

#
