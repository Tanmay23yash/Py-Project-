from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Initialize Flask application
app = Flask(__name__)

# Load and preprocess the dataset
def load_and_preprocess_data():
    dataset = pd.read_csv('dataset.csv')

    # One-hot encode categorical variables
    dataset = pd.get_dummies(dataset, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

    # Scale numerical features
    standardScaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

    return dataset, standardScaler

dataset, standardScaler = load_and_preprocess_data()

# Prepare data for training
y = dataset['target']
X = dataset.drop(['target'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Train the K Neighbors Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=8)
knn_classifier.fit(X_train, y_train)

# Preprocess input data for prediction
def preprocess_input(input_data):
    # Convert input to DataFrame (assuming input is a dictionary)
    df = pd.DataFrame([input_data])

    # Apply dummy variable transformation
    df = pd.get_dummies(df)

    # Add missing dummy columns to match the training data
    for col in X.columns:
        if col not in df.columns:
            df[col] = 0

    # Scale the numerical features
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[columns_to_scale] = standardScaler.transform(df[columns_to_scale])

    # Ensure correct order of columns
    df = df.reindex(columns=X.columns, fill_value=0)

    return df

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    input_data = request.json

    # Preprocess the input
    processed_data = preprocess_input(input_data)

    # Make a prediction
    prediction = knn_classifier.predict(processed_data)

    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
