# data_prediction.py

# Import the required libraries and modules
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import keras
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sagemaker import Session, FeatureStore
from sagemaker.feature_store.feature_group import FeatureGroup

# Define the Sagemaker session and the feature store
session = Session()
feature_store = FeatureStore(session=session)

# Define the feature group for the fused data
fused_feature_group = FeatureGroup(name="fused-feature-group", sagemaker_session=session)

# Define a function to load the fused data from the feature store
def load_data():
    # Load the feature group from the feature store as a pandas dataframe
    df = fused_feature_group.as_dataframe()
    # Return the dataframe
    return df

# Define a function to split the data into train and test sets
def split_data(df, test_size):
    # Shuffle the data
    df = df.sample(frac=1)
    # Split the data into features and labels
    X = df.drop("label", axis=1)
    y = df["label"]
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    # Return the train and test sets
    return X_train, X_test, y_train, y_test

# Define a function to build and train the multi-modal neural network model
def build_and_train_model(X_train, y_train, epochs, batch_size, validation_split):
    # Define the input and output dimensions
    input_dim = X_train.shape[1]
    output_dim = 1 # binary classification
    # Define the multi-modal neural network model
    model = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_dim, activation="sigmoid")
    ])
    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Train the model on the data
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    # Return the model
    return model

# Define a function to evaluate the model on the test data
def evaluate_model(model, X_test, y_test):
    # Predict the labels for the test data
    y_pred = model.predict(X_test)
    # Round the predictions to get the binary labels
    y_pred = np.round(y_pred)
    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = auc(fpr, tpr)
    # Return the evaluation metrics
    return accuracy, precision, recall, f1, fpr, tpr, auc

# Define a function to visualize the results and the charts
def visualize_results(y_test, y_pred, fpr, tpr, auc):
    # Create a confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    # Plot the confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    # Plot the ROC curve using matplotlib
    plt.plot(fpr, tpr, label="AUC = {:.2f}".format(auc))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    # Plot the distribution of the predictions using plotly
    fig = px.histogram(x=y_pred, nbins=2, labels={"x": "Prediction"}, title="Distribution of Predictions")
    fig.show()

# Load the fused data from the feature store
df = load_data()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

# Build and train the multi-modal neural network model
model = build_and_train_model(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
accuracy, precision, recall, f1, fpr, tpr, auc = evaluate_model(model, X_test, y_test)

# Visualize the results and the charts
visualize_results(y_test, y_pred, fpr, tpr, auc)
