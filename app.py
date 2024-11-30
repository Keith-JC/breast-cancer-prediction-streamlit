# Import required libraries
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the saved models and objects using pickle
with open('Notebook/gcv_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('Notebook/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('Notebook/selector.pkl', 'rb') as file:
    selector = pickle.load(file)

# Load the breast cancer dataset
data = load_breast_cancer(as_frame=True)
df = data.frame

# Define feature and target columns
X = df.drop(columns=['target'])
y = df['target']

# Model evaluation
X_scaled = scaler.transform(X)  # Apply the scaler to the dataset
X_selected = selector.transform(X_scaled)  # Apply feature selection to the scaled dataset
y_pred_test = model.predict(X_selected)

test_accuracy = accuracy_score(y, y_pred_test)
test_precision = precision_score(y, y_pred_test)
test_recall = recall_score(y, y_pred_test)
test_f1 = f1_score(y, y_pred_test)

# Streamlit App
st.title("Breast Cancer Prediction")

st.write("### Model Performance on Test Data")
st.write(f"Accuracy: {test_accuracy:.2f}")
st.write(f"Precision: {test_precision:.2f}")
st.write(f"Recall: {test_recall:.2f}")
st.write(f"F1 Score: {test_f1:.2f}")
# Sidebar input for new data prediction
st.write("### Input New Data for Prediction")


# Collecting user input
user_inputs = []
input_dict = {}  # Dictionary to store inputs for creating DataFrame
for feature in X.columns:
    min_value = float(X[feature].min())
    max_value = float(X[feature].max())
    mean_value = float(X[feature].mean())

    # Handle constant feature values
    if min_value == max_value:
        user_input = st.sidebar.number_input(f"Enter value for {feature}", value=mean_value)
    else:
        user_input = st.sidebar.slider(f"Select value for {feature}",
                                       min_value=min_value, max_value=max_value, value=mean_value)
    
    # Store the input in the dictionary
    input_dict[feature] = user_input
    user_inputs.append(user_input)


# When the user clicks "Predict"
if st.button('Predict'):
    # Display the user input summary in a table (DataFrame)
    st.subheader("User Input Summary:")

    # Convert the dictionary to DataFrame for better readability
    user_input_df = pd.DataFrame(list(input_dict.items()), columns=["Feature", "User Input"])
    st.dataframe(user_input_df)  # Display the DataFrame as a table

    # Preprocess the user input
    user_inputs = np.array(user_inputs).reshape(1, -1)
    user_inputs_scaled = scaler.transform(user_inputs)
    user_inputs_selected = selector.transform(user_inputs_scaled)

    # Make the prediction
    prediction = model.predict(user_inputs_selected)

    # Display the prediction result
    st.subheader("Prediction Result:")
    if prediction == 0:
        st.write("The predicted class is: Malignant (0)")
    else:
        st.write("The predicted class is: Benign (1)")