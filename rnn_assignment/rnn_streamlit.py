import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model in HDF5 format
model = load_model('rnn_lstm_model.h5')  # Update to your saved model path

# Function to prepare input data for prediction
def prepare_input(data, n_steps=3):
    data = np.array(data).reshape((1, n_steps, 1))  # Reshape for the model
    return data

# Streamlit app layout
st.title("LSTM Time Series Prediction")


st.write('original_data = [110, 125, 133, 146, 158, 172, 187, 196, 210] ')

# User input for the last three days
input_data = st.text_input("Enter the last three values separated by commas:", "187, 196, 210")

if st.button("Predict Next 10 values"):
    # Process input
    input_values = [int(x) for x in input_data.split(',')]
    predictions = []

    for _ in range(10):
        input_array = prepare_input(input_values)
        prediction = model.predict(input_array)
        predictions.append(int(prediction[0][0]))  # Append the predicted value
        input_values.append(prediction[0][0])  # Update input for next prediction
        input_values = input_values[1:]  # Keep only the last three values


    st.write("Predicted values for the next 10 days: ", ", ".join(map(str, predictions)))
    # Plotting both original data and predictions
    # Original time series data (first 9 days)
    original_data = [110, 125, 133, 146, 158, 172, 187, 196, 210]  # Update to your original data
    all_data = original_data + predictions

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the original time series data
    plt.plot(range(1, 10), original_data, label="Original Time Series", marker='o', color='blue', linewidth=2)

    # Plot the next 10 days' predicted data
    plt.plot(range(10, 20), predictions, label="Next 10-Day Predictions", marker='o', color='red', linestyle='--', linewidth=2)

    # Add titles and labels
    plt.title("Combined Plot of Original and Predicted Time Series", fontsize=16)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Values", fontsize=12)

    # Add grid for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add legend for clarity
    plt.legend()

    # Show the plot in Streamlit
    st.pyplot(plt)

