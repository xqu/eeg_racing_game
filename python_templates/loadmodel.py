import tensorflow as tf
import muselsl
from muselsl import stream, record
import numpy as np
import pandas as pd
import pyautogui
import time
from pylsl import StreamInlet, resolve_stream

# Load the trained model
model = tf.keras.models.load_model("muse_tux_racer_model.h5")

# Function to preprocess live EEG data
def preprocess_eeg_data(eeg_data):
    # Assuming eeg_data is a list or array with raw EEG values from channels
    # Example preprocessing: Convert to numpy array, normalize or reshape if needed
    eeg_data = np.array(eeg_data)
    return eeg_data.reshape(1, -1)  # Reshape for model input

# Function to get live EEG data from Muse
def get_live_eeg_data():
    # Resolving EEG stream (will search for Muse device)
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    # Pulling a sample from the stream (this should be adapted as needed)
    sample, _ = inlet.pull_sample()
    
    # Assuming sample contains EEG channel data; modify to use relevant channels
    # Example: return only a few channels as needed for the model
    return sample[:2]  # Return first two channels, adjust if needed

# Function to control Tux Racer
def control_tux_racer(eeg_data):
    # Preprocess EEG data
    processed_data = preprocess_eeg_data(eeg_data)
    # Make a prediction using the trained model
    prediction = model.predict(processed_data)
    
    # Control Tux Racer
    if prediction > 0.5:
        pyautogui.press('right')  # Simulate pressing right arrow key
    else:
        pyautogui.press('left')   # Simulate pressing left arrow key

# Main script to stream data and control game
def main():
    # Start Muse data stream
    print("Starting Muse data stream...")
    stream()  # Begin streaming data
    time.sleep(2)  # Allow stream to initialize

    print("Starting control loop...")
    try:
        while True:
            eeg_data = get_live_eeg_data()  # Get real-time EEG data
            control_tux_racer(eeg_data)     # Control game based
            time.sleep(0.1)  # Adjust as needed for control speed
    except KeyboardInterrupt:
        print("Exiting control loop.")

# Run the main function
if __name__ == "__main__":
    main()
