import numpy as np
import torch, os
import time
from pylsl import StreamInlet, resolve_stream  # For real-time EEG data streaming
from trainmodel import EEGTransformer  # Import the trained EEG Transformer model
from collections import deque
from sklearn.model_selection import train_test_split  # For splitting training data
from pynput.keyboard import Controller, Key  # For simulating keyboard button presses

# Load preprocessed EEG data
segments = np.load(os.path.join("training_data\preprocessed\eeg_segments.npy"))  # CHANGE THIS PATH
labels = np.load(os.path.join("training_data\preprocessed\eeg_labels.npy"))

#for normalization
mean = segments.mean(axis=(0, 1))
std = segments.std(axis=(0, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Initialize the model
input_dim = 5  # Number of input features (e.g., EEG channels)
num_labels = 2  # Number of output classes (e.g., Left and Right)
model = EEGTransformer(input_dim=input_dim, num_labels=num_labels)

# Load pretrained model checkpoint
checkpoint_path = os.path.join("eeg_racing_game_trained_model", "eeg_transformer_model") # CHANGE THIS PATH
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
# set model for evaluation
model.to(device)
model.eval()

# Initialize keyboard controller
keyboard = Controller()

# Real-time EEG stream handling
print("Looking for an EEG stream...")
try:
    streams = resolve_stream('type', 'EEG')  # Resolve EEG stream
    inlet = StreamInlet(streams[0])  # Connect to the first available EEG stream
    print("EEG stream found. Starting real-time classification.")
except Exception as e:
    print(f"Error: could not find EEG stream. {e}")
    exit()

#define buffer
sequence_length = 256 #samples per sequence
buffer = deque(maxlen=sequence_length)

# Define a smoothing function for incoming EEG samples
def moving_average(data, window_size=5):
    """
    Applies a moving average to smooth incoming EEG data.

    Parameters:
        data: 1D numpy array of EEG signal data.
        window_size: int, the number of samples to average over.

    Returns:
        Smoothed EEG data as a 1D numpy array.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Real-time prediction loop
try:
    print("Starting real-time EEG classification...")
    with torch.no_grad(): #Disable gradient calculation for inference
        while True:

            start_time = time.time()

            # Get a new EEG sample
            sample, _ = inlet.pull_sample()  # Retrieve an EEG sample from the stream
            if sample is None:
                print("No new EEG sample recieved")
                continue
            sample = np.array(sample)  # Convert sample to numpy array

            # Smooth the sample using a moving average
            smoothed_sample = moving_average(sample, window_size=5)

            # Normalize the sample using training data statistics
            normalized_sample = (smoothed_sample - mean) / std

            # Add the normalized sample to the buffer
            buffer.append(normalized_sample)

            # Make a prediction if the buffer has enough data/full
            if len(buffer) == sequence_length:
                sequence = np.array(buffer)  # Convert buffer to numpy array
                data_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # Convert to PyTorch tensor

                with torch.cuda.amp.autocast():
                    outputs = model(data_tensor)  # Forward pass through the model
                    probabilities = torch.softmax(outputs, dim=1)  # Compute probabilities
                    print(f"Class Probabilities: {probabilities}")
                    
                    # Apply class weights
                    #class_weights = torch.tensor([1.0, 1.7]).to(device)  # Example weights: higher weight for class 1
                    #weighted_probabilities = probabilities * class_weights  # Scale probabilities by class weights
                    prediction = torch.argmax(probabilities, dim=1).item()  # Predict based on weighted probabilities

                # Log predictions for debugging
                print(f"Raw outputs: {outputs}")
                print(f"Probabilities: {probabilities}")
                print(f"Predicted class: {prediction}")

                # Simulate key presses based on prediction
                if prediction == 0:  # Class 0 corresponds to "Left"
                    action = "Left"
                else:  # Class 1 corresponds to "Right"
                    action = "Right"

                if action == "Left":
                    keyboard.press("a")
                    keyboard.release("a")
                elif action == "Right":
                    keyboard.press("d")
                    keyboard.release("d")

                print(f"Prediction: {action} | Probabilities: {probabilities.cpu().numpy()}")

except KeyboardInterrupt:
    print("Prediction stopped.")
except Exception as e:
    print(f"Error in prediction loop: {e}")

