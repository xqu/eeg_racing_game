import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a bandpass filter to the EEG signal.
    Parameters:
        data: 1D numpy array of EEG signal data (single channel).
        lowcut: Lower cutoff frequency of the filter (Hz).
        highcut: Upper cutoff frequency of the filter (Hz).
        fs: Sampling frequency of the EEG signal (Hz).
        order: The order of the Butterworth filter.
    Returns:
        Filtered EEG signal as a 1D numpy array.
    """
    nyquist = 0.5 * fs  # Nyquist frequency (half the sampling frequency)
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')  # Create filter coefficients
    return lfilter(b, a, data)  # Apply the filter

# Parallelized bandpass filter function
def parallel_bandpass_filter(data, lowcut, highcut, fs, workers = 4):
    """
    Applies bandpass filter in parallel to all EEG channels.
    Parameters:
        data: 2D numpy array (time points x channels).
        lowcut, highcut, fs: Filter parameters.
        workers: Number of parallel processes.
    Returns:
        2D numpy array of filtered EEG signals.
    """
    with multiprocessing.Pool(processes=workers) as pool:
        filtered_data = pool.starmap(
            bandpass_filter, [(data[:, i], lowcut, highcut, fs) for i in range(data.shape[1])]
        )
    return np.column_stack(filtered_data)

# Segment function
def segment_data(data, labels, window_size, step_size):
    """
    Segments EEG data into fixed-size windows with overlapping.
    Parameters:
        data: 2D numpy array (time points x channels).
        labels: 1D numpy array of labels corresponding to each time point.
        window_size: Number of time points in each segment.
        step_size: Number of time points to slide the window forward.
    Returns:
        segments: 3D numpy array (num_segments x window_size x num_channels).
        segment_labels: 1D numpy array of labels for each segment.
    """
    segments = []
    segment_labels = []
    
    # Parallel segmentation using threadpoolexectuer
    def process_segment(i):
        return data[i : i + window_size], labels[i]
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_segment, range(0, len(data) - window_size, step_size))

    #unpack
    for segment, label in results:
        segments.append(segment)
        segment_labels.append(label)
    return np.array(segments), np.array(segment_labels)

# Main preprocessing function
def preprocess_combined_data(input_file, save_folder, lowcut=1, highcut=50, fs=256, window_size=256, step_size=128):
    """
    Preprocess EEG data from a combined CSV file and save it as NumPy arrays.
    Parameters:
        input_file: Path to the combined CSV file containing raw EEG data and labels.
        save_folder: Folder where the preprocessed data will be saved.
        lowcut: Lower cutoff frequency for the bandpass filter (Hz).
        highcut: Upper cutoff frequency for the bandpass filter (Hz).
        fs: Sampling frequency of the EEG signal (Hz).
        window_size: Number of time points in each segment.
        step_size: Step size for the sliding window (overlap = window_size - step_size).
    """
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists
    
    # Load the combined data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)  # Assumes data is stored in a CSV format
    
    # Separate features and labels
    labels = df['Label'].values  # Ensure your CSV contains a 'Label' column
    data = df.drop(columns=['Label']).values  # Extract EEG channels (all columns except 'Label')
    
    # Apply bandpass filter to each channel
    print("Applying bandpass filter...")
    filtered_data = parallel_bandpass_filter(data, lowcut, highcut, fs, workers=multiprocessing.cpu_count())
    
    # Normalize each channel (z-score normalization)
    print("Normalizing data...")
    normalized_data = (filtered_data - np.mean(filtered_data, axis=0)) / np.std(filtered_data, axis=0)
    
    # Segment data
    print("Segmenting data...")
    segments, segment_labels = segment_data(normalized_data, labels, window_size, step_size)
    
    # Save preprocessed data
    print(f"Saving preprocessed data to {save_folder}...")
    np.save(os.path.join(save_folder, "eeg_segments.npy"), segments)
    np.save(os.path.join(save_folder, "eeg_labels.npy"), segment_labels)
    print(f"Preprocessed data saved to {save_folder}")

# Run the preprocessing script
if __name__ == "__main__":
    # Define input and output paths
    input_file = os.path.join("combined_training_data.csv")
    save_folder = os.path.join("training_data", "preprocessed")
    
    # Preprocess EEG data
    preprocess_combined_data(input_file, save_folder)
