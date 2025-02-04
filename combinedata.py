import pandas as pd
import os
import multiprocessing

# Path to the folder containing the training data
data_folder = os.path.join("training_data") # CHANGE THIS PATH AS NEEDED

# Initialize an empty list to store dataframes
dataframes = []

# Function to process a single CSV file
def process_csv(file):
    """
    Loads an EEG CSV files, then assigns a label based on file name, return dataframe
    """
    file_path = os.path.join(data_folder, file)

    try:
        df = pd.read_csv(file_path) # Load CSV
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None #Skip file
    
    # Extract label from the filename and assign it to a new column
    if "left" in file.lower():
        df['Label'] = 0  # Assign label 0 for "left"
    elif "right" in file.lower():
        df['Label'] = 1  # Assign label 1 for "right"
    else:
        print(f"Warning: No label assigned for file {file}. Skipping...")
        return None  # Skip files without "left" or "right" in the filename
    return df

# Get a list of all CSV files in folder
csv_files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]

#Use multiprocessing to process files in parallel
if __name__ == "__main__":
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        dataframes = pool.map(process_csv,csv_files) #process parallel

# Combine all dataframes into one
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)  # Combine all loaded dataframes
    
    # Save the combined dataframe to a CSV file
    output_file = "combined_training_data.csv"  # CHANGE OUTPUT FILE NAME OR PATH IF NEEDED
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
else:
    print("No valid CSV files found or processed. Exiting...")
