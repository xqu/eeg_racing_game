import subprocess
import time

#run files together
def run_script(script):
    """
    Runs a Python script using subprocess.
    
    Parameters:
        script (str): The Python script filename (e.g., 'combinedata.py').
    """
    print(f"Running {script}...")
    start_time = time.time()

    try:
        subprocess.run(["python",script], check=True)
        print(f"{script}completed in {time.time() - start_time:.2f} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
        exit(f"Stopping pipeline. {script} failed")

#main pipeline
if __name__ == "__main__":
    """
    Runs the EEG processing pipeline:
        1. Combine EEG data
        2. Preprocess EEG data
        3. Train the EEG model
    """
    run_script("combinedata.py")
    run_script("preprocess_eeg.py")
    run_script("trainmodel.py")
    print("Pipeline completed successfully")