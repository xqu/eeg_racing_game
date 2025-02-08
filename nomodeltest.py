from pylsl import StreamInlet, resolve_streams
from pynput.keyboard import Key, Controller
import time
import numpy as np
from collections import deque

# Find an EEG stream on LSL
print("Looking for an EEG stream...")
streams = resolve_streams()
inlet = StreamInlet(streams[0])

keyboard = Controller()

# Calibration parameters
CALIBRATION_TIME = 5  # Collect baseline data for 5 seconds
WINDOW_SIZE = 100  # Number of past samples to track for threshold updates

# Collect baseline EEG data
print("Calibrating... Please remain still and relaxed.")
baseline_values = []
start_time = time.time()

while time.time() - start_time < CALIBRATION_TIME:
    sample, _ = inlet.pull_sample()
    baseline_values.append(sample[0])
    time.sleep(0.05)  # Small delay to avoid excessive sampling

# Calculate initial dynamic thresholds
mean_baseline = np.mean(baseline_values)
std_baseline = np.std(baseline_values)

LEFT_THRESHOLD = mean_baseline + 1.5 * std_baseline
RIGHT_THRESHOLD = mean_baseline - 1.5 * std_baseline

print(f"Calibration complete: LEFT_THRESHOLD={LEFT_THRESHOLD:.3f}, RIGHT_THRESHOLD={RIGHT_THRESHOLD:.3f}")

# Rolling buffer to update thresholds dynamically
buffer = deque(maxlen=WINDOW_SIZE)

while True:
    sample, timestamp = inlet.pull_sample()
    buffer.append(sample[0])

    # Periodically update thresholds
    if len(buffer) == WINDOW_SIZE:
        mean_dynamic = np.mean(buffer)
        std_dynamic = np.std(buffer)
        LEFT_THRESHOLD = mean_dynamic + 1.5 * std_dynamic
        RIGHT_THRESHOLD = mean_dynamic - 1.5 * std_dynamic

    # Assume sample[0] contains the data needed for left/right movement
    if sample[0] > LEFT_THRESHOLD:
        keyboard.press(Key.left)
        time.sleep(0.1)
        keyboard.release(Key.left)

    elif sample[0] < RIGHT_THRESHOLD:
        keyboard.press(Key.right)
        time.sleep(0.1)
        keyboard.release(Key.right)
