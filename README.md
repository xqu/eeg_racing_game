# eeg_racing_game
EEG controlled racing games

# How to setup for Windows:

## Step 1: Install TuxRacer
Tux Racer is a free, open-source game where you control a penguin down a slope. Here’s how to install it:

Download Tux Racer on SourceForge from their [website.](https://tuxracer.sourceforge.net/download.html#Windows)\
Run the installer and follow the prompts.

## Step 2: Install BlueMuse

BlueMuse is software that allows the Muse 2 headband to communicate over Bluetooth.

Download BlueMuse from the official GitHub [repo.](https://github.com/kowalej/BlueMuse)\
Run the installer and follow the prompts to complete the installation.\
Ensure your Muse 2 headband is fully charged and in pairing mode.\
Open BlueMuse and connect to the Muse headband.

## Step 3A: Create Python Script
You can use this [repo](https://github.com/YYK2007/VirtualScrollableKeyboard/tree/main) and follow the instructions provided by Andrews\
Then you can run this and you should be done.

## Step 3B: Setup Python 
If you are having a hard time following that you can also follow what I did:

### A:
Open a terminal or command prompt.\
Install the necessary Python libraries:\
```pip install muselsl numpy tensorflow```\
muselsl: Allows streaming data from the Muse headband.\
numpy: Used for numerical operations.\
tensorflow: For building and training the machine learning model.

### B:
Ensure BlueMuse is running and your Muse 2 is connected.\
Use the muselsl library to start a data stream:\
```muselsl stream```\
Open another terminal to record data:\
```muselsl record -d 60 -f data.csv```\
This command records 60 seconds of data and saves it as data.csv. Repeat for each action (e.g., "left" or "right" thoughts) to collect training data for different movements.

### C:
Train the Model for Basic Controls\
Use the recorded EEG data for training. Organize data into two categories: "left" and "right."

Theres a basic example of a python code to train a simple model using binary classification with a feedforward neural network in the folder, which is meant to be used as a template

### D:
Then you can use live EEG data to control Tux Racer. Use the following template code in the folder
