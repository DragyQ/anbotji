
# Welcome to Anbotji ;)

This repository contains the code and resources for our project on personality prediction, where a machine learning model was trained to predict personality traits from textual input and integrated into a chatbot for real-time use. 

## Requirements
All necessary packages and dependencies are listed in requirements.txt.
To install all required packages, run: `pip install -r requirements.txt`

## Dataset 
The data/ folder contains the raw and processed datasets along with preprocessing scripts:
data_transform.py : Preprocesses the raw data for it to be used to training and testing. 
raw_data.csv : Raw training data
data.json : Preprocessed training data
raw_dev.csv : Raw testing data
dev_data.json : Preprocessed testing dataset
data_split.py : Splitting the dataset into training, development and test set

## Main scripts:
Model Training and Evaluation
model_prep.py: Handles model setup, training, testing, evaluation, and performance visualization. Comments are included to explain each function.
Model Info:
We use the DistilBERT pretrained model: "distilbert-base-uncased".

## Chatbot Integration
main.py: Entry point for the chatbot. It interacts with the user, collects demographic information, and conducts conversations using personality-related questions.
questions.json: Pool of personality-related questions used by the chatbot.
questions_parser.py: Randomly selects and serves questions during the chatbot interaction.
model.py: Utilizes trained model weights to predict personality traits based on user input.
helper.py: Converts model outputs into user-friendly personality insights.

## How to Use
Install dependencies using requirements.txt (see Requirements section).
Preprocess the dataset using data_transform.py if starting from raw data.
Train and evaluate the model with model_prep.py.
Run the chatbot using main.py to interact with users and provide personality predictions.


For detailed instructions, refer to comments within each script. If you have questions or need assistance, feel free to open an issue or reach out! 😊


