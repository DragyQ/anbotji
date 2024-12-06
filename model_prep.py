import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import json
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, mean_absolute_error
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

#Constants
TEST_PATH = "/content/drive/MyDrive/CSE_354_project/data/test_data.json"
TRAIN_PATH = "/content/drive/MyDrive/CSE_354_project/data/train_data.json"
VAL_PATH = "/content/drive/MyDrive/CSE_354_project/data/dev_data.json"
# Models are stored in this path
SAVE_PATH = "/content/drive/MyDrive/CSE_354_project/data/DistilBERT"

# Load the cleaned data
def load_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Define the dataset class
class PersonalityDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text = entry['text']
        labels = entry['labels']

        # Tokenizing the essay text
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Extracting demographic and personality information
        demographic_features = [
            labels['gender'], labels['education'], labels['race'],
            self.convert_to_float(labels['age']),  # Convert age to float
            self.convert_to_float(labels['income'])  # Convert income to float
        ]
        demographic_features = torch.tensor(
            [float(feature) if isinstance(feature, (int, float)) else 0 for feature in demographic_features],
            dtype=torch.float32
        )

        personality_scores = torch.tensor([
            self.convert_to_float(labels['personality_conscientiousness']),
            self.convert_to_float(labels['personality_openess']),
            self.convert_to_float(labels['personality_extraversion']),
            self.convert_to_float(labels['personality_agreeableness']),
            self.convert_to_float(labels['personality_stability'])
        ], dtype=torch.float32)

        iri_scores = torch.tensor([
            self.convert_to_float(labels['iri_perspective_taking']),
            self.convert_to_float(labels['iri_personal_distress']),
            self.convert_to_float(labels['iri_fantasy']),
            self.convert_to_float(labels['iri_empathatic_concern'])
        ], dtype=torch.float32)

        #Empathy and distress (these are numeric, so we can directly convert them)
        # empathy_score = torch.tensor(labels['empathy'], dtype=torch.float32)
        # distress_score = torch.tensor(labels['distress'], dtype=torch.float32)

        empathy_distress = torch.tensor([labels['empathy'],labels['distress']], dtype=torch.float32)


        # Emotion handling (multi-hot encoding)
        # emotion_tensor = self.get_emotion_vector(labels['emotion'])
        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "demographics": demographic_features,
            "personality": personality_scores,
            "iri": iri_scores,
            "empathy_distress":empathy_distress,
            # "empathy": empathy_score,
            # "distress": distress_score,
            # "emotion": emotion_tensor
        }
    def convert_to_float(self, value):
        try:
            return float(value)
        except ValueError:
            return 0.0  # Default value in case conversion fails (e.g., for non-numeric strings)

    # def get_emotion_vector(self, emotion_str):  #since some of the entries have more than 1 emotion, it is better to have multi-hot encoding, that is to set the value to 1 in matrix if the emotion is present
    #     emotion_mapping = {
    #         "sadness": 0,
    #         "neutral": 1,
    #         "anger": 2,
    #         "disgust": 3,
    #         "hope": 4,
    #         "joy": 5,
    #         "surprise": 6,
    #         "fear": 7
    #     }
    #     emotions = emotion_str.split('/')
    #     emotion_vector = torch.zeros(len(emotion_mapping), dtype=torch.float32)

    #     for emotion in emotions:
    #         emotion = emotion.strip().lower()
    #         if emotion in emotion_mapping:
    #             emotion_vector[emotion_mapping[emotion]] = 1.0  # Set the corresponding emotion index to 1

    #     return emotion_vector

# Define the model
class PersonalityPredictionModel(torch.nn.Module):
    def __init__(self, base_model_name, num_emotions=8):
        super(PersonalityPredictionModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)

        self.fc_demographics = torch.nn.Linear(5, 16)  # process 5 Demographic features and transform them into 16 features
        self.fc_empathy_distress = torch.nn.Linear(2, 8)  # Empathy and distress features (2 features)
        self.fc_text = torch.nn.Linear(self.base_model.config.hidden_size, 16) #reducing the size to 16 because gemini's hidden size for output would be very high
        # total dimensions = 16 (demographics) + 8(empathy, distress) + 16(text) = 40
        self.fc_out_personality = torch.nn.Linear(40, 5)  # Personality scores (5 traits)
        self.fc_out_iri = torch.nn.Linear(40, 4)  # IRI scores (4 types)
        # self.fc_out_emotions = torch.nn.Linear(40, num_emotions)  # Emotion scores (8 possible emotions)

    def forward(self, input_ids, attention_mask, demographics,empathy_distress): #specifying the sequence of operations for model to produce predictions
        '''
        input_ids: tokenized input text
        attention_mask: mask which distinguishes padding token from actual token
        demographics: demographic info of the person
        empathy_distress: empathy and distress features
        '''
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask) # passing through model for generating embeddings for the text
        # Extract the last hidden state from the model output (DistilBERT doesn't have pooler_output)
        last_hidden_state = base_output.last_hidden_state
        text_features = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        text_features = self.fc_text(text_features)

        demographic_features = self.fc_demographics(demographics) #demographic info is transformed to 16 dimension space
        empathy_distress_features = self.fc_empathy_distress(empathy_distress) #empathy and distress info is tranformed to 8 dimension space

        combined_features = torch.cat([text_features, demographic_features, empathy_distress_features], dim=1) #combining all the features to get 40 total features
        # the combined features is passed through fully connected layer and converted into respective dimension space
        personality_output = self.fc_out_personality(combined_features) # 5 in this case
        iri_output = self.fc_out_iri(combined_features) # 4 in this case
        # emotions_output = self.fc_out_emotions(combined_features) #8 here
        # print("Shape of personality_output:", personality_output.shape)
        return personality_output, iri_output
    def get_tokenizer_and_model(self):
            return self.model, self.tokenizer

    def save_pretrained(self, save_directory):
        # Save model weights
        torch.save(self.state_dict(), f"{save_directory}/model_weights.pth")
    
class Trainer():
    def __init__(self, model, optimizer, device,tokenizer,loss_function=None):
        self.model = model  # The model to be trained
        self.optimizer = optimizer  # The optimizer for parameter updates
        self.device = device  # The device to run the training (e.g., 'cuda' for GPU, 'cpu' for CPU)
        self.save_path = SAVE_PATH
        self.tokenizer = tokenizer 
        self.loss_function = loss_function if loss_function else nn.CrossEntropyLoss()

        # Move the model to the selected device
        self.model.to(self.device)

        # Initialize lists to store metrics for visualization
        # Separate lists for training and validation metrics
        self.train_losses = []
        self.val_losses = []

        self.train_mae_personality = []
        self.val_mae_personality = []

        self.train_mse_personality = []
        self.val_mse_personality = []

        self.train_mae_iri = []
        self.val_mae_iri = []

        self.train_mse_iri = []
        self.val_mse_iri = []

        # self.train_precision_emotions = []
        # self.val_precision_emotions = []

        # self.train_recall_emotions = []
        # self.val_recall_emotions = []

        # self.train_f1_emotions = []
        # self.val_f1_emotions = []


    # Training function
    def train(self, data_loader, optimizer):
        self.model.train()  # Set model to training mode

        # Initialize variables to keep track of metrics and loss
        total_mae_personality = 0
        total_mse_personality = 0
        total_mae_iri = 0
        total_mse_iri = 0
        # total_precision_emotions = 0
        # total_recall_emotions = 0
        # total_f1_emotions = 0
        total_loss = 0

        # Loop over each batch in the data loader
        for batch_idx, batch in enumerate(tqdm(data_loader)):

            # Move data to the correct device (e.g., GPU)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            demographics = batch['demographics'].to(self.device)
            empathy_distress = batch['empathy_distress'].to(self.device)
            personality_targets = batch['personality'].to(self.device)
            iri_targets = batch['iri'].to(self.device)
            # emotion_targets = batch['emotion'].to(self.device)

            # Zero out the gradients for the optimizer
            optimizer.zero_grad()

            # applying the model
            personality_output, iri_output = self.model(input_ids, attention_mask, demographics, empathy_distress)

            loss_function_regression = nn.MSELoss()

            # Compute the losses for each output (personality, IRI, emotions)
            loss_personality = loss_function_regression(personality_output, personality_targets)
            loss_iri = loss_function_regression(iri_output, iri_targets)
            # loss_emotions = nn.BCEWithLogitsLoss()(emotions_output, emotion_targets)

            #loss_emotions = self.loss_function(emotions_output, emotion_targets)

            # Combine all the losses
            loss = loss_personality + loss_iri 

            # Backward pass to calculate the gradients
            loss.backward()

            # then update the model parameters as per the gradients
            optimizer.step()

            total_loss += loss.item()

            #computing argmax and then applying numpy
            personality_preds = personality_output.detach().cpu().numpy()
            iri_preds = iri_output.detach().cpu().numpy()
            # emotions_preds = torch.sigmoid(emotions_output).detach().cpu().numpy()
            #emotions_preds = torch.argmax(emotions_output, dim=1).detach().cpu().numpy() #for crossEntropyLoss

            personality_targets_np = personality_targets.detach().cpu().numpy()
            iri_targets_np = iri_targets.detach().cpu().numpy()
            # emotion_targets_np = emotion_targets.detach().cpu().numpy()
            #emotion_targets_np = torch.argmax(emotion_targets, dim=1).detach().cpu().numpy() #for crossEntropyLoss

            # Mean Absolute Error and Mean Squared Error for personality
            mae_personality = mean_absolute_error(personality_targets_np, personality_preds)
            mse_personality = mean_squared_error(personality_targets_np, personality_preds)

            # Mean Absolute Error and Mean Squared Error for IRI
            mae_iri = mean_absolute_error(iri_targets_np, iri_preds)
            mse_iri = mean_squared_error(iri_targets_np, iri_preds)

            # For emotions, since we used BCEWithLogitsLoss, apply a threshold to predict labels
            # emotion_preds_labels = (emotions_preds > 0.5).astype(int)

            # # Calculate metrics for emotions
            # precision_emotions = precision_score(emotion_targets_np, emotion_preds_labels, average='macro', zero_division=0)
            # recall_emotions = recall_score(emotion_targets_np, emotion_preds_labels, average='macro', zero_division=0)
            # f1_emotions = f1_score(emotion_targets_np, emotion_preds_labels, average='macro', zero_division=0)
            #f1_emotions = f1_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)

            total_mae_personality += mae_personality
            total_mse_personality += mse_personality
            total_mae_iri += mae_iri
            total_mse_iri += mse_iri

            # total_precision_emotions += precision_emotions
            # total_recall_emotions += recall_emotions
            # total_f1_emotions += f1_emotions

        # Average metrics for the entire training epoch
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches

        avg_mae_personality = total_mae_personality / num_batches
        avg_mse_personality = total_mse_personality / num_batches
        avg_mae_iri = total_mae_iri / num_batches
        avg_mse_iri = total_mse_iri / num_batches

        # avg_precision_emotions = total_precision_emotions / num_batches
        # avg_recall_emotions = total_recall_emotions / num_batches
        # avg_f1_emotions = total_f1_emotions / num_batches

        # Append the metrics to the lists for visualization
        self.train_losses.append(avg_loss)
        self.train_mae_personality.append(avg_mae_personality)
        self.train_mse_personality.append(avg_mse_personality)
        self.train_mae_iri.append(avg_mae_iri)
        self.train_mse_iri.append(avg_mse_iri)
        # self.train_precision_emotions.append(avg_precision_emotions)
        # self.train_recall_emotions.append(avg_recall_emotions)
        # self.train_f1_emotions.append(avg_f1_emotions)

        # Printing out the metrics for the training epoch
        print(f"Training Loss: {avg_loss:.4f}")
        print(f"Personality -> MAE: {avg_mae_personality:.4f}, MSE: {avg_mse_personality:.4f}")
        print(f"IRI -> MAE: {avg_mae_iri:.4f}, MSE: {avg_mse_iri:.4f}")
        # print(f"Emotions -> Precision: {avg_precision_emotions:.4f}, Recall: {avg_recall_emotions:.4f}, F1 Score: {avg_f1_emotions:.4f}")

        return avg_mae_personality, avg_mse_personality, avg_mae_iri, avg_mse_iri, avg_loss

    # Evaluation function
    def evaluate(self, data_loader):
        self.model.eval()  # Set model to evaluation mode

        # Initialize variables to keep track of metrics and loss
        total_mae_personality = 0
        total_mse_personality = 0
        total_mae_iri = 0
        total_mse_iri = 0
        # total_precision_emotions = 0
        # total_recall_emotions = 0
        # total_f1_emotions = 0
        total_loss = 0

        # Loop over each batch in the data loader
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):

                # Move data to the correct device (e.g., GPU)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                demographics = batch['demographics'].to(self.device)
                empathy_distress = batch['empathy_distress'].to(self.device)
                personality_targets = batch['personality'].to(self.device)
                iri_targets = batch['iri'].to(self.device)
                # emotion_targets = batch['emotion'].to(self.device)

                # applying the model
                personality_output, iri_output = self.model(input_ids, attention_mask, demographics, empathy_distress)

                loss_function_regression = nn.MSELoss()  # Or nn.L1Loss() for MAE

                # Compute the losses for each output (personality, IRI, emotions)
                loss_personality = loss_function_regression(personality_output, personality_targets)
                loss_iri = loss_function_regression(iri_output, iri_targets)
                # loss_emotions = nn.BCEWithLogitsLoss()(emotions_output, emotion_targets)

                #loss_emotions = self.loss_function(emotions_output, emotion_targets)

                # Combine all the losses
                loss = loss_personality + loss_iri 
                total_loss += loss.item() # loss is not back propagated here

                #computing argmax and then applying numpy
                personality_preds = personality_output.detach().cpu().numpy()
                iri_preds = iri_output.detach().cpu().numpy()
                # emotions_preds = torch.sigmoid(emotions_output).detach().cpu().numpy()
                #emotions_preds = torch.argmax(emotions_output, dim=1).detach().cpu().numpy()

                personality_targets_np = personality_targets.detach().cpu().numpy()
                iri_targets_np = iri_targets.detach().cpu().numpy()
                # emotion_targets_np = emotion_targets.detach().cpu().numpy()
                #emotion_targets_np = torch.argmax(emotion_targets, dim=1).detach().cpu().numpy()

                # Mean Absolute Error and Mean Squared Error for personality
                mae_personality = mean_absolute_error(personality_targets_np, personality_preds)
                mse_personality = mean_squared_error(personality_targets_np, personality_preds)

                # Mean Absolute Error and Mean Squared Error for IRI
                mae_iri = mean_absolute_error(iri_targets_np, iri_preds)
                mse_iri = mean_squared_error(iri_targets_np, iri_preds)

                # For emotions, since we used BCEWithLogitsLoss, apply a threshold to predict labels
                # emotion_preds_labels = (emotions_preds > 0.5).astype(int)

                # # Calculate metrics for emotions
                # precision_emotions = precision_score(emotion_targets_np, emotion_preds_labels, average='macro', zero_division=0)
                # recall_emotions = recall_score(emotion_targets_np, emotion_preds_labels, average='macro', zero_division=0)
                # f1_emotions = f1_score(emotion_targets_np, emotion_preds_labels, average='macro', zero_division=0)
                #f1_emotions = f1_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)

                # Accumulate metrics for averaging later
                total_mae_personality += mae_personality
                total_mse_personality += mse_personality
                total_mae_iri += mae_iri
                total_mse_iri += mse_iri

                # total_precision_emotions += precision_emotions
                # total_recall_emotions += recall_emotions
                # total_f1_emotions += f1_emotions

        # Average metrics for the entire evaluation
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches

        avg_mae_personality = total_mae_personality / num_batches
        avg_mse_personality = total_mse_personality / num_batches
        avg_mae_iri = total_mae_iri / num_batches
        avg_mse_iri = total_mse_iri / num_batches
        # avg_precision_emotions = total_precision_emotions / num_batches
        # avg_recall_emotions = total_recall_emotions / num_batches
        # avg_f1_emotions = total_f1_emotions / num_batches

        # Append the metrics to the lists for visualization
        self.val_losses.append(avg_loss)
        self.val_mae_personality.append(avg_mae_personality)
        self.val_mse_personality.append(avg_mse_personality)
        self.val_mae_iri.append(avg_mae_iri)
        self.val_mse_iri.append(avg_mse_iri)
        # self.val_precision_emotions.append(avg_precision_emotions)
        # self.val_recall_emotions.append(avg_recall_emotions)
        # self.val_f1_emotions.append(avg_f1_emotions)

        return avg_mae_personality, avg_mse_personality, avg_mae_iri, avg_mse_iri, avg_loss

    def save_transformer(self):
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

    def plot_metrics(self):
        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Plot other metrics like MAE, MSE for personality
        plt.subplot(2, 2, 2)
        plt.plot(self.train_mae_personality, label='Training MAE Personality')
        plt.plot(self.train_mse_personality, label='Training MSE Personality')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Personality Metrics over Epochs')
        plt.legend()

        # Plot other metrics like MAE, MSE for personality
        plt.subplot(2, 2, 2)
        plt.plot(self.val_mae_personality, label='Dev MAE Personality')
        plt.plot(self.val_mse_personality, label='Dev MSE Personality')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Personality Metrics over Epochs')
        plt.legend()

        # Plot MAE, MSE for IRI
        plt.subplot(2, 2, 3)
        plt.plot(self.train_mae_iri, label='Training MAE IRI')
        plt.plot(self.train_mse_iri, label='Training MSE IRI')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('IRI Metrics over Epochs')
        plt.legend()

        # Plot MAE, MSE for IRI
        plt.subplot(2, 2, 3)
        plt.plot(self.val_mae_iri, label='Dev MAE IRI')
        plt.plot(self.val_mse_iri, label='Dev MSE IRI')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('IRI Metrics over Epochs')
        plt.legend()

        # # Plot emotion metrics: precision, recall, F1 score
        # plt.subplot(2, 2, 4)
        # plt.plot(self.train_precision_emotions, label='Training Precision Emotions')
        # plt.plot(self.train_recall_emotions, label='Training Recall Emotions')
        # plt.plot(self.train_f1_emotions, label='Training F1 Score Emotions')
        # plt.xlabel('Epoch')
        # plt.ylabel('Value')
        # plt.title('Emotion Metrics over Epochs')
        # plt.legend()

        # # Plot emotion metrics: precision, recall, F1 score
        # plt.subplot(2, 2, 4)
        # plt.plot(self.val_precision_emotions, label='Dev Precision Emotions')
        # plt.plot(self.val_recall_emotions, label='Dev Recall Emotions')
        # plt.plot(self.val_f1_emotions, label='Dev F1 Score Emotions')
        # plt.xlabel('Epoch')
        # plt.ylabel('Value')
        # plt.title('Emotion Metrics over Epochs')
        # plt.legend()

        plt.tight_layout()
        plt.show()

    def execute(self):
        last_best_per = float('inf') # Since lower MAE and MSE are better
        last_best_iri=float('inf')
        last_best_emotions=0
        # Training and evaluation loop
        epochs = 17
        for epoch in range(epochs):
            avg_mae_personality, avg_mse_personality, avg_mae_iri, avg_mse_iri, train_loss = self.train(train_loader, optimizer)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}")
            print(f"Personality -> MAE: {avg_mae_personality:.4f}, MSE: {avg_mse_personality:.4f}")
            print(f"IRI -> MAE: {avg_mae_iri:.4f}, MSE: {avg_mse_iri:.4f}")
            # print(f"Emotions -> Precision: {precision_emotions:.4f}, Recall: {recall_emotions:.4f}, F1 Score: {f1_emotions:.4f}")

            avg_mae_personality_dev, avg_mse_personality_dev, avg_mae_iri_dev, avg_mse_iri_dev, dev_loss = self.evaluate(dev_loader)
            print(f"Epoch {epoch+1}: Dev Loss: {dev_loss:.4f}")
            print(f"Personality -> MAE: {avg_mae_personality_dev:.4f}, MSE: {avg_mse_personality_dev:.4f}")
            print(f"IRI -> MAE: {avg_mae_iri_dev:.4f}, MSE: {avg_mse_iri_dev:.4f}")
            # print(f"Emotions -> Precision: {precision_emotions_dev:.4f}, Recall: {recall_emotions_dev:.4f}, F1 Score: {f1_emotions_dev:.4f}")

            if (avg_mae_personality_dev < last_best_per and avg_mse_personality_dev < last_best_per and avg_mae_iri_dev < last_best_iri and avg_mse_iri_dev < last_best_iri ):
              print("Saving model..")
              self.save_transformer()  # Save the model
              last_best_per = avg_mae_personality_dev
              last_best_iri = avg_mse_iri_dev  # We use MSE for IRI
              print("Model saved.")

        self.plot_metrics()
    
class Tester():  #Currently not using this function because evaluation is used for testing purpose
    def __init__(self, model, device):
        self.model = model  # The model to be trained
        self.device = device  # The device to run the training (e.g., 'cuda' for GPU, 'cpu' for CPU)
        # Moving the model to the selected device
        self.model.to(self.device)

    def test(self, data_loader):
        # testing function
        self.model.eval()  # Set model to evaluation/testing mode

        # Initialize variables to keep track of metrics and loss
        total_mae_personality = 0
        total_mse_personality = 0
        total_mae_iri = 0
        total_mse_iri = 0
        total_precision_emotions = 0
        total_recall_emotions = 0
        total_f1_emotions = 0
        total_loss = 0

        # Loop over each batch in the data loader
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):

                # Move data to the correct device (e.g., GPU)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                demographics = batch['demographics'].to(self.device)
                empathy_distress = batch['empathy_distress'].to(self.device)
                personality_targets = batch['personality'].to(self.device)
                iri_targets = batch['iri'].to(self.device)
                emotion_targets = batch['emotion'].to(self.device)

                # applying the model
                personality_output, iri_output, emotions_output = self.model(input_ids, attention_mask, demographics, empathy_distress)

                loss_function_regression = nn.MSELoss()

                # Compute the losses for each output (personality, IRI, emotions)
                loss_personality = loss_function_regression(personality_output, personality_targets)
                loss_iri = loss_function_regression(iri_output, iri_targets)
                loss_emotions = self.loss_function(emotions_output, emotion_targets)

                # Combine all the losses
                loss = loss_personality + loss_iri + loss_emotions
                total_loss += loss.item() # loss is not back propagated here

                #computing argmax and then applying numpy
                personality_preds = torch.argmax(personality_output, dim=1).detach().cpu().numpy()
                iri_preds = torch.argmax(iri_output, dim=1).detach().cpu().numpy()
                emotions_preds = torch.argmax(emotions_output, dim=1).detach().cpu().numpy()

                personality_targets_np = personality_targets.detach().cpu().numpy()
                iri_targets_np = iri_targets.detach().cpu().numpy()
                emotion_targets_np = emotion_targets.detach().cpu().numpy()

                # Mean Absolute Error and Mean Squared Error for personality
                mae_personality = mean_absolute_error(personality_targets_np, personality_preds)
                mse_personality = mean_squared_error(personality_targets_np, personality_preds)

                # Mean Absolute Error and Mean Squared Error for IRI
                mae_iri = mean_absolute_error(iri_targets_np, iri_preds)
                mse_iri = mean_squared_error(iri_targets_np, iri_preds)

                # Calculate metrics for emotions
                precision_emotions = precision_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)
                recall_emotions = recall_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)
                f1_emotions = f1_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)

                # Accumulate metrics for averaging later
                total_mae_personality += mae_personality
                total_mse_personality += mse_personality
                total_mae_iri += mae_iri
                total_mse_iri += mse_iri

                total_precision_emotions += precision_emotions
                total_recall_emotions += recall_emotions
                total_f1_emotions += f1_emotions

        # Average metrics for the entire evaluation
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches

        avg_mae_personality = total_mae_personality / num_batches
        avg_mse_personality = total_mse_personality / num_batches
        avg_mae_iri = total_mae_iri / num_batches
        avg_mse_iri = total_mse_iri / num_batches

        avg_precision_emotions = total_precision_emotions / num_batches
        avg_recall_emotions = total_recall_emotions / num_batches
        avg_f1_emotions = total_f1_emotions / num_batches

        return avg_mae_personality, avg_mse_personality, avg_mae_iri, avg_mse_iri, avg_precision_emotions, avg_recall_emotions, avg_f1_emotions, avg_loss

    def execute(self):
        # Testing loop
        epochs = 5
        for epoch in range(epochs):
           avg_mae_personality, avg_mse_personality, avg_mae_iri, avg_mse_iri, precision_emotions_test, recall_emotions_test, f1_emotions_test, test_loss = self.test(test_loader)
           print(f"Epoch {epoch+1}: Dev Loss: {test_loss:.4f}")
           print(f"Personality -> MAE: {avg_mae_personality:.4f}, MSE: {avg_mse_personality:.4f}")
           print(f"IRI -> MAE: {avg_mae_iri:.4f}, MSE: {avg_mse_iri:.4f}")
           print(f"Emotions -> Precision: {precision_emotions_test:.4f}, Recall: {recall_emotions_test:.4f}, F1 Score: {f1_emotions_test:.4f}")



# Main script
if __name__ == "__main__":
    # File paths
    train_path = "./data/data.json"
    dev_path = "./data/dev.json"
    test_path = "./data/test.json"

    # Load data
    train_data = load_data(train_path)
    dev_data = load_data(dev_path)
    test_data = load_data(test_path)

    # Initialize tokenizer and datasets
    base_model_name = "distilbert-base-uncased"
    #change to an encoder only pretrained model --> distillbert,  
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_dataset = PersonalityDataset(train_data, tokenizer)
    dev_dataset = PersonalityDataset(dev_data, tokenizer)
    test_dataset = PersonalityDataset(test_data, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalityPredictionModel(base_model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-2, eps=1e-8) # we can try lr = 5e-5 or lower after checking the results of training and evaluation
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, device, tokenizer, loss_function=loss_fn)
    trainer.execute()

   # Final testing --> comment this while training and evaluating 
    tester = Tester(model, device)
    tester.execute()

 
    