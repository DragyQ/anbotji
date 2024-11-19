import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import json
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from tqdm import tqdm

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
            labels['gender'], labels['education'], labels['race'], labels['age'], labels['income']
        ]
        demographic_features = torch.tensor(
            [float(feature) if isinstance(feature, (int, float)) else 0 for feature in demographic_features],
            dtype=torch.float32
        )
        
        personality_scores = torch.tensor([
            labels['personality_conscientiousness'],
            labels['personality_openess'],
            labels['personality_extraversion'],
            labels['personality_agreeableness'],
            labels['personality_stability']
        ], dtype=torch.float32)

        iri_scores = torch.tensor([
            labels['iri_perspective_taking'],
            labels['iri_personal_distress'],
            labels['iri_fantasy'],
            labels['iri_empathatic_concern']
        ], dtype=torch.float32)

        #Empathy and distress (these are numeric, so we can directly convert them)
        empathy_score = torch.tensor(labels['empathy'], dtype=torch.float32)
        distress_score = torch.tensor(labels['distress'], dtype=torch.float32)

        # Emotion handling (multi-hot encoding)
        emotion_tensor = self.get_emotion_vector(labels['emotion'])
        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "demographics": demographic_features,
            "personality": personality_scores,
            "iri": iri_scores,
            "empathy": empathy_score,
            "distress": distress_score,
            "emotion": emotion_tensor
        }
    def get_emotion_vector(self, emotion_str):  #since some of the entries have more than 1 emotion, it is better to have multi-hot encoding, that is to set the value to 1 in matrix if the emotion is present
        emotion_mapping = {
            "sadness": 0,
            "neutral": 1,
            "anger": 2,
            "disgust": 3,
            "hope": 4,
            "joy": 5,
            "surprise": 6,
            "fear": 7
        }
        emotions = emotion_str.split('/')
        emotion_vector = torch.zeros(len(emotion_mapping), dtype=torch.float32)
        
        for emotion in emotions:
            emotion = emotion.strip().lower()
            if emotion in emotion_mapping:
                emotion_vector[emotion_mapping[emotion]] = 1.0  # Set the corresponding emotion index to 1
        
        return emotion_vector

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
        self.fc_out_emotions = torch.nn.Linear(40, num_emotions)  # Emotion scores (8 possible emotions)

    def forward(self, input_ids, attention_mask, demographics,empathy_distress): #specifying the sequence of operations for model to produce predictions 
        '''
        input_ids: tokenized input text
        attention_mask: mask which distinguishes padding token from actual token
        demographics: demographic info of the person
        empathy_distress: empathy and distress features
        '''
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask) # passing through model for generating embeddings for the text
        text_features = self.fc_text(base_output.pooler_output) #pooler_output layer has the embeddings for the input text which is mapped to 16 features using fc_text layer
        demographic_features = self.fc_demographics(demographics) #demographic info is transformed to 16 dimension space
        empathy_distress_features = self.fc_empathy_distress(empathy_distress) #empathy and distress info is tranformed to 8 dimension space

        combined_features = torch.cat([text_features, demographic_features, empathy_distress_features], dim=1) #combining all the features to get 40 total features
        # the combined features is passed through fully connected layer and converted into respective dimension space
        personality_output = self.fc_out_personality(combined_features) # 5 in this case
        iri_output = self.fc_out_iri(combined_features) # 4 in this case
        emotions_output = self.fc_out_emotions(combined_features) #8 here
        return personality_output, iri_output, emotions_output
    
class Trainer():
    def __init__(self, model, optimizer, device):
        self.model = model  # The model to be trained
        self.optimizer = optimizer  # The optimizer for parameter updates
        self.device = device  # The device to run the training (e.g., 'cuda' for GPU, 'cpu' for CPU)

        # Move the model to the selected device
        self.model.to(self.device)

    # Training function
    def train(self, data_loader, optimizer):
        self.model.train()  # Set model to training mode
        
        # Initialize variables to keep track of metrics and loss
        total_precision_personality = 0
        total_recall_personality = 0
        total_f1_personality = 0
        total_precision_iri = 0
        total_recall_iri = 0
        total_f1_iri = 0
        total_precision_emotions = 0
        total_recall_emotions = 0
        total_f1_emotions = 0
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
            emotion_targets = batch['emotion'].to(self.device) 
            
            # Zero out the gradients for the optimizer
            optimizer.zero_grad()
            
            # applying the model
            personality_output, iri_output, emotions_output = self.model(input_ids, attention_mask, demographics, empathy_distress)
            
            # Compute the losses for each output (personality, IRI, emotions)
            loss_personality = self.loss_function(personality_output, personality_targets)
            loss_iri = self.loss_function(iri_output, iri_targets)
            loss_emotions = self.loss_function(emotions_output, emotion_targets)
            
            # Combine all the losses
            loss = loss_personality + loss_iri + loss_emotions
            
            # Backward pass to calculate the gradients
            loss.backward()
            
            # then update the model parameters as per the gradients
            optimizer.step()
            
            total_loss += loss.item()

            #computing argmax and then applying numpy
            personality_preds = torch.argmax(personality_output, dim=1).detach().cpu().numpy()
            iri_preds = torch.argmax(iri_output, dim=1).detach().cpu().numpy()
            emotions_preds = torch.argmax(emotions_output, dim=1).detach().cpu().numpy()

            personality_targets_np = personality_targets.detach().cpu().numpy()
            iri_targets_np = iri_targets.detach().cpu().numpy()
            emotion_targets_np = emotion_targets.detach().cpu().numpy()

            # Calculate metrics for personality
            precision_personality = precision_score(personality_targets_np, personality_preds, average='macro',zero_division=0) # for average = 'macro', it calculates precision/recall/F1 score for each class and then averages them
            recall_personality = recall_score(personality_targets_np, personality_preds, average='macro',zero_division=0)
            f1_personality = f1_score(personality_targets_np, personality_preds, average='macro', zero_division=0)

            # Calculate metrics for IRI
            precision_iri = precision_score(iri_targets_np, iri_preds, average='macro', zero_division=0)
            recall_iri = recall_score(iri_targets_np, iri_preds, average='macro', zero_division=0)
            f1_iri = f1_score(iri_targets_np, iri_preds, average='macro',zero_division=0 )

            # Calculate metrics for emotions
            precision_emotions = precision_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)
            recall_emotions = recall_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)
            f1_emotions = f1_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)

            # Accumulate metrics for averaging later
            total_precision_personality += precision_personality
            total_recall_personality += recall_personality
            total_f1_personality += f1_personality

            total_precision_iri += precision_iri
            total_recall_iri += recall_iri
            total_f1_iri += f1_iri

            total_precision_emotions += precision_emotions
            total_recall_emotions += recall_emotions
            total_f1_emotions += f1_emotions
        
        # Average metrics for the entire training epoch
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches

        avg_precision_personality = total_precision_personality / num_batches
        avg_recall_personality = total_recall_personality / num_batches
        avg_f1_personality = total_f1_personality / num_batches
        
        avg_precision_iri = total_precision_iri / num_batches
        avg_recall_iri = total_recall_iri / num_batches
        avg_f1_iri = total_f1_iri / num_batches
        
        avg_precision_emotions = total_precision_emotions / num_batches
        avg_recall_emotions = total_recall_emotions / num_batches
        avg_f1_emotions = total_f1_emotions / num_batches
        
        # Printing out the metrics for the training epoch
        print(f"Training Loss: {avg_loss:.4f}")
        print(f"Personality -> Precision: {avg_precision_personality:.4f}, Recall: {avg_recall_personality:.4f}, F1 Score: {avg_f1_personality:.4f}")
        print(f"IRI -> Precision: {avg_precision_iri:.4f}, Recall: {avg_recall_iri:.4f}, F1 Score: {avg_f1_iri:.4f}")
        print(f"Emotions -> Precision: {avg_precision_emotions:.4f}, Recall: {avg_recall_emotions:.4f}, F1 Score: {avg_f1_emotions:.4f}")
        
        return avg_precision_personality, avg_recall_personality, avg_f1_personality, avg_precision_iri, avg_recall_iri, avg_f1_iri, avg_precision_emotions, avg_recall_emotions, avg_f1_emotions, avg_loss

    # Evaluation function
    def evaluate(self, data_loader):
        self.model.eval()  # Set model to evaluation mode
        
        # Initialize variables to keep track of metrics and loss
        total_precision_personality = 0
        total_recall_personality = 0
        total_f1_personality = 0
        total_precision_iri = 0
        total_recall_iri = 0
        total_f1_iri = 0
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
                
                # Compute the losses for each output (personality, IRI, emotions)
                loss_personality = self.loss_function(personality_output, personality_targets)
                loss_iri = self.loss_function(iri_output, iri_targets)
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

                # Calculate metrics for personality
                precision_personality = precision_score(personality_targets_np, personality_preds, average='macro', zero_division=0) # for average = 'macro', it calculates precision/recall/F1 score for each class and then averages them
                recall_personality = recall_score(personality_targets_np, personality_preds, average='macro', zero_division=0)
                f1_personality = f1_score(personality_targets_np, personality_preds, average='macro', zero_division=0)

                # Calculate metrics for IRI
                precision_iri = precision_score(iri_targets_np, iri_preds, average='macro', zero_division=0)
                recall_iri = recall_score(iri_targets_np, iri_preds, average='macro', zero_division=0)
                f1_iri = f1_score(iri_targets_np, iri_preds, average='macro', zero_division=0)

                # Calculate metrics for emotions
                precision_emotions = precision_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)
                recall_emotions = recall_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)
                f1_emotions = f1_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)

                # Accumulate metrics for averaging later
                total_precision_personality += precision_personality
                total_recall_personality += recall_personality
                total_f1_personality += f1_personality

                total_precision_iri += precision_iri
                total_recall_iri += recall_iri
                total_f1_iri += f1_iri

                total_precision_emotions += precision_emotions
                total_recall_emotions += recall_emotions
                total_f1_emotions += f1_emotions
        
        # Average metrics for the entire evaluation 
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches

        avg_precision_personality = total_precision_personality / num_batches
        avg_recall_personality = total_recall_personality / num_batches
        avg_f1_personality = total_f1_personality / num_batches
        
        avg_precision_iri = total_precision_iri / num_batches
        avg_recall_iri = total_recall_iri / num_batches
        avg_f1_iri = total_f1_iri / num_batches
        
        avg_precision_emotions = total_precision_emotions / num_batches
        avg_recall_emotions = total_recall_emotions / num_batches
        avg_f1_emotions = total_f1_emotions / num_batches
        
        return avg_precision_personality, avg_recall_personality, avg_f1_personality, avg_precision_iri, avg_recall_iri, avg_f1_iri, avg_precision_emotions, avg_recall_emotions, avg_f1_emotions, avg_loss
    
    def save_transformer(self):
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

    def execute(self):
        last_best_per = 0
        last_best_iri=0
        last_best_emotions=0
        # Training and evaluation loop
        epochs = 5
        for epoch in range(epochs):
            precision_perso, recall_perso, f1_perso, precision_iri, recall_iri, f1_iri, precision_emotions, recall_emotions, f1_emotions, train_loss = self.train(train_loader, optimizer)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}")
            print(f"Personality -> Precision: {precision_perso:.4f}, Recall: {recall_perso:.4f}, F1 Score: {f1_perso:.4f}")
            print(f"IRI -> Precision: {precision_iri:.4f}, Recall: {recall_iri:.4f}, F1 Score: {f1_iri:.4f}")
            print(f"Emotions -> Precision: {precision_emotions:.4f}, Recall: {recall_emotions:.4f}, F1 Score: {f1_emotions:.4f}")
        
            precision_perso_dev, recall_perso_dev, f1_perso_dev, precision_iri_dev, recall_iri_dev, f1_iri_dev, precision_emotions_dev, recall_emotions_dev, f1_emotions_dev,dev_loss = self.evaluate(dev_loader)
            print(f"Epoch {epoch+1}: Dev Loss: {dev_loss:.4f}")
            print(f"Personality -> Precision: {precision_perso_dev:.4f}, Recall: {recall_perso_dev:.4f}, F1 Score: {f1_perso_dev:.4f}")
            print(f"IRI -> Precision: {precision_iri_dev:.4f}, Recall: {recall_iri_dev:.4f}, F1 Score: {f1_iri_dev:.4f}")
            print(f"Emotions -> Precision: {precision_emotions_dev:.4f}, Recall: {recall_emotions_dev:.4f}, F1 Score: {f1_emotions_dev:.4f}")

            if (f1_perso_dev > last_best_per and f1_iri_dev > last_best_iri and f1_emotions_dev > last_best_emotions):
                print("Saving model..")
                self.save_transformer()
                last_best_per = f1_perso_dev
                last_best_iri = f1_iri_dev
                last_best_emotions = f1_emotions_dev
                print("Model saved.")
    
class Tester():
    def __init__(self, model, device):
        self.model = model  # The model to be trained
        self.device = device  # The device to run the training (e.g., 'cuda' for GPU, 'cpu' for CPU)
        # Moving the model to the selected device
        self.model.to(self.device)
    
    def test(self, data_loader):
        # testing function
        self.model.eval()  # Set model to evaluation/testing mode
        
        # Initialize variables to keep track of metrics and loss
        total_precision_personality = 0
        total_recall_personality = 0
        total_f1_personality = 0
        total_precision_iri = 0
        total_recall_iri = 0
        total_f1_iri = 0
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
                
                # Compute the losses for each output (personality, IRI, emotions)
                loss_personality = self.loss_function(personality_output, personality_targets)
                loss_iri = self.loss_function(iri_output, iri_targets)
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

                # Calculate metrics for personality
                precision_personality = precision_score(personality_targets_np, personality_preds, average='macro', zero_division=0) # for average = 'macro', it calculates precision/recall/F1 score for each class and then averages them
                recall_personality = recall_score(personality_targets_np, personality_preds, average='macro', zero_division=0)
                f1_personality = f1_score(personality_targets_np, personality_preds, average='macro', zero_division=0)

                # Calculate metrics for IRI
                precision_iri = precision_score(iri_targets_np, iri_preds, average='macro', zero_division=0)
                recall_iri = recall_score(iri_targets_np, iri_preds, average='macro', zero_division=0)
                f1_iri = f1_score(iri_targets_np, iri_preds, average='macro', zero_division=0)

                # Calculate metrics for emotions
                precision_emotions = precision_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)
                recall_emotions = recall_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)
                f1_emotions = f1_score(emotion_targets_np, emotions_preds, average='macro', zero_division=0)

                # Accumulate metrics for averaging later
                total_precision_personality += precision_personality
                total_recall_personality += recall_personality
                total_f1_personality += f1_personality

                total_precision_iri += precision_iri
                total_recall_iri += recall_iri
                total_f1_iri += f1_iri

                total_precision_emotions += precision_emotions
                total_recall_emotions += recall_emotions
                total_f1_emotions += f1_emotions
        
        # Average metrics for the entire evaluation 
        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches

        avg_precision_personality = total_precision_personality / num_batches
        avg_recall_personality = total_recall_personality / num_batches
        avg_f1_personality = total_f1_personality / num_batches
        
        avg_precision_iri = total_precision_iri / num_batches
        avg_recall_iri = total_recall_iri / num_batches
        avg_f1_iri = total_f1_iri / num_batches
        
        avg_precision_emotions = total_precision_emotions / num_batches
        avg_recall_emotions = total_recall_emotions / num_batches
        avg_f1_emotions = total_f1_emotions / num_batches
        
        return avg_precision_personality, avg_recall_personality, avg_f1_personality, avg_precision_iri, avg_recall_iri, avg_f1_iri, avg_precision_emotions, avg_recall_emotions, avg_f1_emotions, avg_loss
 
    def execute(self):
        # Testing loop
        epochs = 5
        for epoch in range(epochs):
           precision_perso_test, recall_perso_test, f1_perso_test, precision_iri_test, recall_iri_test, f1_iri_test, precision_emotions_test, recall_emotions_test, f1_emotions_test, test_loss = self.test(test_loader)
           print(f"Epoch {epoch+1}: Dev Loss: {test_loss:.4f}")
           print(f"Personality -> Precision: {precision_perso_test:.4f}, Recall: {recall_perso_test:.4f}, F1 Score: {f1_perso_test:.4f}")
           print(f"IRI -> Precision: {precision_iri_test:.4f}, Recall: {recall_iri_test:.4f}, F1 Score: {f1_iri_test:.4f}")
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
    optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8) # we can try lr = 5e-5 or lower after checking the results of training and evaluation
    trainer = Trainer(model, optimizer, device)
    trainer.execute()

   # Final testing --> comment this while training and evaluating 
    tester = Tester(model, device)
    tester.execute()

 
    