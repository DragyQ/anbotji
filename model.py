import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict
import os


BASE_MODEL_NAME = "distilbert-base-uncased"

class PersonalityPredictionModel(torch.nn.Module):
    def __init__(self, base_model_name: str = BASE_MODEL_NAME):
        super().__init__()
        
        # Load pre-trained model
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        
        # Fully connected layers
        self.fc_text = torch.nn.Linear(hidden_size, 16)
        self.fc_demographics = torch.nn.Linear(5, 16)
        self.fc_empathy_distress = torch.nn.Linear(2, 8)
        
        # Output layers
        self.fc_out_personality = torch.nn.Linear(40, 5)
        self.fc_out_iri = torch.nn.Linear(40, 4)

    
    def forward(self, input_ids, attention_mask, demographics, empathy_distress):
        # Extract text features
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = base_output.last_hidden_state[:, 0, :]  # CLS token representation
        text_features = self.fc_text(text_features)
        
        # Process demographic and empathy/distress features
        demographic_features = self.fc_demographics(demographics)
        empathy_distress_features = self.fc_empathy_distress(empathy_distress)
        
        # Concatenate all features
        combined_features = torch.cat([text_features, demographic_features, empathy_distress_features], dim=1)
        
        # Generate outputs
        personality_output = self.fc_out_personality(combined_features)
        iri_output = self.fc_out_iri(combined_features)
        
        return personality_output, iri_output

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = PersonalityPredictionModel()

# Load checkpoint weights
state_dict = torch.load("model_weights.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)  # Allow non-strict matching

# Clear warning printed out in terminal.
os.system('cls' if os.name == 'nt' else 'clear')

def predict_outputs(essay: str, demographic_labels: Dict, empathy_distress_labels: Dict[str, str]):
    input = tokenizer(
        text=essay,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = input['input_ids']
    attention_mask = input['attention_mask']
    
    demographic_features = torch.tensor(
        [
            float(demographic_labels.get(key, 0)) for key in ["gender", "education", "race"]
        ] + [
            convert_to_float(demographic_labels.get("age", "0")),
            convert_to_float(demographic_labels.get("income", "0"))
        ],
        dtype=torch.float32
    ).unsqueeze(0)
    
    empathy_distress_features = torch.tensor(
        [
            convert_to_float(empathy_distress_labels.get("empathy", "0")),
            convert_to_float(empathy_distress_labels.get("distress", "0"))
        ],
        dtype=torch.float32
    ).unsqueeze(0)
    
    # Forward 
    with torch.no_grad():
        personality_output, iri_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            demographics=demographic_features,
            empathy_distress=empathy_distress_features
        )
    
    return personality_output, iri_output
    

def convert_to_float(val: str) -> float:
    try:
        return float(val)
    except ValueError:
        return 0.0
    

if __name__ == "__main__":
    """This module should not be ran as main."""
    pass

