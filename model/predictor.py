import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from .utils import clean_text

class StressPredictor:
    def __init__(self, device):
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load model
        checkpoint = torch.load('best_model.pt', map_location=device)
        
        # Initialize model components
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2,
            output_hidden_states=True
        ).to(device)
        
        self.liwc_layer = torch.nn.Sequential(
            torch.nn.Linear(11, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.LayerNorm(32)
        ).to(device)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768 + 32, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 2)
        ).to(device)
        
        # Load saved weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.liwc_layer.load_state_dict(checkpoint['liwc_layer_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        # Set to evaluation mode
        self.model.eval()
        self.liwc_layer.eval()
        self.classifier.eval()
    
    def predict(self, text):
        cleaned_text = clean_text(text)
        
        encoded = self.tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        dummy_liwc = torch.zeros((1, 11)).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            bert_features = outputs.hidden_states[-1][:, 0, :]
            liwc_output = self.liwc_layer(dummy_liwc)
            combined_features = torch.cat((bert_features, liwc_output), dim=1)
            logits = self.classifier(combined_features)
            probabilities = torch.softmax(logits, dim=1)
        
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return {
            'text': cleaned_text,
            'stress_level': predicted_class,
            'confidence': confidence,
            'prediction': 'Stressed' if predicted_class == 1 else 'Not Stressed'
        }