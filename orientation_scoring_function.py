import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.model_selection import train_test_split
import numpy as np

class IdeologyDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=128):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'score': score
        }

class OrientationScoringModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout_rate=0.1):
        super(OrientationScoringModel, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        score = self.regressor(pooled_output)
        score = score * 10.0
        return score

class OrientationScorer:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OrientationScoringModel(model_name).to(self.device)
        self.is_trained = False
        
    def train(self, texts, scores, epochs=10, batch_size=16, learning_rate=2e-5):
        train_texts, val_texts, train_scores, val_scores = train_test_split(
            texts, scores, test_size=0.2, random_state=42
        )
        
        train_dataset = IdeologyDataset(train_texts, train_scores, self.model.tokenizer)
        val_dataset = IdeologyDataset(val_texts, val_scores, self.model.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        print("Training orientation scoring model...")
        
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['score'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['score'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask).squeeze()
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        self.is_trained = True
        print("Model training completed!")
    
    def __call__(self, text):
        if not self.is_trained:
            raise ValueError("Model not trained, please call train() first")
        
        self.model.eval()
        with torch.no_grad():
            encoding = self.model.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            score = self.model(input_ids, attention_mask).squeeze().cpu().item()
            
            return score
    
    def score_batch(self, texts):
        return [self(text) for text in texts]
    
    def get_ideology_class(self, score):
        if 0 <= score < 2:
            return "leftist"
        elif 3 <= score < 4:
            return "leftleaning"
        elif 5 <= score < 6:
            return "moderate"
        elif 7 <= score < 8:
            return "rightleaning"
        elif 9 <= score <= 10:
            return "rightist"
        else:
            return "Score out of range"

def create_sample_data():
    sample_texts = [
        "We should increase investment in social welfare and healthcare to ensure everyone has basic living security.",
        "Lowering taxes can stimulate economic growth and create more job opportunities.",
        "Environmental protection and economic development need balanced consideration, neither should be neglected.",
        "The government should reduce intervention in the market and let free competition determine resource allocation.",
        "We need stricter gun control laws to ensure public safety.",
        "Personal freedom is most important, and government power should be strictly limited."
    ]
    
    sample_scores = [2.1, 7.8, 4.9, 8.2, 3.5, 9.1]
    
    return sample_texts, sample_scores

def main():
    scorer = OrientationScorer()
    
    texts, scores = create_sample_data()
    scorer.train(texts, scores, epochs=5, batch_size=2)
    
    test_texts = [
        "A fair society should provide equal opportunities for everyone.",
        "Market competition is the most effective mechanism for driving social progress."
    ]
    
    print("\nTest scoring results:")
    for text in test_texts:
        score = scorer(text)
        ideology_class = scorer.get_ideology_class(score)
        print(f"Text: {text}")
        print(f"Orientation Score: {score:.2f} - Classification: {ideology_class}")
        print("-" * 50)

if __name__ == "__main__":
    main()