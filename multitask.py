import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW

# toy dataset with positive and negative examples
data = [
    {"hatespeech": "This is offensive.", "counter_fact": "Statistics show that diversity improves innovation.", "relevance": 1, "stance": 1},
    {"hatespeech": "Women are not good at tech.", "counter_fact": "Women in tech contribute to major innovations.", "relevance": 1, "stance": 1},
    {"hatespeech": "Immigrants are ruining the economy.", "counter_fact": "Research shows immigrants boost the economy.", "relevance": 1, "stance": 1},
    {"hatespeech": "Diversity doesn't matter.", "counter_fact": "Studies indicate diversity enhances creativity.", "relevance": 1, "stance": 1},
    {"hatespeech": "Vaccines cause autism.", "counter_fact": "New fashion trends are rising globally.", "relevance": 0, "stance": 0},  # Irrelevant counter_fact
    {"hatespeech": "Climate change is a hoax.", "counter_fact": "Scientists have proven climate change is real and dangerous.", "relevance": 1, "stance": 1},  # Opposing stance
    {"hatespeech": "Dogs are better than cats.", "counter_fact": "Studies show cats are better for small spaces.", "relevance": 1, "stance": 0},  # Opposing stance
    {"hatespeech": "Smoking is not harmful.", "counter_fact": "Statistics show that smoking leads to lung cancer.", "relevance": 1, "stance": 1},
    {"hatespeech": "Global warming is not happening.", "counter_fact": "New video games are breaking sales records.", "relevance": 0, "stance": 0}  # Irrelevant counter_fact
]

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5
MAX_LEN = 512

# Dataset class for BERT input preparation
class HateCounterFactDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        hatespeech = item["hatespeech"]
        counter_fact = item["counter_fact"]
        relevance = item["relevance"]
        stance = item["stance"]
        
        inputs = self.tokenizer(hatespeech, counter_fact, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "relevance": torch.tensor(relevance, dtype=torch.float),
            "stance": torch.tensor(stance, dtype=torch.float)
        }

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare the dataset and dataloader
dataset = HateCounterFactDataset(data, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Multi-task Learning BERT Model
class MultiTaskBERT(nn.Module):
    def __init__(self):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        
        # Separate heads for relevance and stance prediction
        self.fc_relevance = nn.Linear(768, 1)  # Relevance score (binary classification)
        self.fc_stance = nn.Linear(768, 1)     # Stance score (regression 0-1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # CLS token output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Predict relevance and stance
        relevance_score = torch.sigmoid(self.fc_relevance(pooled_output))  # Binary classification (0-1)
        stance_score = torch.sigmoid(self.fc_stance(pooled_output))        # Regression (0-1)
        
        return relevance_score, stance_score

# Instantiate the model
model = MultiTaskBERT()

# Loss functions and optimizer
criterion_relevance = nn.BCELoss()  # Binary Cross Entropy Loss for relevance
criterion_stance = nn.MSELoss()     # Mean Squared Error Loss for stance (regression)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training Loop for Multi-task Learning
def train_model(model, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            relevance = batch["relevance"].unsqueeze(1)
            stance = batch["stance"].unsqueeze(1)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            relevance_pred, stance_pred = model(input_ids, attention_mask)

            # Compute loss for both tasks
            relevance_loss = criterion_relevance(relevance_pred, relevance)
            stance_loss = criterion_stance(stance_pred, stance)
            loss = relevance_loss + stance_loss  # Combine losses for multi-task learning

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# Training the model
train_model(model, dataloader, EPOCHS)

# Inference
def predict(model, hatespeech, counter_fact):
    model.eval()
    inputs = tokenizer(hatespeech, counter_fact, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        relevance_pred, stance_pred = model(input_ids, attention_mask)
    
    return relevance_pred.item(), stance_pred.item()

# Example inference
hatespeech = "Vaccines cause autism."
counter_fact = "Studies have shown vaccines save lives."

relevance, stance = predict(model, hatespeech, counter_fact)
print(f"Relevance: {relevance:.2f}, Stance: {stance:.2f}")
