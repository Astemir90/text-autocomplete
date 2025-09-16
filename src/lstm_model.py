# код модели LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# функция чтения конфиг
with open('configs/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=config['lstm_model']['embed_dim'], hidden_dim=config['lstm_model']['hidden_dim'], 
                 n_layers=config['lstm_model']['n_layers']):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb)
        out = self.fc(out)
        return out, hidden
    
    def predict_next_token(self, x):
        logits, hidden = self.forward(x)
        next_token_logits = logits[:, -1, :]
        return next_token_logits, hidden
    
    def generate_sequence(self, seed_tokens, max_length=config['lstm_model']['generated_max_length'], stop_token=None):
        self.eval()
        generated = seed_tokens.copy()
        new_text = []
        with torch.no_grad():
            for _ in range(max_length):
                input_ids = torch.tensor([generated], dtype=torch.long)
                logits, _ = self.predict_next_token(input_ids)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                if stop_token is not None and next_token == stop_token:
                    break
                generated.append(next_token)
                new_text.append(next_token)
        return new_text