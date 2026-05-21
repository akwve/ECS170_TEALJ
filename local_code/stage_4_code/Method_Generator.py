'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class Method_RNN_Generator(method, nn.Module):
    data = None
    max_epoch = 60
    learning_rate = 3e-4

    def __init__(self, mName, mDescription, vocab_size, embed_dim=256, hidden_dim=512):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size=embed_dim,hidden_size=hidden_dim,dropout=0.3,num_layers=2,batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.to(self.device)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.embed_drop(x)
        out, hidden = self.lstm(x, hidden)
        out = self.norm(out)
        out = self.drop(out)
        logits = self.fc(out)
        return logits, hidden

    def train_model(self, X, y):
        loss_record = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        X = np.array(X)
        y = np.array(y)
        batch_size = 64
        for epoch in range(self.max_epoch):
            epoch_loss = 0
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]
            for i in range(0, len(X), batch_size):
                X_batch = torch.LongTensor(X[i:i+batch_size]).to(self.device)
                y_batch = torch.LongTensor(y[i:i+batch_size]).to(self.device)
                logits, _ = self.forward(X_batch)
                loss = loss_function(
                    logits[:, -1, :],   # ONLY last token prediction (IMPORTANT FIX)
                    y_batch
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / (len(X) / batch_size)
            print(f"Epoch {epoch+1}/{self.max_epoch} | Loss: {avg_loss:.4f}")
            loss_record.append(avg_loss)
        return loss_record

    def generate(self, word_to_idx, idx_to_word, start_words, max_len=20):
        self.eval()
        words = start_words.lower().split()
        idxs = [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in words]
        input_seq = torch.LongTensor(idxs).unsqueeze(0).to(self.device)
        hidden = None
        for _ in range(max_len):
            logits, hidden = self.forward(input_seq)
            temperature = 0.65
            top_k = 5
            step_logits = logits[0, -1] / temperature
            recent_words = words[-5:]
            for word in set(recent_words):
                token_id = word_to_idx.get(word)
                if token_id is not None:
                    step_logits[token_id] /= 2.5
            values, indices = torch.topk(step_logits, top_k)
            probs = torch.softmax(values, dim=0)
            sampled_idx = torch.multinomial(probs, 1).item()
            next_token = indices[sampled_idx].item()
            next_word = idx_to_word[next_token]
            if next_word in ["<PAD>", "<UNK>"]:
                break
            if next_word == "<END>":
                break
            words.append(next_word)
            input_seq = torch.LongTensor(
                [word_to_idx.get(w, word_to_idx["<UNK>"]) for w in words[-3:]]
            ).unsqueeze(0).to(self.device)
        return " ".join(words)

    def run(self):
        loss_per_epoch = self.train_model(self.data['train']['X'],self.data['train']['y'])
        epochs = np.arange(len(loss_per_epoch))
        plt.plot(epochs, loss_per_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("LSTM Training Loss Curve")
        plt.grid(True)
        plt.savefig("result/stage_4_result/LSTM_generator_loss_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        print('--start testing...')
        samples = [
            self.generate(
                self.data['vocab']['word_to_idx'],
                self.data['vocab']['idx_to_word'],
                "what did the"
            ),
            self.generate(
                self.data['vocab']['word_to_idx'],
                self.data['vocab']['idx_to_word'],
                "why don't the"
            )
        ]
        return {'generated_text': samples}