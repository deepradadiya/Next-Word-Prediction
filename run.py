import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import sentencepiece as spm
import numpy as np
import math
from collections import Counter
import os
import pickle
from torch.optim.lr_scheduler import LambdaLR

# Vocabulary class with SentencePiece
class SPVocabulary:
    def __init__(self, model_prefix="spm_model", vocab_size=10000):
        self.model_prefix = model_prefix
        self.sp = spm.SentencePieceProcessor()
        self.vocab_size = vocab_size

    def train(self, texts, model_type="bpe"):
        with open("temp_corpus.txt", "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
        spm.SentencePieceTrainer.train(
            input="temp_corpus.txt",
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type=model_type,
            user_defined_symbols=["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        )
        self.sp.load(f"{self.model_prefix}.model")
        os.remove("temp_corpus.txt")

    def tokenize(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

    def get_vocab_size(self):
        return self.sp.get_piece_size()

# Load pretrained GloVe embeddings
def load_glove_embeddings(glove_path, vocab, embedding_dim=300):
    embeddings = np.random.uniform(-0.1, 0.1, (vocab.get_vocab_size(), embedding_dim))
    word2idx = {vocab.sp.id_to_piece(i): i for i in range(vocab.get_vocab_size())}
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                embeddings[word2idx[word]] = np.array([float(v) for v in parts[1:]])
                found += 1
    print(f"Loaded {found} GloVe embeddings")
    return torch.tensor(embeddings, dtype=torch.float)

# Dataset with dynamic padding
class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_len):
        self.vocab = vocab
        self.max_len = max_len
        self.data = []
        for text in texts:
            tokens = [vocab.sp.piece_to_id("<SOS>")] + vocab.tokenize(text) + [vocab.sp.piece_to_id("<EOS>")]
            if len(tokens) <= max_len + 1:
                self.data.append(tokens)
            else:
                for i in range(0, len(tokens) - max_len, max_len):
                    self.data.append(tokens[i:i + max_len + 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        input_seq = tokens[:-1]
        target_seq = tokens[1:]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Transformer Decoder Model
class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, glove_embeddings=None):
        super().__init__()
        self.d_model = d_model
        if glove_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(glove_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.init_weights()

    def init_weights(self):
        if not hasattr(self.embedding, 'weight'):
            initrange = 0.1
            self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-0.1, 0.1)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.layer_norm(src)
        if tgt is None:  # Inference mode
            output = self.transformer_decoder(src, src, tgt_mask=src_mask)
        else:  # Training mode
            tgt = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt)
            tgt = self.layer_norm(tgt)
            output = self.transformer_decoder(tgt, src, tgt_mask=tgt_mask, memory_mask=src_mask)
        output = self.fc_out(self.dropout(output))
        return output

# Learning rate warmup
def get_lr_scheduler(optimizer, warmup_steps=1000):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

# Training function with mixed precision and checkpointing
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    scaler = GradScaler()
    model.train()
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = generate_square_subsequent_mask(src.size(1)).to(device)
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(src, tgt, src_mask, tgt_mask)
                output = output.view(-1, output.size(-1))
                tgt = tgt.view(-1)
                loss = criterion(output, tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_loss, top1_acc, top5_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Top-1 Acc: {top1_acc:.4f}, Top-5 Acc: {top5_acc:.4f}")
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, "best_model.pth"))

# Evaluation function with perplexity and top-k accuracy
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = generate_square_subsequent_mask(src.size(1)).to(device)
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
            output = model(src, tgt, src_mask, tgt_mask)
            output = output.view(-1, output.size(-1))
            tgt_flat = tgt.view(-1)
            loss = criterion(output, tgt_flat)
            total_loss += loss.item()
            
            # Top-k accuracy
            _, top_indices = output.topk(5, dim=-1)
            top1_correct += (top_indices[:, 0] == tgt_flat).sum().item()
            top5_correct += sum([tgt_flat[i] in top_indices[i] for i in range(tgt_flat.size(0))]).item()
            total_samples += tgt_flat.size(0)
    
    avg_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_loss)
    top1_acc = top1_correct / total_samples
    top5_acc = top5_correct / total_samples
    model.train()
    return perplexity, top1_acc, top5_acc

# Beam search decoding
def beam_search(model, vocab, input_text, seq_length, device, beam_width=3, max_len=20):
    model.eval()
    input_tokens = [vocab.sp.piece_to_id("<SOS>")] + vocab.tokenize(input_text)
    input_tokens = input_tokens[-seq_length:]
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
    beams = [(input_tensor, 0.0, input_tokens)]  # (sequence, score, tokens)
    completed = []
    
    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            for seq, score, tokens in beams:
                if tokens[-1] == vocab.sp.piece_to_id("<EOS>"):
                    completed.append((seq, score, tokens))
                    continue
                src_mask = generate_square_subsequent_mask(seq.size(1)).to(device)
                output = model(seq, src_mask=src_mask)
                probs = torch.softmax(output[0, -1, :], dim=-1)
                top_probs, top_indices = torch.topk(probs, beam_width)
                for prob, idx in zip(top_probs, top_indices):
                    new_tokens = tokens + [idx.item()]
                    new_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                    new_score = score + math.log(prob.item())
                    new_beams.append((new_seq, new_score, new_tokens))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if not beams:
                break
        if beams:
            completed.extend(beams)
    
    best_seq = sorted(completed, key=lambda x: x[1], reverse=True)[0]
    return vocab.decode(best_seq[2])

# Generate square subsequent mask
def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# Main execution
if __name__ == "__main__":
    # Sample dataset (replace with WikiText-103 or BooksCorpus for production)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Transformers are powerful models for natural language processing",
        "Deep learning enables computers to learn from data",
        "Neural networks can model complex patterns effectively",
    ]
    val_texts = [
        "The fast red fox leaps over obstacles",
        "Artificial intelligence powers modern technology",
    ]

    # Hyperparameters
    SEQ_LENGTH = 20
    BATCH_SIZE = 16
    D_MODEL = 300  # Match GloVe embedding dimension
    NHEAD = 6
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1200
    DROPOUT = 0.1
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    WARMUP_STEPS = 1000

    # Initialize vocabulary and dataset
    vocab = SPVocabulary(vocab_size=10000)
    vocab.train(sample_texts + val_texts)
    train_dataset = TextDataset(sample_texts, vocab, SEQ_LENGTH)
    val_dataset = TextDataset(val_texts, vocab, SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Load GloVe embeddings (replace with path to glove.6B.300d.txt)
    glove_path = "glove.6B.300d.txt"  # Download from https://nlp.stanford.edu/projects/glove/
    glove_embeddings = load_glove_embeddings(glove_path, vocab, D_MODEL) if os.path.exists(glove_path) else None

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextWordPredictor(
        vocab_size=vocab.get_vocab_size(),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        glove_embeddings=glove_embeddings
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.sp.piece_to_id("<PAD>"))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, NUM_EPOCHS)

    # Example beam search prediction
    input_text = "The quick brown"
    predicted_sequence = beam_search(model, vocab, input_text, SEQ_LENGTH, device)
    print(f"\nInput: {input_text}")
    print(f"Predicted Sequence: {predicted_sequence}")