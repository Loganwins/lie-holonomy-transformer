#!/usr/bin/env python3
"""LHT + LoRA Training"""
import torch
from torch.utils.data import DataLoader, Dataset
from lht import LieHolonomyTransformer, LHTConfig, LHTTrainer

class ReasoningDataset(Dataset):
    def __init__(self, n=1000, seq_len=64, vocab=32000):
        self.data = [torch.randint(0, vocab, (seq_len,)) for _ in range(n)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return {"input_ids": self.data[i], "labels": self.data[i]}

if __name__ == "__main__":
    config = LHTConfig(vocab_size=32000, d_model=256, d_fiber=32, n_heads=4, n_layers=4)
    model = LieHolonomyTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trainer = LHTTrainer(model, optimizer, config)
    loader = DataLoader(ReasoningDataset(), batch_size=32, shuffle=True)
    
    for epoch in range(10):
        loss = sum(trainer.train_step(b)["loss"] for b in loader) / len(loader)
        print(f"Epoch {epoch+1}: {loss:.4f}")
    
    torch.save(model.state_dict(), "lht/lht_weights.pt")
    print("Saved to lht/lht_weights.pt")
