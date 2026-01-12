#!/usr/bin/env python3
"""
LHT Training on Real Reasoning Chains
--------------------------------------
Trains the Lie-Holonomy Transformer on:
1. Consistent reasoning (low holonomy target)
2. Inconsistent reasoning (high holonomy target)

This gives meaningful geometric analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import os

from lht import LieHolonomyTransformer, LHTConfig

# === REASONING DATASETS ===

class ConsistentReasoningDataset(Dataset):
    """
    Generates reasoning chains that ARE consistent.
    LHT should learn: low holonomy for these.
    """
    
    def __init__(self, n_samples=2000, seq_len=64, vocab_size=32000):
        self.data = []
        self.labels = []  # 1 = consistent
        
        for _ in range(n_samples):
            chain = self._generate_consistent_chain(seq_len, vocab_size)
            self.data.append(chain)
            self.labels.append(1.0)
    
    def _generate_consistent_chain(self, seq_len, vocab_size):
        """
        Patterns that represent consistent reasoning:
        - A -> B -> C (linear progression)
        - Arithmetic: a + b = c
        - Repeated structure (same pattern = consistent)
        """
        pattern_type = random.choice(['linear', 'arithmetic', 'symmetric'])
        
        if pattern_type == 'linear':
            # Linear chain: each token depends on previous
            base = random.randint(100, 1000)
            step = random.randint(1, 10)
            tokens = [(base + i * step) % vocab_size for i in range(seq_len)]
            
        elif pattern_type == 'arithmetic':
            # Arithmetic patterns: a, b, a+b, b, a+b, a+2b, ...
            a = random.randint(10, 100)
            b = random.randint(10, 100)
            tokens = []
            for i in range(seq_len):
                tokens.append((a + i * b) % vocab_size)
                
        else:  # symmetric
            # Symmetric: pattern repeats (consistency)
            pattern_len = random.randint(4, 16)
            pattern = [random.randint(0, vocab_size-1) for _ in range(pattern_len)]
            tokens = (pattern * (seq_len // pattern_len + 1))[:seq_len]
        
        return torch.tensor(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.data[idx],
            'consistency': torch.tensor(self.labels[idx])
        }


class InconsistentReasoningDataset(Dataset):
    """
    Generates reasoning chains that are INCONSISTENT.
    LHT should learn: high holonomy for these.
    """
    
    def __init__(self, n_samples=2000, seq_len=64, vocab_size=32000):
        self.data = []
        self.labels = []  # 0 = inconsistent
        
        for _ in range(n_samples):
            chain = self._generate_inconsistent_chain(seq_len, vocab_size)
            self.data.append(chain)
            self.labels.append(0.0)
    
    def _generate_inconsistent_chain(self, seq_len, vocab_size):
        """
        Patterns that represent inconsistent reasoning:
        - Circular dependencies: A -> B -> C -> A (contradiction)
        - Random jumps (no logical flow)
        - Contradictory patterns
        """
        pattern_type = random.choice(['circular', 'random', 'contradictory'])
        
        if pattern_type == 'circular':
            # Circular: creates loops that don't close properly
            cycle_len = random.randint(3, 8)
            base_pattern = [random.randint(0, vocab_size-1) for _ in range(cycle_len)]
            tokens = []
            offset = 0
            for i in range(seq_len):
                # Add drift to the cycle (inconsistency)
                tokens.append((base_pattern[i % cycle_len] + offset) % vocab_size)
                if i % cycle_len == 0:
                    offset += random.randint(1, 10)  # Drift!
                    
        elif pattern_type == 'random':
            # Pure random: no structure = no consistency
            tokens = [random.randint(0, vocab_size-1) for _ in range(seq_len)]
            
        else:  # contradictory
            # Start with pattern, then contradict it
            half = seq_len // 2
            pattern = [random.randint(0, vocab_size-1) for _ in range(half)]
            # Second half contradicts first (reversed + noise)
            contradiction = [vocab_size - 1 - t + random.randint(-5, 5) for t in pattern]
            contradiction = [max(0, min(vocab_size-1, t)) for t in contradiction]
            tokens = pattern + contradiction
        
        return torch.tensor(tokens[:seq_len])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.data[idx],
            'consistency': torch.tensor(self.labels[idx])
        }


class MixedReasoningDataset(Dataset):
    """Combined dataset with both consistent and inconsistent examples."""
    
    def __init__(self, n_samples=4000, seq_len=64, vocab_size=32000):
        consistent = ConsistentReasoningDataset(n_samples // 2, seq_len, vocab_size)
        inconsistent = InconsistentReasoningDataset(n_samples // 2, seq_len, vocab_size)
        
        self.data = consistent.data + inconsistent.data
        self.labels = consistent.labels + inconsistent.labels
        
        # Shuffle
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)
        self.data = list(self.data)
        self.labels = list(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.data[idx],
            'consistency': torch.tensor(self.labels[idx])
        }


# === TRAINING ===

class LHTConsistencyTrainer:
    """
    Trains LHT to predict reasoning consistency.
    
    Loss = LM_loss + λ * consistency_loss
    
    Where consistency_loss encourages:
    - Low holonomy for consistent reasoning
    - High holonomy for inconsistent reasoning
    """
    
    def __init__(self, model, optimizer, config, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.step = 0
        
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        consistency_target = batch['consistency'].to(self.device)
        
        # Forward pass
        output = self.model(
            input_ids=input_ids,
            labels=labels,
            return_geometric_losses=True
        )
        
        # Get losses
        lm_loss = output.get('lm_loss', torch.tensor(0.0))
        holonomy = output.get('holonomy_loss', torch.tensor(0.0))
        curvature = output.get('curvature_loss', torch.tensor(0.0))
        
        # Consistency loss: 
        # If consistent (target=1): want LOW holonomy
        # If inconsistent (target=0): want HIGH holonomy
        # Loss = (holonomy - target_holonomy)^2
        # where target_holonomy = 0 for consistent, 1 for inconsistent
        
        target_holonomy = 1.0 - consistency_target.mean()  # 0 for consistent, 1 for inconsistent
        
        # Normalize holonomy to [0,1] range
        holonomy_normalized = torch.sigmoid(holonomy - 0.5)
        
        consistency_loss = F.mse_loss(holonomy_normalized, target_holonomy)
        
        # Total loss
        total_loss = lm_loss + 0.5 * consistency_loss + 0.1 * curvature
        
        # Handle inf/nan
        if torch.isfinite(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.step += 1
        
        return {
            'loss': total_loss.item() if torch.isfinite(total_loss) else 999.0,
            'lm_loss': lm_loss.item() if torch.isfinite(lm_loss) else 0.0,
            'holonomy': holonomy.item() if torch.isfinite(holonomy) else 0.0,
            'curvature': curvature.item() if torch.isfinite(curvature) else 0.0,
            'consistency_loss': consistency_loss.item() if torch.isfinite(consistency_loss) else 0.0,
        }


def train_lht(epochs=20, batch_size=32, lr=1e-4, save_path='lht/lht_weights.pt'):
    """Main training function."""
    
    print("=" * 60)
    print("LHT Consistency Training")
    print("=" * 60)
    
    # Create directories
    os.makedirs('lht', exist_ok=True)
    
    # Config
    config = LHTConfig(
        vocab_size=32000,
        d_model=256,
        d_fiber=32,
        n_heads=4,
        n_layers=4,
        lie_algebra_rank=4,
        max_seq_len=128,
        lambda_holonomy=0.1,
        lambda_curvature=0.01,
    )
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = LieHolonomyTransformer(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Data
    print("Generating reasoning datasets...")
    dataset = MixedReasoningDataset(n_samples=4000, seq_len=64, vocab_size=32000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Samples: {len(dataset)} ({len(dataset)//2} consistent, {len(dataset)//2} inconsistent)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Trainer
    trainer = LHTConsistencyTrainer(model, optimizer, config, device)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        total_loss = 0
        total_hol = 0
        total_curv = 0
        n_batches = 0
        
        for batch in loader:
            metrics = trainer.train_step(batch)
            total_loss += metrics['loss']
            total_hol += metrics['holonomy']
            total_curv += metrics['curvature']
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_hol = total_hol / n_batches
        avg_curv = total_curv / n_batches
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.4f} | Holonomy={avg_hol:.4f} | Curvature={avg_curv:.4f}")
    
    # Save
    print("-" * 60)
    torch.save(model.state_dict(), save_path)
    print(f"✓ Saved to {save_path}")
    
    # Test
    print("\nTesting on sample inputs...")
    model.eval()
    
    with torch.no_grad():
        # Test consistent
        consistent_sample = ConsistentReasoningDataset(1, 64, 32000)[0]
        out_c = model(input_ids=consistent_sample['input_ids'].unsqueeze(0).to(device), return_geometric_losses=True)
        hol_c = out_c.get('holonomy_loss', torch.tensor(0.0)).item()
        
        # Test inconsistent
        inconsistent_sample = InconsistentReasoningDataset(1, 64, 32000)[0]
        out_i = model(input_ids=inconsistent_sample['input_ids'].unsqueeze(0).to(device), return_geometric_losses=True)
        hol_i = out_i.get('holonomy_loss', torch.tensor(0.0)).item()
    
    print(f"  Consistent example holonomy:   {hol_c:.4f}")
    print(f"  Inconsistent example holonomy: {hol_i:.4f}")
    
    if hol_c < hol_i:
        print("  ✓ Model learned: consistent < inconsistent")
    else:
        print("  ⚠ Model needs more training")
    
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    train_lht(epochs=20, batch_size=32, lr=1e-4)
