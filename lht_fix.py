#!/usr/bin/env python3
"""
Fix the holonomy explosion in lht.py
Problem: Matrix exponentials explode with random init
Solution: Constrain the Lie algebra outputs
"""

import torch
import torch.nn as nn

# Patch to apply to ConnectionNetwork in lht.py
# The connection output needs to be bounded

def bounded_connection_forward(self, x):
    """
    Bounded version: constrain A to small values
    so exp(A) doesn't explode.
    """
    batch, seq_len, d_model = x.shape
    
    A_flat = self.connection_mlp(x)
    A = A_flat.view(batch, seq_len, d_model, self.k, self.k)
    
    # KEY FIX: Constrain to small values with tanh
    A = 0.1 * torch.tanh(A)
    
    # Make antisymmetric (so(k) instead of gl(k)) - more stable
    A = 0.5 * (A - A.transpose(-1, -2))
    
    return A

print("Patching lht.py to fix holonomy explosion...")

# Read current lht.py
with open('lht.py', 'r') as f:
    content = f.read()

# Find and replace the connection forward method
old_code = '''        # Optional: Project to so(k) for orthogonal structure (antisymmetric)
        # A = 0.5 * (A - A.transpose(-1, -2))
        
        return A'''

new_code = '''        # STABILITY FIX: Constrain to small values
        A = 0.1 * torch.tanh(A)
        
        # Project to so(k) for orthogonal structure (antisymmetric) - more stable
        A = 0.5 * (A - A.transpose(-1, -2))
        
        return A'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('lht.py', 'w') as f:
        f.write(content)
    print("✓ Patched ConnectionNetwork - bounded output with tanh")
else:
    print("⚠ Could not find exact code to patch. Manual edit needed.")
    print("\nIn lht.py, find ConnectionNetwork.forward() and add before 'return A':")
    print("    A = 0.1 * torch.tanh(A)")
    print("    A = 0.5 * (A - A.transpose(-1, -2))")

print("\nNow run: python lht_train_real.py")
