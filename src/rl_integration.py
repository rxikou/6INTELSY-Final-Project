"""
RL Integration: Threshold Optimization with Trained Model
Trains an RL agent to find optimal classification threshold for DistilBERT.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import gymnasium as gym
import json

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.distilbert_model import DistilBERTClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_validation_data():
    """Load validation data."""
    data_dir = Path(__file__).parent.parent / "data"
    val_df = pd.read_csv(data_dir / "cleaned_val.csv")
    return val_df


def get_bert_probabilities(model, texts, tokenizer):
    """Get probability predictions from DistilBERT model."""
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    dataset = TensorDataset(
        encodings["input_ids"].to(device),
        encodings["attention_mask"].to(device)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            output = model(input_ids, attention_mask=attention_mask)
            logits = output["logits"] if isinstance(output, dict) else output
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_probs)


class ThresholdOptimizationEnv(gym.Env):
    """
    Custom environment for threshold optimization.
    Action space: adjust threshold by discrete steps.
    Reward: Macro-F1 score on validation data.
    """
    def __init__(self, y_true, y_probs, step_size=0.02):
        super().__init__()
        self.y_true = y_true
        self.y_probs = y_probs[:, 1]  # Use prob of positive class
        self.step_size = step_size
        self.best_threshold = 0.5
        self.best_f1 = 0.0
        
        # Action space: 0=decrease, 1=no-op, 2=increase
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation: current threshold
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.threshold = 0.5
        
    def step(self, action):
        # Apply action
        if action == 0:
            self.threshold = max(0.01, self.threshold - self.step_size)
        elif action == 2:
            self.threshold = min(0.99, self.threshold + self.step_size)
        
        # Get predictions with current threshold
        preds = (self.y_probs >= self.threshold).astype(int)
        
        # Calculate reward
        f1 = f1_score(self.y_true, preds, average='macro', zero_division=0)
        
        # Track best
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_threshold = self.threshold
        
        reward = f1
        done = False
        truncated = False
        
        return (
            np.array([self.threshold], dtype=np.float32),
            reward,
            done,
            truncated,
            {"f1": f1, "best_f1": self.best_f1, "best_threshold": self.best_threshold}
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.threshold = 0.5
        return np.array([self.threshold], dtype=np.float32), {}


def train_rl_agent(env, episodes=100, random_prob=0.3):
    """
    Simple epsilon-greedy RL agent to optimize threshold.
    Strategy: Explore random actions, track best threshold found.
    """
    print(f"\nTraining RL Agent ({episodes} episodes)...")
    
    rewards_history = []
    best_f1_history = []
    best_threshold_found = 0.5
    best_f1_found = 0.0
    
    obs, _ = env.reset()
    
    for episode in range(episodes):
        # Epsilon-greedy: random action with prob, else best known
        if np.random.random() < random_prob:
            action = env.action_space.sample()
        else:
            # Exploit: move towards best threshold
            if obs[0] < best_threshold_found:
                action = 2  # increase
            elif obs[0] > best_threshold_found:
                action = 0  # decrease
            else:
                action = 1  # no-op
        
        obs, reward, done, _, info = env.step(action)
        
        rewards_history.append(reward)
        best_f1_history.append(info["best_f1"])
        best_threshold_found = info["best_threshold"]
        best_f1_found = info["best_f1"]
        
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode+1:3d}: F1={reward:.4f}, Best={best_f1_found:.4f} (threshold={best_threshold_found:.3f})")
    
    return rewards_history, best_f1_history, best_threshold_found, best_f1_found


def plot_learning_curve(rewards_history, best_f1_history, best_threshold):
    """Plot and save learning curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reward per episode
    ax1.plot(range(len(rewards_history)), rewards_history, 'b-', alpha=0.6, label='Episode F1')
    ax1.axhline(y=max(best_f1_history), color='r', linestyle='--', label=f'Best F1: {max(best_f1_history):.4f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Macro-F1 Score')
    ax1.set_title('RL Agent: Reward per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Best F1 over time
    ax2.plot(range(len(best_f1_history)), best_f1_history, 'g-', linewidth=2, label='Best F1 Found')
    ax2.axhline(y=max(best_f1_history), color='r', linestyle='--', label=f'Optimal: {max(best_f1_history):.4f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Best Macro-F1 Score')
    ax2.set_title(f'RL Optimization Progress (Best Threshold: {best_threshold:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "rl_integration_learning_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved learning curves: {save_path}")


def main():
    """Main RL integration pipeline."""
    print("\n" + "="*70)
    print("RL INTEGRATION: THRESHOLD OPTIMIZATION")
    print("="*70)
    
    # Load validation data
    print("\n[1/5] Loading validation data...")
    val_df = load_validation_data()
    y_val = val_df["label"].values
    X_val = val_df["statement"].reset_index(drop=True)
    print(f"Validation set: {len(val_df)} samples")
    print(f"Classes: FAKE={np.sum(y_val==0)}, REAL={np.sum(y_val==1)}")
    
    # Load model
    print("\n[2/5] Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBERTClassifier(num_classes=2, dropout_rate=0.2, label_smoothing=0.15)
    
    try:
        state = torch.load("best_bert.pt", map_location=device)
        if "criterion.weight" in state:
            del state["criterion.weight"]
        model.load_state_dict(state, strict=False)
    except:
        model.load_state_dict(torch.load("best_bert.pt", map_location=device), strict=False)
    
    model.to(device)
    model.eval()
    print("[OK] Model loaded")
    
    # Get probabilities
    print("\n[3/5] Generating probability predictions...")
    y_probs = get_bert_probabilities(model, X_val, tokenizer)
    default_f1 = f1_score(y_val, (y_probs[:, 1] >= 0.5).astype(int), average='macro')
    print(f"Default F1 (threshold=0.5): {default_f1:.4f}")
    
    # Setup and train RL agent
    print("\n[4/5] Setting up RL environment...")
    env = ThresholdOptimizationEnv(y_val, y_probs, step_size=0.01)
    
    rewards_history, best_f1_history, best_threshold, best_f1 = train_rl_agent(
        env, episodes=200, random_prob=0.25
    )
    
    improvement = best_f1 - default_f1
    print(f"\n[RESULTS]")
    print(f"  Default threshold (0.5):     F1 = {default_f1:.4f}")
    print(f"  Optimized threshold ({best_threshold:.3f}): F1 = {best_f1:.4f}")
    print(f"  Improvement:                 +{improvement:.4f} ({100*improvement/default_f1:+.2f}%)")
    
    # Save results
    print("\n[5/5] Saving results...")
    
    # Plot curves
    plot_learning_curve(rewards_history, best_f1_history, best_threshold)
    
    # Save optimal threshold config
    config = {
        "model": "DistilBERT",
        "default_threshold": 0.5,
        "optimized_threshold": float(best_threshold),
        "default_f1": float(default_f1),
        "optimized_f1": float(best_f1),
        "improvement": float(improvement),
        "improvement_percent": float(100 * improvement / default_f1),
        "episodes": 200,
        "validation_samples": len(y_val),
    }
    
    config_path = RESULTS_DIR / "rl_threshold_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Saved threshold config: {config_path}")
    
    # Save history
    history_df = pd.DataFrame({
        "episode": range(len(rewards_history)),
        "episode_f1": rewards_history,
        "best_f1": best_f1_history,
    })
    
    history_path = RESULTS_DIR / "rl_optimization_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"[OK] Saved optimization history: {history_path}")
    
    print("\n" + "="*70)
    print("[DONE] RL integration complete!")
    print("="*70)
    print(f"\nOptimal threshold: {best_threshold:.4f}")
    print(f"Use this threshold for inference to maximize Macro-F1 on similar data.\n")


if __name__ == "__main__":
    main()
