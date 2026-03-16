import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class ThresholdEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    The agent adjusts the classification threshold to maximize the Macro-F1 score.
    """
    def __init__(self, y_true, y_probs):
        super(ThresholdEnv, self).__init__()
        self.y_true = y_true
        self.y_probs = y_probs
        
        # Action space: 0 = decrease threshold, 1 = keep same, 2 = increase threshold
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: The current threshold value (0.0 to 1.0)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.threshold = 0.5

    def step(self, action):
        # Apply action to threshold
        if action == 0:
            self.threshold -= 0.02
        elif action == 2:
            self.threshold += 0.02
            
        # Keep threshold within valid probability bounds
        self.threshold = np.clip(self.threshold, 0.01, 0.99)
        
        # Generate predictions using the new threshold
        preds = (self.y_probs >= self.threshold).astype(int)
        
        # Reward is the Macro-F1 score
        reward = f1_score(self.y_true, preds, average='macro')
        
        # For this stub, an episode is a single step evaluation
        done = True
        truncated = False
        
        return np.array([self.threshold], dtype=np.float32), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.threshold = 0.5
        return np.array([self.threshold], dtype=np.float32), {}

def run_rl_stub():
    print("Initializing RL Agent Stub...")
    
    # 1. Create Dummy Validation Data (Simulating DistilBERT probabilities)
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    # Simulate a model that is somewhat accurate but needs tuning
    y_probs = np.where(y_true == 1, np.random.uniform(0.3, 0.9, 1000), np.random.uniform(0.1, 0.7, 1000))

    # 2. Setup the Environment
    env = ThresholdEnv(y_true, y_probs)
    
    episodes = 50
    rewards = []
    
    # 3. Run a random agent to generate "early noisy learning curves"
    for ep in range(episodes):
        obs, _ = env.reset()
        action = env.action_space.sample() # Take a random action
        obs, reward, done, _, _ = env.step(action)
        
        rewards.append(reward)

    # 4. Plot and Save the Results
    # Ensure the experiments/results directory exists (run from project root)
    save_dir = os.path.join("experiments", "results")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, episodes + 1), rewards, marker='o', linestyle='-', color='b', alpha=0.6)
    plt.title('RL Agent Stub: Reward (Macro-F1) vs Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Macro-F1 Score')
    plt.grid(True)
    
    save_path = os.path.join(save_dir, "rl_learning_curve.png")
    plt.savefig(save_path)
    print(f"Success! Noisy learning curve saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_rl_stub()