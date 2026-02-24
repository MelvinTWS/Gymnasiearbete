"""
RL Agent Training Module

This module trains a Deep Q-Network (DQN) agent to learn optimal air defense
strategies against UAV swarm attacks using Stable-Baselines3.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import time

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    CheckpointCallback, 
    EvalCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_environment import AirDefenseEnv


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback for tracking training metrics.
    
    Logs episode rewards, penetration rates, cost-exchange ratios,
    and other key metrics during training.
    """
    
    def __init__(self, verbose: int = 0):
        """
        Initialize the callback.
        """
        super().__init__(verbose)
        
        # Storage for metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.penetration_rates = []
        self.cost_exchange_ratios = []
        self.defense_costs = []
        self.swarm_sizes = []
        self.timestamps = []
        
        # Current episode tracking
        self.episode_count = 0
        self.episode_start_time = None
    
    def _on_training_start(self) -> None:
        """Called before training starts."""
        self.episode_start_time = time.time()
    
    def _on_step(self) -> bool:
        """
        Called at each step. Check if episode ended and log metrics.
        """
        # Check if episode ended
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    # Get info from environment
                    info = self.locals['infos'][idx]
                    
                    # Extract episode metrics
                    if 'episode_length' in info:
                        self.episode_count += 1
                        
                        # Store metrics
                        self.episode_rewards.append(info.get('episode_reward', 0))
                        self.episode_lengths.append(info.get('episode_length', 0))
                        self.penetration_rates.append(info.get('penetration_rate', 0))
                        self.cost_exchange_ratios.append(info.get('cost_exchange_ratio', 0))
                        self.defense_costs.append(info.get('total_defense_cost', 0))
                        self.swarm_sizes.append(info.get('swarm_size', 0))
                        self.timestamps.append(time.time())
                        
                        # Print progress every 100 episodes
                        if self.verbose >= 1 and self.episode_count % 100 == 0:
                            recent_pen = np.mean(self.penetration_rates[-100:])
                            recent_cost_ex = np.mean(self.cost_exchange_ratios[-100:])
                            recent_reward = np.mean(self.episode_rewards[-100:])
                            
                            print(f"Episode {self.episode_count:5d}: "
                                  f"Avg Reward={recent_reward:>12,.0f}, "
                                  f"Pen={recent_pen*100:>5.2f}%, "
                                  f"CostEx={recent_cost_ex:>6.2f}")
        
        return True
    
    def get_metrics(self) -> Dict[str, List]:
        """
        Get all collected metrics.
        """
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'penetration_rates': self.penetration_rates,
            'cost_exchange_ratios': self.cost_exchange_ratios,
            'defense_costs': self.defense_costs,
            'swarm_sizes': self.swarm_sizes,
            'timestamps': self.timestamps,
            'episode_count': self.episode_count
        }


def train_dqn_agent(
    total_timesteps: int = 1_000_000,
    learning_rate: float = 0.0001,
    buffer_size: int = 100_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    exploration_fraction: float = 0.9,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.01,
    target_update_interval: int = 1000,
    swarm_size_range: tuple = (100, 500),
    save_dir: str = "trained_models",
    model_name: str = "dqn_air_defense",
    checkpoint_freq: int = 50_000,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 10,
    random_seed: int = 42,
    verbose: int = 1
) -> tuple:
    """
    Train a DQN agent for air defense.
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}"
    
    if verbose >= 1:
        print("=" * 70)
        print("DQN AGENT TRAINING")
        print("=" * 70)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Learning rate: {learning_rate}")
        print(f"Buffer size: {buffer_size:,}")
        print(f"Batch size: {batch_size}")
        print(f"Gamma: {gamma}")
        print(f"Exploration: {exploration_initial_eps} -> {exploration_final_eps}")
        print(f"Target update interval: {target_update_interval}")
        print(f"Swarm size range: {swarm_size_range}")
        print(f"Save directory: {save_path}")
        print(f"Random seed: {random_seed}")
        print("=" * 70)
        print()
    
    # Create training environment
    env = AirDefenseEnv(
        swarm_size_range=swarm_size_range,
        random_seed=random_seed
    )
    env = Monitor(env)  # Wrap with Monitor for automatic logging
    
    # Create evaluation environment
    eval_env = AirDefenseEnv(
        swarm_size_range=swarm_size_range,
        random_seed=random_seed + 1000  # Different seed for eval
    )
    eval_env = Monitor(eval_env)
    
    # Configure DQN agent
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,  # Start learning after 1000 steps
        batch_size=batch_size,
        gamma=gamma,
        train_freq=4,  # Train every 4 steps
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=dict(net_arch=[256, 256]),  # 2 hidden layers, 256 units each
        verbose=verbose,
        seed=random_seed,
        tensorboard_log=str(save_path / "tensorboard_logs")
    )
    
    # Setup callbacks
    callbacks = []
    
    # Metrics callback
    metrics_callback = TrainingMetricsCallback(verbose=verbose)
    callbacks.append(metrics_callback)
    
    # Checkpoint callback (save model periodically)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_path / "checkpoints"),
        name_prefix=model_filename,
        verbose=verbose
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(save_path / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=False,  # Use stochastic policy for realistic eval
        verbose=verbose
    )
    callbacks.append(eval_callback)
    
    # Train the agent
    if verbose >= 1:
        print("Starting training...")
        print()
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100 if verbose >= 1 else None,
            progress_bar=True
        )
    except KeyboardInterrupt:
        if verbose >= 1:
            print("\n\nTraining interrupted by user!")
    
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = save_path / f"{model_filename}_final"
    model.save(str(final_model_path))
    
    if verbose >= 1:
        print()
        print("=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Final model saved to: {final_model_path}")
        print("=" * 70)
        print()
    
    # Get training metrics
    training_metrics = metrics_callback.get_metrics()
    
    # Save training metrics to JSON
    metrics_path = save_path / f"{model_filename}_metrics.json"
    save_training_metrics(training_metrics, metrics_path, training_time)
    
    # Generate and save learning curves (skip plotting to avoid PIL issues on some systems)
    # if len(training_metrics['episode_rewards']) > 0:
    #     plot_path = save_path / f"{model_filename}_learning_curves.png"
    #     try:
    #         plot_learning_curves(training_metrics, plot_path)
    #         
    #         if verbose >= 1:
    #             print(f"Learning curves saved to: {plot_path}")
    #     except Exception as e:
    #         if verbose >= 1:
    #             print(f"Warning: Could not save learning curves plot: {e}")
    #             print("Continuing without plot...")
    
    # Save training configuration
    config = {
        'total_timesteps': total_timesteps,
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'gamma': gamma,
        'exploration_fraction': exploration_fraction,
        'exploration_initial_eps': exploration_initial_eps,
        'exploration_final_eps': exploration_final_eps,
        'target_update_interval': target_update_interval,
        'swarm_size_range': swarm_size_range,
        'random_seed': random_seed,
        'training_time': training_time,
        'timestamp': timestamp
    }
    
    config_path = save_path / f"{model_filename}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    if verbose >= 1:
        print(f"Training config saved to: {config_path}")
        print()
    
    return model, training_metrics, str(final_model_path)


def save_training_metrics(
    metrics: Dict[str, Any],
    filepath: Path,
    training_time: float
) -> None:
    """
    Save training metrics to JSON file.
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            serializable_metrics[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                         for v in value]
        else:
            serializable_metrics[key] = value
    
    serializable_metrics['training_time_seconds'] = training_time
    
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


def plot_learning_curves(
    metrics: Dict[str, List],
    save_path: Path,
    window_size: int = 100
) -> None:
    """
    Plot and save learning curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Progress', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(metrics['episode_rewards']) + 1)
    
    # Helper function for moving average
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, metrics['episode_rewards'], alpha=0.3, label='Raw')
    if len(metrics['episode_rewards']) >= window_size:
        smoothed = moving_average(metrics['episode_rewards'], window_size)
        ax1.plot(range(window_size, len(episodes) + 1), smoothed, 
                linewidth=2, label=f'{window_size}-episode MA')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episode Rewards (Higher is Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Penetration Rate
    ax2 = axes[0, 1]
    pen_rates_pct = [p * 100 for p in metrics['penetration_rates']]
    ax2.plot(episodes, pen_rates_pct, alpha=0.3, label='Raw')
    if len(pen_rates_pct) >= window_size:
        smoothed = moving_average(pen_rates_pct, window_size)
        ax2.plot(range(window_size, len(episodes) + 1), smoothed, 
                linewidth=2, label=f'{window_size}-episode MA')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Penetration Rate (%)')
    ax2.set_title('UAV Penetration Rate (Lower is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cost-Exchange Ratio
    ax3 = axes[1, 0]
    ax3.plot(episodes, metrics['cost_exchange_ratios'], alpha=0.3, label='Raw')
    if len(metrics['cost_exchange_ratios']) >= window_size:
        smoothed = moving_average(metrics['cost_exchange_ratios'], window_size)
        ax3.plot(range(window_size, len(episodes) + 1), smoothed, 
                linewidth=2, label=f'{window_size}-episode MA')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cost-Exchange Ratio')
    ax3.set_title('Cost-Exchange Ratio (Lower is Better)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Defense Cost
    ax4 = axes[1, 1]
    defense_costs_M = [c / 1_000_000 for c in metrics['defense_costs']]
    ax4.plot(episodes, defense_costs_M, alpha=0.3, label='Raw')
    if len(defense_costs_M) >= window_size:
        smoothed = moving_average(defense_costs_M, window_size)
        ax4.plot(range(window_size, len(episodes) + 1), smoothed, 
                linewidth=2, label=f'{window_size}-episode MA')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Defense Cost (Million USD)')
    ax4.set_title('Defense Cost per Episode')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_trained_model(model_path: str) -> DQN:
    """
    Load a trained DQN model.
    """
    model_path = Path(model_path)
    if model_path.suffix != '.zip':
        model_path = model_path.with_suffix('.zip')
    
    return DQN.load(model_path)


if __name__ == "__main__":
    model, metrics, path = train_dqn_agent(
        total_timesteps=1_000_000,
        save_dir="thesis_results/trained_models",
        model_name="dqn_agent",
        random_seed=42,
        verbose=1
    )
    print(f"Training complete. Model saved to: {path}")
