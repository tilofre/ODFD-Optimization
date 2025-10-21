import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h3
from collections import defaultdict

class RLAgentVisualizer:
    """
    Visualization tool for PPO agent decisions and policies.
    """
    
    def __init__(self, agent, state_handler, grid=None, monitor=None):
        """
        Initialize the visualizer.
        
        Args:
            agent: PPOAgent instance.
            state_handler: EnhancedStateHandler instance.
            grid: Optional city grid for geographic plots.
            monitor: Optional SystemMonitor for additional metrics.
        """
        self.agent = agent
        self.state_handler = state_handler
        self.grid = grid
        self.monitor = monitor
        
        # Initialize containers for decisions
        self.decision_history = []

    def record_decision(self, state_features, action, reward, action_probs):
        """
        Record a decision made by the agent for later visualization.
        PPO-compatible.

        Args:
            state_features (np.array): The state vector [distance, utilization, queue, courier_count].
            action (int): Action taken (0=direct, 1=split).
            reward (float): Reward received.
            action_probs (np.array): The output probabilities from the actor network.
        """
        # Confidence is the absolute difference between the probabilities of the two actions.
        confidence = np.abs(action_probs[0] - action_probs[1])

        decision_info = {
            'distance': state_features[0],
            'system_stress': (state_features[1] * 0.4 + state_features[2] * 0.4 + (1.0 - state_features[3]) * 0.2),
            'action': action,
            'action_name': 'Split' if action == 1 else 'Direct',
            'reward': reward,
            'confidence': confidence,
            'prob_split': action_probs[1] # Probability for "Split"
        }
        
        self.decision_history.append(decision_info)

    def visualize_decision_distribution(self, save_path=None):
        """
        Visualizes the distribution of decisions and the agent's confidence.
        
        Args:
            save_path (str, optional): Path to save the figure.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if not self.decision_history:
            print("No decision history available to visualize.")
            return None
            
        df = pd.DataFrame(self.decision_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 15))
        fig.suptitle('Analysis of Agent Decisions', fontsize=20, weight='bold')
        
        # Scatter plot with corrected 'size' parameter and rasterization
        ax = axes[0, 0]
        sns.scatterplot(
            data=df, 
            x='distance', 
            y='system_stress', 
            hue='action_name',
            style='action_name',
            palette={'Direct': 'blue', 'Split': 'red'},
            size='confidence',
            sizes=(20, 200),
            alpha=0.6,
            ax=ax,
            rasterized=True  # Rasterize the scatter points
        )
        ax.set_title('Decisions by State (Size = Confidence)', fontsize=16)
        ax.set_xlabel('Normalized Distance', fontsize=12)
        ax.set_ylabel('System Stress Index', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Action')

        ax = axes[0, 1]
        sns.histplot(data=df, x='confidence', hue='action_name', 
                     palette={'Direct': 'blue', 'Split': 'red'}, 
                     multiple='stack', ax=ax)
        ax.set_title('Distribution of Decision Confidence', fontsize=16)
        ax.set_xlabel('Confidence (Difference in Probabilities)', fontsize=12)
        ax.set_ylabel('Number of Decisions', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        df['distance_bin'] = pd.cut(df['distance'], bins=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        decision_counts = df.groupby(['distance_bin', 'action_name']).size().unstack(fill_value=0)
        decision_counts.plot(kind='bar', ax=ax, colormap='coolwarm', rot=45)
        ax.set_title('Number of Decisions by Distance', fontsize=16)
        ax.set_xlabel('Distance Category', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        ax = axes[1, 1]
        sns.boxplot(data=df, x='distance_bin', y='reward', hue='action_name', ax=ax,
                    palette={'Direct': 'lightblue', 'Split': 'salmon'})
        ax.set_title('Reward Received by Distance and Action', fontsize=16)
        ax.set_xlabel('Distance Category', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='grey', linestyle='--')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig

    def clear_history(self):
        """Resets the decision history for the next analysis period."""
        self.decision_history.clear()

    def visualize_unified_policy(self, ax):
        """
        Visualizes the policy on a 2D heatmap combining system state 
        into a single "System Stress" index using defined weights.
        """
        num_points = 50 # Increased points for a smoother plot
        distance_range = np.linspace(0, 1, num_points)
        max_real_stress = 0.6
        stress_range = np.linspace(0, max_real_stress, num_points)
        
        split_probabilities = np.zeros((len(stress_range), len(distance_range)))

        stress_weights = {'utilization': 0.4, 'queue': 0.4, 'courier_count': 0.2}
        
        assert np.isclose(sum(stress_weights.values()), 1.0), "Stress weights must sum to 1.0"

        for i, stress in enumerate(stress_range):
            for j, dist in enumerate(distance_range):
                norm_utilization = stress
                norm_queue = stress
                norm_courier_count = 1.0 - stress
                state = np.array([dist, norm_utilization, norm_queue, norm_courier_count])
                state_reshaped = np.reshape(state, [1, self.agent.state_dim])
                action_probs = self.agent.actor.predict(state_reshaped, verbose=0)[0]
                split_probabilities[i, j] = action_probs[1]

        # Using pcolormesh with rasterization
        c = ax.pcolormesh(distance_range, stress_range, split_probabilities, 
                          cmap='viridis', vmin=0, vmax=1, shading='gouraud',
                          rasterized=True) # Rasterize the heatmap
        
        ax.set_title('Unified Policy: Decision vs. System Stress', fontsize=14)
        ax.set_xlabel('Normalized Distance', fontsize=12)
        ax.set_ylabel('System Stress Index', fontsize=12)
        
        return c

    def visualize_reward_confidence(self, save_path=None):
        """
        Visualizes the correlation between the agent's confidence 
        and the reward received.
        """
        if not self.decision_history:
            print("No decision history available to visualize.")
            return None
            
        df = pd.DataFrame(self.decision_history)
        
        # lmplot directly creates a figure and is therefore handled a bit differently
        g = sns.lmplot(
            data=df,
            x="confidence",
            y="reward",
            hue="action_name",
            height=6,
            aspect=1.5,
            palette={'Direct': 'blue', 'Split': 'red'},
            legend=False,
            scatter_kws={'alpha': 0.5, 'rasterized': True}  # Rasterize scatter points
        )

        g.ax.set_title('Correlation of Reward and Confidence', fontsize=16, weight='bold')
        g.set_axis_labels("Decision Confidence", "Received Reward")
        g.ax.axhline(0, ls='--', color='grey')
        plt.legend(title='Action', loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        return plt.gcf()

    def visualize_action_probabilities(self, save_path=None):
        """
        Visualizes the distribution of the raw "Split" probabilities,
        separated by the final decision made.
        """
        if not self.decision_history:
            print("No decision history available to visualize.")
            return None
            
        df = pd.DataFrame(self.decision_history)

        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Violin plots are vector-based and generally don't need rasterization
        sns.violinplot(
            data=df,
            x="action_name",
            y="prob_split",
            palette={'Direct': 'lightblue', 'Split': 'salmon'},
            order=['Direct', 'Split'],
            ax=ax
        )

        ax.set_title('Distribution of the "split" probability after final action', fontsize=16, weight='bold')
        ax.set_xlabel('Action actually selected', fontsize=12)
        ax.set_ylabel('Split probability predicted by the model', fontsize=12)
        ax.axhline(0.5, ls='--', color='grey', label='50% Threshold')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        return fig