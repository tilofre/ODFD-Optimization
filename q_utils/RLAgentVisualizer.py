import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import h3


class RLAgentVisualizer:
    """
    Visualization tool for RL agent decisions and policies.
    Designed to integrate with existing ABM/RL food delivery simulation.
    """
    
    def __init__(self, agent, state_handler, grid, monitor=None):
        """
        Initialize the visualizer with components from the main simulation
        
        Args:
            agent: QLearningAgent instance
            state_handler: EnhancedStateHandler instance
            grid: City grid
            monitor: Optional SystemMonitor for additional metrics
        """
        self.agent = agent
        self.state_handler = state_handler
        self.grid = grid
        self.monitor = monitor
        
        # Boundary coordinates for visualization
        # Boundary coordinates for visualization
        if grid is not None:
            self.min_lat, self.min_lon = grid[0].P1['lat'], grid[0].P1['lon']
            self.max_lat, self.max_lon = grid[4919].P3['lat'], grid[4919].P3['lon']
        else:
            # Setze Standardwerte, wenn kein Grid übergeben wird
            self.min_lat, self.min_lon, self.max_lat, self.max_lon = 0, 0, 46.000000, 174.700000
        
        # Initialize containers for decisions and examples
        self.decision_history = []
        self.direct_examples = []  # Store examples of direct deliveries
        self.split_examples = []   # Store examples of split deliveries
        self.courier_paths = defaultdict(dict)  # Track courier paths for visualization


    def record_decision(self, order, state_features, state, action, reward, couriers=None):
        """
        Record a decision made by the agent for later visualization
        
        Args:
            order: Order information 
            state_features: Extracted state features (distance, availability)
            state: Discretized state
            action: Action taken (0=direct, 1=split)
            reward: Reward received
            couriers: List of available couriers
        """
        rest_lat, rest_lon = h3.cell_to_latlng(order['sender_h3'])
        cust_lat, cust_lon = h3.cell_to_latlng(order['recipient_h3'])

        decision_info = {
            'time': order.get('platform_order_time', 0),
            'order_id': order.get('order_id', 'unknown'),
            'rest_coords': [rest_lon, rest_lat], # <-- Speichere die konvertierten Koordinaten
            'cust_coords': [cust_lon, cust_lat], # <-- Speichere die konvertierten Koordinaten
            'distance': state_features[0],
            'courier_avail': state_features[1],
            'state': state,
            'action': action,
            'action_name': 'split' if action == 1 else 'direct',
            'reward': reward,
            'q_values': self.agent.q_table[state].copy(),
            'confidence': abs(self.agent.q_table[state][0] - self.agent.q_table[state][1]) / \
                        (np.sum(np.abs(self.agent.q_table[state])) + 1e-10)
        }
        
        self.decision_history.append(decision_info)
        
        # If this is a particularly clear decision (high confidence), store as example
        if decision_info['confidence'] > 0.5:
            if action == 0:  # Direct delivery
                self.direct_examples.append(decision_info)
            else:  # Split delivery
                self.split_examples.append(decision_info)
    
    def record_courier_position(self, courier, current_time):
        """
        Record a courier's position for path visualization
        
        Args:
            courier: Courier instance
            current_time: Current simulation time
        """
        self.courier_paths[courier.id][current_time] = courier.position.copy()

    # def visualize_policy(self, save_path=None):
    #     """
    #     Visualize the learned policy as a function of distance and courier availability,
    #     using simple bin numbers for the x-axis labels.
    #     """
    #     # Create figure
    #     fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    #     fontsize = 20

    #     # Policy heatmap: Extract policy from q-table
    #     n_dist_bins = self.state_handler.n_distance_bins
    #     n_courier_bins = self.state_handler.n_courier_bins
        
    #     distances = np.linspace(0, 1, n_dist_bins)
    #     utilization_ratios = np.linspace(0, 1, n_courier_bins)

    #     policy_map = np.zeros((len(utilization_ratios), len(distances)))
    #     confidence_map = np.zeros((len(utilization_ratios), len(distances)))

    #     for i, util_ratio in enumerate(utilization_ratios):
    #         for j, dist in enumerate(distances):
    #             availability = 1 - util_ratio
    #             state = self.state_handler.discretize_state((dist, availability))

    #             q_values = self.agent.q_table[state]
    #             policy_map[i, j] = np.argmax(q_values)

    #             if np.sum(np.abs(q_values)) > 0:
    #                 confidence_map[i, j] = abs(q_values[0] - q_values[1]) / np.sum(np.abs(q_values))
    #             else:
    #                 confidence_map[i, j] = 0

    #     # --- ANPASSUNG FÜR DIE X-ACHSENBESCHRIFTUNG (BIN-NUMMERN) ---

    #     # 1. Tick-Positionen sind einfach die Indizes der Bins (0, 1, 2, ...)
    #     tick_positions = np.arange(n_dist_bins)
        
    #     # 2. Labels sind die Bin-Nummern (1, 2, 3, ...)
    #     tick_labels = [str(i + 1) for i in range(n_dist_bins)]

    #     # Plot the policy heatmap
    #     ax = axes[0]
    #     heatmap = ax.imshow(policy_map, cmap='coolwarm', aspect='auto', origin='lower')
    #     ax.set_xlabel('Distance Bin', fontsize=fontsize) # Titel angepasst
    #     ax.set_ylabel('Utilization Ratio', fontsize=fontsize)
    #     cbar = fig.colorbar(heatmap, ax=ax, ticks=[0, 1])
    #     cbar.ax.set_yticklabels(['Direct', 'Split'], fontsize=fontsize)
        
    #     # Die neuen Ticks und Labels setzen
    #     ax.set_xticks(tick_positions)
    #     ax.set_xticklabels(tick_labels, fontsize=fontsize)
        
    #     # Y-Achse bleibt unverändert
    #     ax.set_yticks(np.linspace(0, len(utilization_ratios)-1, 5))
    #     ax.set_yticklabels([f'{a:.2f}' for a in np.linspace(0, 1, 5)], fontsize=fontsize)
    #     ax.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    #     # Plot confidence in the policy
    #     ax = axes[1]
    #     confidence_heatmap = ax.imshow(confidence_map, cmap='viridis', aspect='auto', origin='lower')
    #     ax.set_xlabel('Distance Bin', fontsize=fontsize) # Titel angepasst
    #     cbar = fig.colorbar(confidence_heatmap, ax=ax)
    #     cbar.set_label('Confidence', fontsize=fontsize)
    #     cbar.ax.yaxis.set_tick_params(labelsize=fontsize)
        
    #     # Die neuen Ticks und Labels auch hier setzen
    #     ax.set_xticks(tick_positions)
    #     ax.set_xticklabels(tick_labels, fontsize=fontsize)
        
    #     # Y-Achse bleibt unverändert
    #     ax.set_yticks(np.linspace(0, len(utilization_ratios)-1, 5))
    #     ax.set_yticklabels([f'{a:.2f}' for a in np.linspace(0, 1, 5)], fontsize=fontsize)
    #     ax.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')

    #     return fig
    
    def visualize_policy(self, save_path=None):
        """
        Visualize the learned policy as a function of distance and courier availability.
        """
    # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fontsize = 20

        # Policy heatmap: Extract policy from q-table
        distances = np.linspace(0, 1, 10)
        utilization_ratios = np.linspace(0, 1, 10)

        policy_map = np.zeros((len(utilization_ratios), len(distances)))
        confidence_map = np.zeros((len(utilization_ratios), len(distances)))

        # GEÄNDERT: Die Schleifenvariable heißt jetzt 'util_ratio' für mehr Klarheit
        for i, util_ratio in enumerate(utilization_ratios):
            for j, dist in enumerate(distances):

                # KORREKTUR: Wandle die Utilization Ratio in die Availability um,
                # bevor der Zustand erstellt wird.
                availability = 1 - util_ratio
                state = self.state_handler.discretize_state((dist, availability))

                q_values = self.agent.q_table[state]
                policy_map[i, j] = np.argmax(q_values)

                if np.sum(np.abs(q_values)) > 0:
                    confidence_map[i, j] = abs(q_values[0] - q_values[1]) / np.sum(np.abs(q_values))
                else:
                    confidence_map[i, j] = 0

        # Der restliche Code zum Plotten bleibt exakt gleich.
        # Da wir `origin='lower'` verwenden, wird die y-Achse korrekt von unten nach oben gezeichnet.
        # Plot the policy heatmap
        ax = axes[0]
        heatmap = ax.imshow(policy_map, cmap='coolwarm', aspect='auto', origin='lower')
        ax.set_xlabel('Distance (Normalized)', fontsize=fontsize)
        ax.set_ylabel('Utilization Ratio', fontsize=fontsize)
        cbar = fig.colorbar(heatmap, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Direct', 'Split'], fontsize=fontsize)
        ax.set_xticks(np.linspace(0, len(distances)-1, 5))
        ax.set_xticklabels([f'{d:.1f}' for d in np.linspace(0, 1, 5)], fontsize=fontsize)
        ax.set_yticks(np.linspace(0, len(utilization_ratios)-1, 5))
        ax.set_yticklabels([f'{a:.2f}' for a in np.linspace(0, 1, 5)], fontsize=fontsize)
        ax.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        # Plot confidence in the policy
        ax = axes[1]
        confidence_heatmap = ax.imshow(confidence_map, cmap='viridis', aspect='auto', origin='lower')
        ax.set_xlabel('Distance (Normalized)', fontsize=fontsize)
        cbar = fig.colorbar(confidence_heatmap, ax=ax)
        cbar.set_label('Confidence', fontsize=fontsize)
        cbar.ax.yaxis.set_tick_params(labelsize=fontsize)
        ax.set_xticks(np.linspace(0, len(distances)-1, 5))
        ax.set_xticklabels([f'{d:.1f}' for d in np.linspace(0, 1, 5)], fontsize=fontsize)  
        ax.set_yticks(np.linspace(0, len(utilization_ratios)-1, 5))
        ax.set_yticklabels([f'{a:.2f}' for a in np.linspace(0, 1, 5)], fontsize=fontsize)
        ax.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
    
    def visualize_decision_distribution(self, save_path=None):
        """
        Visualize the distribution of decisions based on distance and courier availability
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            fig: The generated figure
        """
        if not self.decision_history:
            print("No decision history available")
            return None
            
        # Create DataFrame from decision history
        df = pd.DataFrame(self.decision_history)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distance vs Decisions scatter plot
        ax = axes[0, 0]
        sns.scatterplot(
            data=df, 
            x='distance', 
            y='courier_avail', 
            hue='action_name',
            style='action_name',
            palette=['blue', 'red'],
            s=df['confidence'] * 100 + 20,  # Size based on confidence
            alpha=0.7,
            ax=ax
        )
        ax.set_title('Decisions by State', fontsize=14)
        ax.set_xlabel('Distance (Normalized)', fontsize=12)
        ax.set_ylabel('Utilization Ratio', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 2. Distance vs Reward scatter plot
        ax = axes[0, 1]
        sns.scatterplot(
            data=df, 
            x='distance', 
            y='reward', 
            hue='action_name',
            style='action_name',
            palette=['blue', 'red'],
            s=50,
            alpha=0.7,
            ax=ax
        )
        ax.set_title('Rewards by Distance and Action', fontsize=14)
        ax.set_xlabel('Distance (Normalized)', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 3. Decision counts by distance bins
        ax = axes[1, 0]
        df['distance_bin'] = pd.cut(df['distance'], bins=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        decision_counts = df.groupby(['distance_bin', 'action_name']).size().unstack(fill_value=0)
        decision_counts.plot(kind='bar', ax=ax)
        ax.set_title('Decision Counts by Distance Range', fontsize=14)
        ax.set_xlabel('Distance Range', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 4. Average reward by distance bin and action
        ax = axes[1, 1]
        reward_by_distance = df.groupby(['distance_bin', 'action_name'])['reward'].mean().unstack(fill_value=0)
        reward_by_distance.plot(kind='bar', ax=ax)
        ax.set_title('Average Reward by Distance Range and Action', fontsize=14)
        ax.set_xlabel('Distance Range', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_example_decisions(self, save_path=None):
        """
        Visualize specific examples of direct and split deliveries to explain policy
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            fig: The generated figure
        """
        # Check if we have examples
        if not self.direct_examples and not self.split_examples:
            print("No example decisions available")
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 1. Direct delivery example
        ax = axes[0]
        if self.direct_examples:
            example = self.direct_examples[-1]  # Use the most recent example
            self._plot_delivery_example(ax, example, "Direct Delivery Example")
        else:
            ax.text(0.5, 0.5, "No direct delivery examples available", 
                    ha='center', va='center', fontsize=14)
        
        # 2. Split delivery example
        ax = axes[1]
        if self.split_examples:
            example = self.split_examples[-1]  # Use the most recent example
            self._plot_delivery_example(ax, example, "Split Delivery Example")
        else:
            ax.text(0.5, 0.5, "No split delivery examples available", 
                    ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_delivery_example(self, ax, example, title):
        """Helper method to plot a specific delivery example"""
        # Set axis limits and title
        ax.set_xlim(self.min_lon, self.max_lon)
        ax.set_ylim(self.min_lat, self.max_lat)
        ax.set_title(title, fontsize=14)
        
        # Draw restaurant and customer locations
        rest_coords = example['rest_coords']
        cust_coords = example['cust_coords']
        
        ax.plot(rest_coords[0], rest_coords[1], 's', color='green', markersize=10)
        ax.plot(cust_coords[0], cust_coords[1], 'p', color='purple', markersize=10)
        
        # Add labels
        ax.text(rest_coords[0], rest_coords[1], 'Restaurant', fontsize=9, 
                verticalalignment='bottom')
        ax.text(cust_coords[0], cust_coords[1], 'Customer', fontsize=9, 
                verticalalignment='bottom')
        
        # Add direct route
        ax.plot(
            [rest_coords[0], cust_coords[0]], 
            [rest_coords[1], cust_coords[1]], 
            '--', color='blue' if example['action'] == 0 else 'red', alpha=0.7
        )
        
        # If it's a split delivery, also show the meeting point
        if example['action'] == 1:
            # Calculate meeting point (midpoint)
            meeting_point = [
                (rest_coords[0] + cust_coords[0]) / 2,
                (rest_coords[1] + cust_coords[1]) / 2
            ]
            
            ax.plot(meeting_point[0], meeting_point[1], 'h', color='orange', markersize=10)
            ax.text(meeting_point[0], meeting_point[1], 'Meeting Point', fontsize=9, 
                   verticalalignment='bottom')
        
        # Add decision information as text
        info_text = (
            f"Distance: {example['distance']:.2f} (normalized)\n"
            f"Courier Availability: {example['courier_avail']:.2f}\n"
            f"Action: {example['action_name'].upper()}\n"
            f"Q-values: Direct={example['q_values'][0]:.2f}, "
            f"Split={example['q_values'][1]:.2f}\n"
            f"Confidence: {example['confidence']:.2f}\n"
            f"Reward: {example['reward']:.2f}"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    def visualize_learning_progress(self, save_path=None,reward_metric='total_performance',window_size=30):
        """
        Visualize the learning progress over episodes using monitor data
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            fig: The generated figure
        """
        if self.monitor is None:
            print("No monitor data available")
            return None
            
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average rewards over episodes
        ax = axes[0, 0]

        if reward_metric == 'per_decision':
            metric_key = 'avg_reward_per_decision'
            plot_title = 'Average Reward per Decision'
        else: # Standardfall
            metric_key = 'avg_rewards'
            plot_title = 'Average Reward per Step (Overall Performance)'
        rewards = self.monitor.episode_metrics.get(metric_key, [])

        episodes = range(len(rewards))

        ax.plot(episodes, rewards, color='gray', alpha=0.4, label='Reward pro Episode')
        if window_size > 1 and len(rewards) > 0:
            reward_series = pd.Series(rewards)
            smoothed_rewards = reward_series.rolling(window=window_size, min_periods=1).mean()
            ax.plot(episodes, smoothed_rewards, color='blue', linewidth=2.5, label=f'Gleitender Durchschnitt (ws={window_size})')
            ax.set_title(f'Trend: {plot_title}', fontsize=14)
        else:
            ax.set_title(plot_title, fontsize=14)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Direct vs Split deliveries
        ax = axes[0, 1]
        direct = self.monitor.episode_metrics.get('avg_n_direct_deliveries', [])
        split = self.monitor.episode_metrics.get('avg_n_split_deliveries', [])
        
        if direct and split:
            ax.plot(episodes, direct, label='Direct', color='blue')
            ax.plot(episodes, split, label='Split', color='red')
            ax.set_title('Delivery Decisions per Episode', fontsize=14)
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Number of Deliveries', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No delivery decision data available", 
                    ha='center', va='center', fontsize=14)
        
        # 3. Courier utilization
        ax = axes[1, 0]
        available = self.monitor.episode_metrics.get('avg_n_available_couriers', [])
        busy = self.monitor.episode_metrics.get('avg_n_busy_couriers', [])
        
        if available and busy:
            ax.plot(episodes, available, label='Available', color='blue')
            ax.plot(episodes, busy, label='Busy', color='red')
            ax.set_title('Courier Utilization', fontsize=14)
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Number of Couriers', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No courier utilization data available", 
                    ha='center', va='center', fontsize=14)
        
        # 4. Policy exploration (epsilon)
        ax = axes[1, 1]
        # We don't directly have epsilon history, so show confidence instead
        if self.decision_history:
            df = pd.DataFrame(self.decision_history)
            df_by_time = df.groupby(pd.cut(df.index, bins=min(20, len(df))))['confidence'].mean()
            
            time_points = range(len(df_by_time))
            ax.plot(time_points, df_by_time.values)
            ax.set_title('Average Decision Confidence Over Time', fontsize=14)
            ax.set_xlabel('Time (binned)', fontsize=12)
            ax.set_ylabel('Average Confidence', fontsize=12)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No decision history available", 
                    ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def draw_episode_summary(self, episode_num, timestart, timeend, total_orders, decisions_made,
                            direct_count, split_count, avg_reward, save_path=None):
        """
        Create a summary visualization for a specific episode
        
        Args:
            episode_num: Episode number
            timestart: Start time of episode
            timeend: End time of episode
            total_orders: Total number of orders processed
            decisions_made: Number of decisions made
            direct_count: Number of direct deliveries
            split_count: Number of split deliveries
            avg_reward: Average reward in the episode
            save_path: Optional path to save the figure
            
        Returns:
            fig: The generated figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 1. Policy visualization
        self.visualize_policy(show_confidence=True, save_path=None)
        policy_fig = plt.gcf()
        plt.close(policy_fig)
        
        fontsize = 20
        
        # Get the first axis from the policy figure
        policy_ax = policy_fig.axes[0]
        axes[0].imshow(policy_ax.images[0].get_array(), cmap='coolwarm', aspect='auto', origin='lower')
        axes[0].set_title('Learned Policy (Blue=Direct, Red=Split)', fontsize=fontsize)
        axes[0].set_xlabel('Distance (Normalized)', fontsize=fontsize)
        axes[0].set_ylabel('Utilization Ratio', fontsize=fontsize)
        
        # Set custom x and y tick labels
        # The y-axis goes from 0 to 0.3, and the x-axis goes from 0 to 1
        axes[0].set_xticks(np.linspace(0, 10, 5), fontsize=fontsize)
        axes[0].set_xticklabels([f'{d:.1f}' for d in np.linspace(0, 1, 5)], fontsize=fontsize)
        axes[0].set_yticks(np.linspace(0, 10, 5), fontsize=fontsize)
        axes[0].set_yticklabels([f'{a:.2f}' for a in np.linspace(0, 0.3, 5)], fontsize=fontsize)
        
        # 2. Decision summary
        ax = axes[1]
        
        # Create a summary pie chart
        if direct_count + split_count > 0:
            ax.pie(
                [direct_count, split_count],
                labels=['Direct', 'Split'],
                colors=['blue', 'red'],
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(width=0.5)  # Make it a donut chart
            )
            ax.set_title('Decision Distribution', fontsize=14)
        else:
            ax.text(0.5, 0.5, "No decisions made in this episode", 
                    ha='center', va='center', fontsize=14)
        
        # Add episode summary text
        summary_text = (
            f"Episode {episode_num}\n"
            f"Duration: {(timeend - timestart) / 60:.1f} minutes\n"
            f"Total Orders: {total_orders}\n"
            f"Decisions Made: {decisions_made}\n"
            f"  - Direct: {direct_count}\n"
            f"  - Split: {split_count}\n"
            f"Average Reward: {avg_reward:.2f}\n"
            f"Current Epsilon: {self.agent.epsilon:.3f}"
        )
        
        fig.text(0.5, 0.01, summary_text, ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                fontsize=12)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        fig.suptitle(f"Episode {episode_num} Summary", fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# def update_courier_status(couriers, current_time):
#     """
#     Update courier status based on completed deliveries

#     Args:
#         couriers: List of active couriers
#         current_time: Current simulation time

#     Returns:
#         int: Number of couriers that were released from deliveries
#     """
#     released_count = 0

#     for courier in couriers:
#         completed_stops = [
#             (time) for time, stop in courier.mandatory_stops.items()
#             if time <= current_time and stop[2] == 'C'
#         ]

#         for stop_time in completed_stops:
#             del courier.mandatory_stops[stop_time]
#             if courier.active_deliveries > 0:
#                 courier.active_deliveries -= 1
#                 released_count += 1

#     return released_count

def integrate_visualizer(run_abm_func):
    """
    Create a wrapper around the run_abm function to record visualization data
    
    Args:
        run_abm_func: Original run_abm function
        
    Returns:
        function: Wrapped run_abm function that records visualization data
    """
    def wrapped_run_abm(current_time, steps, data, couriers, v, state_handler, agent, 
                        reward_calculator, max_active_deliveries=3, max_delay_threshold=1800, visualizer=None):
        # Call the original function
        updated_data, reward, decisions = run_abm_func(
            current_time, steps, data, couriers, v, state_handler, agent, reward_calculator, max_active_deliveries, max_delay_threshold
        )
        
        # If visualizer is provided, record additional data
        if visualizer is not None:
            # Get current orders that were processed
            current_orders = data[
                (data['platform_order_time'] >= current_time) &
                (data['platform_order_time'] < current_time + steps)
            ]
            
            # Record courier positions
            for courier in couriers:
                visualizer.record_courier_position(courier, current_time)
            
            # Record decisions (this is an approximation since we don't have direct access)
            for i, (_, order) in enumerate(current_orders.iterrows()):
                if i < len(decisions):
                    action = decisions[i]
                    
                    # Extract state features
                    state_features = state_handler.get_state_features(
                        order.to_dict(), couriers
                    )
                    state = state_handler.discretize_state(state_features)
                    
                    # Record the decision
                    visualizer.record_decision(
                        order.to_dict(), state_features, state, action, reward / len(decisions) if len(decisions) > 0 else 0
                    )
        
        return updated_data, reward, decisions
    
    return wrapped_run_abm
