import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class SystemMonitor:
    def __init__(self):
        """Initialize system-wide metric tracking"""
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        
    def record_step(self, available_count, busy_count, inactive_count, 
                   current_orders, decisions, rewards):
        """
        Record metrics for a single simulation step with enhanced courier tracking
        
        Args:
            available_count (int): Number of available active couriers
            busy_count (int): Number of busy active couriers
            inactive_count (int): Number of inactive couriers
            current_orders (pd.DataFrame): Current order information
            decisions (list): List of delivery decisions made
            rewards (list): List of rewards received
        """
        # Courier availability metrics
        total_couriers = available_count + busy_count + inactive_count
        
        self.metrics['n_available_couriers'].append(available_count)
        self.metrics['n_busy_couriers'].append(busy_count)
        self.metrics['n_inactive_couriers'].append(inactive_count)
        
        # Calculate true availability ratios
        active_couriers = available_count + busy_count
        if active_couriers > 0:
            availability_ratio = available_count / active_couriers
        else:
            availability_ratio = 0
            
        system_coverage = active_couriers / total_couriers if total_couriers > 0 else 0
        
        self.metrics['courier_availability_ratio'].append(availability_ratio)
        self.metrics['system_coverage_ratio'].append(system_coverage)
        
        # Order and decision metrics
        n_orders = len(current_orders)
        n_direct = sum(1 for d in decisions if d == 0)
        n_split = sum(1 for d in decisions if d == 1)
        
        self.metrics['n_orders'].append(n_orders)
        self.metrics['n_direct_deliveries'].append(n_direct)
        self.metrics['n_split_deliveries'].append(n_split)
        
        # Performance metrics
        mean_reward = np.mean(rewards) if rewards else 0
        self.metrics['rewards'].append(mean_reward)

        self.metrics['reward_sum'].append(np.sum(rewards))
        self.metrics['decision_count'].append(len(decisions))
        
        # Calculate delivery pressure (orders per available courier)
        delivery_pressure = (n_orders / available_count 
                           if available_count > 0 else float('inf'))
        self.metrics['delivery_pressure'].append(delivery_pressure)
        
    def end_episode(self):
        """Record episode summary and reset step metrics"""
        for key, values in self.metrics.items():
            if values:
                self.episode_metrics[f'avg_{key}'].append(np.mean(values))
                self.episode_metrics[f'max_{key}'].append(np.max(values))
                self.episode_metrics[f'min_{key}'].append(np.min(values))
        
        total_reward_sum = sum(self.metrics['reward_sum'])
        total_decision_count = sum(self.metrics['decision_count'])

        # Calculate episode-level metrics
        total_orders = sum(self.metrics['n_orders'])
        total_deliveries = (sum(self.metrics['n_direct_deliveries']) + 
                          sum(self.metrics['n_split_deliveries']))
        
        if total_decision_count > 0:
            avg_reward_per_decision = total_reward_sum / total_decision_count
        else:
            avg_reward_per_decision = 0
            
        self.episode_metrics['avg_reward_per_decision'].append(avg_reward_per_decision)
        
        if total_orders > 0:
            delivery_success_rate = total_deliveries / total_orders
            self.episode_metrics['delivery_success_rate'].append(delivery_success_rate)
        
        # Reset step metrics for next episode
        self.metrics = defaultdict(list)
        
    def get_current_metrics(self):
        """Return current system state metrics for use in decision making"""
        if not self.metrics['courier_availability_ratio']:
            return {
                'availability_ratio': 1.0,
                'delivery_pressure': 0.0,
                'system_coverage': 1.0
            }
            
        return {
            'availability_ratio': self.metrics['courier_availability_ratio'][-1],
            'delivery_pressure': self.metrics['delivery_pressure'][-1],
            'system_coverage': self.metrics['system_coverage_ratio'][-1]
        }


class PolicyAnalyzer:
    def __init__(self, agent, state_handler):
        self.agent = agent
        self.state_handler = state_handler
        
    def analyze_distance_policy(self, n_points=20):
        """Fixed policy analysis"""
        distances = np.linspace(0, 1, n_points)
        courier_ratios = [0.3, 0.5, 0.7]
        
        results = []
        for dist in distances:
            for n_couriers in courier_ratios:
                state = self.state_handler.discretize_state((dist, n_couriers))
                q_values = self.agent.q_table[state]
                action = np.argmax(q_values)
                
                # Calculate confidence based on Q-value difference
                confidence = abs(q_values[0] - q_values[1])
                if np.sum(np.abs(q_values)) > 0:
                    confidence /= np.sum(np.abs(q_values))
                
                results.append({
                    'distance': dist,
                    'n_couriers': n_couriers,
                    'action': action,  # 0 for direct, 1 for split
                    'confidence': confidence,
                    'q_values': q_values
                })
        
        return results

def plot_training_progress(monitor, policy_analyzer):
    """Fixed visualization"""
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, 3)
    title_font = {'fontsize': 20, 'fontweight': 'bold'}
    tick_font = {'fontsize': 20}
    legend_font_size = 20
    tick_font_size = 20

    
    # Plot 1: Courier Utilization
    ax1 = fig.add_subplot(gs[0, 0])
    episodes = range(len(monitor.episode_metrics['avg_n_available_couriers']))
    ax1.plot(episodes, monitor.episode_metrics['avg_n_available_couriers'], 
             label='Available', color='blue')
    ax1.plot(episodes, monitor.episode_metrics['avg_n_busy_couriers'], 
             label='Busy', color='red')
    ax1.set_title('Courier Utilization', fontdict=title_font)
    ax1.set_xlabel('Episode', fontdict=tick_font)
    ax1.set_ylabel('Number of Couriers', fontdict=tick_font)
    ax1.legend()
    # Set x- and y-axis ticks to be larger
    ax1.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # Make legend font size bigger
    ax1.legend(prop={'size': legend_font_size})

    # Plot 2: Delivery Decisions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, monitor.episode_metrics['avg_n_direct_deliveries'], 
             label='Direct', color='blue')
    ax2.plot(episodes, monitor.episode_metrics['avg_n_split_deliveries'], 
             label='Split', color='red')
    ax2.set_title('Delivery Decisions', fontdict=title_font)
    ax2.set_xlabel('Episode', fontdict=tick_font)
    ax2.set_ylabel('Number of Deliveries', fontdict=tick_font)
    ax2.legend(prop={'size': legend_font_size})
    ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # Plot 3: Average Rewards
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(episodes, monitor.episode_metrics['avg_rewards'])
    ax3.set_title('Average Rewards', fontdict=title_font)
    ax3.set_xlabel('Episode', fontdict=tick_font)
    ax3.set_ylabel('Reward', fontdict=tick_font)
    ax3.tick_params(axis='both', which='major', labelsize=tick_font_size)
    
    # Plot 4: Policy Analysis
    ax4 = fig.add_subplot(gs[1, :])
    policy_results = policy_analyzer.analyze_distance_policy()
    
    # Separate plots for each courier ratio
    for n_couriers in [0.3, 0.5, 0.7]:
        subset = [r for r in policy_results if r['n_couriers'] == n_couriers]
        distances = [r['distance'] for r in subset]
        y_values = [n_couriers] * len(subset)
        colors = ['blue' if r['action'] == 0 else 'red' for r in subset]
        sizes = [r['confidence'] * 300 for r in subset]  # Increased size scaling
        
        ax4.scatter(distances, y_values, c=colors, s=sizes, alpha=0.6,
                   label=f'{n_couriers} courier ratio')
    
    ax4.set_title('Learned Policy (Blue: Direct, Red: Split, Size: Confidence)', fontdict=title_font)
    ax4.set_xlabel('Normalized Distance', fontdict=tick_font)
    ax4.set_ylabel('Courier Availability Ratio', fontdict=tick_font)
    ax4.legend(prop={'size': legend_font_size})
    ax4.tick_params(axis='both', which='major', labelsize=tick_font_size)

    plt.tight_layout()
    return fig

def print_policy_summary(policy_analyzer):
    """Enhanced policy summary"""
    results = policy_analyzer.analyze_distance_policy()
    
    print("\nLearned Policy Summary:")
    print("-" * 50)
    
    for n_couriers in [0.3, 0.5, 0.7]:
        subset = [r for r in results if r['n_couriers'] == n_couriers]
        print(f"\nCourier Ratio {n_couriers:.1f}:")
        
        # Find distance threshold where policy switches
        distances = sorted(list(set([r['distance'] for r in subset])))
        actions = [next(r['action'] for r in subset if r['distance'] == d) for d in distances]
        
        # Find switch points
        switch_points = []
        for i in range(1, len(actions)):
            if actions[i] != actions[i-1]:
                switch_points.append(distances[i])
        
        if switch_points:
            print(f"  Policy switches at distances: {', '.join([f'{d:.2f}' for d in switch_points])}")
        else:
            print(f"  Consistent policy: {'Direct' if actions[0] == 0 else 'Split'}")
        
        # Print average confidence
        avg_confidence = np.mean([r['confidence'] for r in subset])
        print(f"  Average confidence: {avg_confidence:.2f}")
