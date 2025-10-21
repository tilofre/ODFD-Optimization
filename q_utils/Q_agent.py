# Description: Enhanced RL agent with state handler and reward calculator

import numpy as np
from collections import deque, defaultdict
import copy
import h3
import ppo_utils.abm3 as abm3


class QLearningAgent:
    """
    Here we define the Agent for the Q-learning approach
    """
    def __init__(self, actions_n=2, learning_rate=0.2, discount_factor=0.95, epsilon=1):
        self.q_table = defaultdict(default_q_value)
        self.actions = actions_n #split vs direct
        self.lr = learning_rate #learning rate for q-value
        self.gamma = discount_factor #for future values
        self.epsilon = epsilon #for exploration (greedy)
        self.action_counts = defaultdict(default_action_counts) #count actions for plotting
        
    def get_action(self, state, couriers=None):
        # If the value for 'state' is still a float, it has never been seen.
        if isinstance(self.q_table[state], float):
            # Create an array of Q-values for all actions
            self.q_table[state] = np.zeros(self.actions)

        # The rest of the function can now safely assume q_table[state] is an array
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.actions)
        else:
            action = np.argmax(self.q_table[state])
        
        self.action_counts[state][action] += 1
        return action
    
    #Learn zpdate the q-table
    def learn(self, state, action, reward, next_state):
        if isinstance(self.q_table[state], float):
            self.q_table[state] = np.zeros(self.actions)
        if isinstance(self.q_table[next_state], float):
            self.q_table[next_state] = np.zeros(self.actions)

        # Now the following access is safe
        current_q = self.q_table[state][action] #take the old Q-value
        next_max_q = np.max(self.q_table[next_state]) #calculate the next q-value
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q) #This is bellmans value function
        self.q_table[state][action] = new_q #update q value
    
    #use epsilon decay for exploration at the beginning
    def decrease_epsilon(self, decay=0.995):
        self.epsilon = max(0.05, self.epsilon * decay)
    
    def get_copy(self):
        return copy.deepcopy(self)

class StateRepresentation:
    """
    It is responsible for converting the state of the H3 Hexagon simulation
    into a normalised vector that the RL agent understands.
    """
    def __init__(self, couriers, min_lat, min_lon, max_lat, max_lon, window_size=10):
        """
        Initialises the state handler.
        
        Args:
            couriers (list): The list of your courier objects.
            min_lat, min_lon, max_lat, max_lon (float): The boundaries of the
                                                        area of operation for normalisation.
            window_size (int): The size of the window for historical features.
        """
        # We do not store a grid list, but rather the boundaries of the area.
        self.couriers = couriers
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon
        
        # Rolling Windows
        self.order_history = deque(maxlen=window_size)
        self.delivery_time_history = deque(maxlen=window_size)

    def normalize_location(self, h3_index):
        """
        Normalises the locations for order features
        """
        if not h3_index or not isinstance(h3_index, str):
            return 0.0, 0.0
            
        try:
            lat, lon = h3.cell_to_latlng(h3_index)
            norm_lat = (lat - self.min_lat) / (self.max_lat - self.min_lat)
            norm_lon = (lon - self.min_lon) / (self.max_lon - self.min_lon)
            return norm_lat, norm_lon
        except:
            return 0.0, 0.0

    def get_courier_features(self):
        """
        Extracts courier features. The logic is identical,
        but calls our new normalisation function. The number of stackings and so on
        """
        features = []
        for courier in self.couriers:
            # Uses the normalisation function
            norm_lat, norm_lon = self.normalize_location(courier.position)
            norm_active_deliveries = min(courier.active_deliveries / 5.0, 1.0)

            if courier.mandatory_stops:
                next_stop_hex = courier.mandatory_stops[0][0]
                next_stop_lat, next_stop_lon = self.normalize_location(next_stop_hex)
                has_next_stop = 1.0
            else:
                next_stop_lat, next_stop_lon = 0.0, 0.0
                has_next_stop = 0.0
            
            features.extend([norm_lat, norm_lon, norm_active_deliveries, has_next_stop, next_stop_lat, next_stop_lon])
            
        return np.array(features)

    def get_order_features(self, current_orders):
        """
        Extracts order features. Logic uses
        data fields (“sender_h3”) and H3 normalisation.
        """
        features = []
        self.order_history.append(len(current_orders))
        
        # Logic for order density
        order_density = sum(self.order_history) / self.order_history.maxlen if self.order_history else 0
        norm_order_density = min(order_density / 20.0, 1.0)
        
        #Observe 10 orders 
        max_orders = 10
        for i in range(min(len(current_orders), max_orders)):
            order = current_orders[i]
            
            # Get the restaurant and customer location
            rest_lat, rest_lon = self.normalize_location(order['sender_h3'])
            cust_lat, cust_lon = self.normalize_location(order['recipient_h3'])
            
            current_time = order['platform_order_time']
            prep_time = (order['estimate_meal_prepare_time'] - current_time) / 3600.0
            delivery_time = (order['estimate_arrived_time'] - current_time) / 3600.0
            
            features.extend([
                rest_lat, rest_lon, cust_lat, cust_lon,
                min(prep_time, 2.0) / 2.0, min(delivery_time, 2.0) / 2.0
            ])
            
        # Padding, if less than 10 orders, filled with zeros
        num_features_per_order = 6
        padding_needed = (max_orders * num_features_per_order) - len(features)
        if padding_needed > 0:
            features.extend([0.0] * padding_needed)
            
        features.append(norm_order_density)
        return np.array(features)

    def get_system_features(self):
        """
        Get system features, such as average delivery time, total active deliveries and utilization
        """
        if self.delivery_time_history:
            avg_delivery_time = sum(self.delivery_time_history) / len(self.delivery_time_history)
            norm_avg_delivery_time = min(avg_delivery_time / 7200.0, 1.0)
        else:
            norm_avg_delivery_time = 0.0
            
        total_active_deliveries = sum(c.active_deliveries for c in self.couriers)
        courier_utilization = total_active_deliveries / len(self.couriers) if self.couriers else 0
        norm_courier_utilization = min(courier_utilization / 3.0, 1.0)
        
        return np.array([norm_avg_delivery_time, norm_courier_utilization])

    def get_state(self, current_orders):
        """Get all the different features for the model"""
        courier_features = self.get_courier_features()
        order_features = self.get_order_features(current_orders)
        system_features = self.get_system_features()
        
        return np.concatenate([courier_features, order_features, system_features])


    


class EnhancedRewardCalculator:
    """
    The class and its functions calculates a differentiated reward that takes into account success, delivery time,
    the chosen strategy (split/direct) and current system utilisation
    .
    """
    def __init__(self, base_reward=1.0, max_h3_distance=1346):
        """
        Initialises the Reward Calculator.

        Args:
            base_reward (float): The base value for the reward.
            max_h3_distance (int): The maximum distance in hexagons used for
                                   normalisation (should match the
                                   value in H3StateHandler).
        """
        self.base_reward = base_reward
        self.max_distance = max_h3_distance

    def calculate_reward(self, success, order, delivery_time, action, state_features):
        """
        Calculates the reward based on action, distance and success
        """
        reward = self.base_reward if success else -self.base_reward
        
        # Delay penalty (normalized by 5 minutes = 300 seconds)

        delay_midpoint = 600.0
        steepness = 0.02


        delay = max(0, (order['platform_order_time'] + delivery_time) - order['estimate_arrived_time'])
        
        if delay > 1200:
            return -self.base_reward
            
        sigmoid_penalty = 1.0 / (1.0 + np.exp(-steepness*(delay-delay_midpoint)))
        #Using sigmoid penalty, if an order is later than 20 minutes it is determined as unsuccesful
        reward = reward * (1-sigmoid_penalty)
        
        norm_distance = state_features[0]
        # Distance-based modifiers
        if norm_distance >= 0.5 and action == 1:
            reward += self.base_reward * 0.0
        elif norm_distance <= 0.2 and action == 1:
            # Penalty for unnecessary short-distance splits
            reward -= self.base_reward * 0.5
                
        return reward
    

def default_q_value():#we start with a value of 0
    return 0.0

def default_action_counts():
    return np.zeros(2)