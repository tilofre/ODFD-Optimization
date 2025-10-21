# ppo_agent.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import logging
import os
import h3
from collections import deque
import numpy as np
from collections import deque
import ppo_utils.abm3 as abm3


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class PPOAgent:
    def __init__(self, state_dim, action_dim, total_episodes=200):
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.learning_rate_actor = 0.0001
        self.learning_rate_critic = 0.0001
        self.epochs = 10
        self.gae_lamda = 0.95
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = self._build_actor()
        self.critic = self._build_critic()


        self.entropy_start = 0.1   # start with high entropy, ensuring the weights are not too balanced too soon
        self.entropy_end = 0.005    # end value of entropy, as we are decreasing the value
        self.total_episodes = total_episodes
        self.current_episode = 0    # counter for number of episodes

        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_actor,
            clipnorm=1.0  #the clip value for actor
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_critic,
            clipnorm=1.0  #the clip value for the critic
        )
        
        #Empty lists as our experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

    #Build actor NN with two dense layers with 128 neurons each and relu as activation function
    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)
    #Build critic NN with two dense layers with 256 neurons each and relu as activation function
    def _build_critic(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs)

    #store the values for batch
    def store_transition(self, state, action, reward, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    #clear memory after trained batch
    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.dones.clear()

    #get action from the actors NN by transferring the state (used for the test data set)
    def get_action(self, state, deterministic=False):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor(state_tensor)
        
        if deterministic:
            #deterministic = false 
            action = tf.argmax(action_probs, axis=1).numpy()[0]
        else:
            #choose action based on training (if 0.7 for 0 and 0.3 for 1 and draw 0.8 it is still action 1)
            action = tf.random.categorical(tf.math.log(action_probs), 1).numpy().flatten()[0]
        #save the log probability for the experience buffer and loss calculation
        log_prob = tf.math.log(action_probs[0, action] + 1e-10)
        return action, action_probs.numpy()[0], log_prob.numpy()
    
    #the same as get action, but we use it for our train data set
    def select_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        action_probs = self.actor(state)
        action = np.random.choice(self.action_dim, p=np.squeeze(action_probs))
        log_prob = tf.math.log(action_probs[0, action])
        return action, log_prob, action_probs
    
    #Calculate the advantage based on reward and Value function
    def _calculate_advantages(self, rewards, dones, values):
        last_value = values[-1]
        values = values[:-1]
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t]+self.gamma*last_value*mask-values[t]
            last_adv = delta + self.gamma * self.gae_lamda * last_adv * mask #this is the advantage function
            advantages[t] = last_adv
            last_value = values[t]
        returns = advantages + values
        return advantages, returns
        
    #the entropy to explore more (it is manipulating the probabilities of each action)
    def _get_entropy_coeff(self):
    # Basis-Decay
        fraction = min(1.0, self.current_episode / self.total_episodes)
        coeff = self.entropy_start + fraction * (self.entropy_end - self.entropy_start)
        return coeff

    def save_models(self, directory="ppo_models"):
        """To save the models weights."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.actor.save_weights(os.path.join(directory, "ppo_actor.weights.h5"))
        self.critic.save_weights(os.path.join(directory, "ppo_critic.weights.h5"))

    def load_models(self, directory="ppo_models"):
        """For testing."""
        try:
            self.actor.load_weights(os.path.join(directory, "ppo_actor.weights.h5"))
            self.critic.load_weights(os.path.join(directory, "ppo_critic.weights.h5"))
            print(f"Models'{directory}' loaded ")
        except (FileNotFoundError, OSError):
            print(f"Weights not found {directory}")

    def train(self):
    # training the NNs
        states_arr = np.array(self.states)
        actions_arr = np.array(self.actions)
        rewards_arr = np.array(self.rewards)
        dones_arr = np.array(self.dones)
        log_probs_arr = tf.convert_to_tensor(self.log_probs, dtype=tf.float32)

        # values from critic used for advantage calculation
        states_with_last = np.append(states_arr, [self.states[-1]],axis=0)
        values = self.critic.predict(states_with_last, verbose=0)
        values = np.squeeze(values)

        # Calculate advantage and return 
    
        advantages, returns = self._calculate_advantages(rewards_arr, dones_arr, values)

        # Normalisation important
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        # Get the hyperparameters for update
        batch_size = len(states_arr)
        minibatch_size = max(64, batch_size // 4) #Using mini batches, not always necessary
        target_kl = 0.02  # Using early stopping

        entropy_coeff = self._get_entropy_coeff()
        actor_losses = [] #all from the batch
        critic_losses = []
        entropies = []

        #Now the real training starts
        for ep in range(self.epochs):
            idxs = np.arange(batch_size)
            np.random.shuffle(idxs)
             #Initiate list for losses and entropy for epochs
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_entropies = []
            epoch_kls = []

            for start in range(0, batch_size, minibatch_size):
                mb_idx = idxs[start:start+minibatch_size]#Using minibatches
                #state, action, advantage, return, old log probability
                s_mb = states_arr[mb_idx]
                a_mb = actions_arr[mb_idx]
                adv_mb = tf.gather(advantages, mb_idx)
                ret_mb = tf.gather(returns, mb_idx)
                old_logp_mb = tf.gather(log_probs_arr, mb_idx)

                with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                    current_probs = self.actor(s_mb, training=True) #actor probability
                    entropy = -tf.reduce_sum(current_probs * tf.math.log(current_probs + 1e-8), axis=1) #entropy

                    action_indices = tf.stack([tf.range(len(a_mb)), a_mb], axis=1)
                    current_action_probs = tf.gather_nd(current_probs, action_indices) #collect all probs

                    ratio = current_action_probs / (tf.exp(old_logp_mb) + 1e-8) #calculate with old log probs

                    surr1 = ratio * adv_mb #calculate surrogate function with advantage
                    surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_mb #clip the value

                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2) + entropy_coeff * entropy)#calculate the actor loss based on surrogate
                    critic_values = tf.squeeze(self.critic(s_mb, training=True)) 
                    critic_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(ret_mb, critic_values)) #calculate the critic loss

                # Calculate the gradient for descent
                actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)

                # Use gradient clipping
                actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)
                critic_grads, _ = tf.clip_by_global_norm(critic_grads, 0.5)

                #Train the models
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                # Approximate with KL
                new_logp_mb = tf.math.log(current_action_probs + 1e-8)
                approx_kl = tf.reduce_mean(old_logp_mb - new_logp_mb).numpy()

                epoch_actor_losses.append(actor_loss.numpy())
                epoch_critic_losses.append(critic_loss.numpy())
                epoch_entropies.append(tf.reduce_mean(entropy).numpy())
                epoch_kls.append(approx_kl)

            # Calculate metrics within this epoch
            actor_losses.append(np.mean(epoch_actor_losses))
            critic_losses.append(np.mean(epoch_critic_losses))
            entropies.append(np.mean(epoch_entropies))

            mean_kl = np.mean(epoch_kls)

            # Early-Stopping
            if mean_kl > 1.5 * target_kl:
                break

        self.clear_memory()#After a batch is done with all epochs clear for new 
        self.current_episode += 1

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropies)



class StateRepresentation:
    """
    It is responsible for converting the state of the H3 Hexagon simulation
    into a normalised vector that the RL agent understands.
    """
    def __init__(self, couriers, min_lat, min_lon, max_lat, max_lon, window_size=10):
        """
       Initialises the state handler.
        
        Args:
            couriers (list): The list of courier objects.
            min_lat, min_lon, max_lat, max_lon (float): The boundaries of the area of operation for normalisation.
            window_size (int): The size of the window for historical features.
        """
        # Get borders
        self.couriers = couriers
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon
        
        # Using rolling window
        self.order_history = deque(maxlen=window_size)
        self.delivery_time_history = deque(maxlen=window_size)

    def normalize_location(self, h3_index):
        """
        Take H3 index and normalise
        """
        if not h3_index or not isinstance(h3_index, str):
            return 0.0, 0.0
            
        try:
            lat, lon = h3.cell_to_latlng(h3_index)
            norm_lat = (lat - self.min_lat) / (self.max_lat - self.min_lat)
            norm_lon = (lon - self.min_lon) / (self.max_lon - self.min_lon)
            return norm_lat, norm_lon
        except:
            # Fallback, is no valid index
            return 0.0, 0.0

    def get_courier_features(self):
        """
        Extracting courier features to use
        """
        features = []
        for courier in self.couriers:
            # Use the normalisation function
            norm_lat, norm_lon = self.normalize_location(courier.position)
            norm_active_deliveries = min(courier.active_deliveries / 5.0, 1.0)
            
            # for stops
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
        Extracts order features. Logic but uses
        data fields (“sender_h3”) and H3 normalisation.
        """
        features = []
        self.order_history.append(len(current_orders))
        
        # Calculate order density
        order_density = sum(self.order_history) / self.order_history.maxlen if self.order_history else 0
        norm_order_density = min(order_density / 20.0, 1.0)
        
        #For order features just take the first 10 orders
        max_orders = 10
        for i in range(min(len(current_orders), max_orders)):
            order = current_orders[i]
            
            # Get restaurant and customer location for normalisation 
            rest_lat, rest_lon = self.normalize_location(order['sender_h3'])
            cust_lat, cust_lon = self.normalize_location(order['recipient_h3'])
            
            current_time = order['platform_order_time']
            prep_time = (order['estimate_meal_prepare_time'] - current_time) / 3600.0
            delivery_time = (order['estimate_arrived_time'] - current_time) / 3600.0
            
            features.extend([
                rest_lat, rest_lon, cust_lat, cust_lon,
                min(prep_time, 2.0) / 2.0, min(delivery_time, 2.0) / 2.0
            ])
            
        # Padding, if less than 10 fill with zeros
        num_features_per_order = 6
        padding_needed = (max_orders * num_features_per_order) - len(features)
        if padding_needed > 0:
            features.extend([0.0] * padding_needed)
            
        features.append(norm_order_density)
        return np.array(features)

    def get_system_features(self):
        """
        Get the features of the system about utilisation and stuff
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
        """
        Combines all states
        """
        courier_features = self.get_courier_features()
        order_features = self.get_order_features(current_orders)
        system_features = self.get_system_features()
        
        return np.concatenate([courier_features, order_features, system_features])    



class EnhancedStateHandler:
    """
    This class separates the calculation of global
    (once per time step) and task-specific (once per task)
    state characteristics in order to massively accelerate the simulation.
    """
    def __init__(self, grid=None):
        
        self.grid = grid
        max_h3_distance = 1346 #is the max value, everything above = 1.0
        self.max_distance = max_h3_distance
        self.max_sqrt_distance = np.sqrt(self.max_distance)
        self.delivery_times = deque(maxlen=100)
        self.system_load = deque(maxlen=100)
        self.max_queue_length = 500.0 #max queue length
        self.max_couriers = 1284.0 #max couriers 
        self.min_couriers = 600.0 #min couriers (50%)
        self.courier_range = self.max_couriers - self.min_couriers
        if self.courier_range == 0:
            self.courier_range = 1.0

    #Normalise stuff
    def normalize_courier_count(self, num_couriers):
        return np.clip((num_couriers - self.min_couriers) / self.courier_range, 0.0, 1.0)

    def normalize_queue_length(self, queue_length):
        return min(queue_length / self.max_queue_length, 1.0)

    def normalize_distance(self, h3_distance):
        if h3_distance == float('inf'):
            return 1.0
        sqrt_distance = np.sqrt(h3_distance)
        norm_sqrt_distance = sqrt_distance / self.max_sqrt_distance
        return min(norm_sqrt_distance, 1.0)
           
    def get_global_state_features(self, couriers, order_queue):
        """
        Calculates features that apply to the entrire time step.
        Is called only once per time step.
        """
        # Courier Utilization
        active_couriers = [c for c in couriers if c.state != 'INACTIVE'] #not used is from old code
        if not active_couriers:
            norm_courier_utilization = 0.0
        else:
            total_active_deliveries = sum(c.active_deliveries for c in active_couriers)
            courier_utilization = total_active_deliveries / len(active_couriers)
            norm_courier_utilization = min(courier_utilization / 3.0, 1.0)
            
        # determine value
        norm_queue_length = self.normalize_queue_length(len(order_queue))
        norm_courier_count = self.normalize_courier_count(len(active_couriers))
        self.system_load.append(norm_courier_utilization)
        
        # All three are state representations
        return np.array([norm_courier_utilization, norm_queue_length, norm_courier_count])

    def get_order_specific_feature(self, order):
        """
        Calculates the feature that depends on the task
        Called for every job in the loop
        """
        distance_in_hex = abm3.get_hex_distance(order['sender_h3'], order['recipient_h3'])
        norm_distance = self.normalize_distance(distance_in_hex)
        return norm_distance

    def update_delivery_time(self, delivery_time):
        self.delivery_times.append(delivery_time)

class EnhancedRewardCalculator:
    """
    This class calculates a differentiated reward.
    It penalises the occurrence of a delay more severely than its duration.
    """
    def __init__(self, base_reward=1.0, max_h3_distance=1346, 
                 binary_penalty=5.0, proportional_penalty_weight=2.0):
        """
        Initialises the Reward Calculator.

        Args:
            base_reward (float): The base value for the reward.
            binary_penalty (float): A fixed penalty applied for each delay > 0.
            proportional_penalty_weight (float): The maximum weight for the proportional penalty.
        """
        self.base_reward = base_reward
        self.max_distance = max_h3_distance
        self.binary_penalty = binary_penalty
        self.proportional_penalty_weight = proportional_penalty_weight
        

    def calculate_reward(self, success, order, delivery_time, action, state_features):
        """
        We define the calculate reward function
        """
        reward = self.base_reward if success else -self.base_reward
        
        # Delay penalty (normalized by 10 minutes = 300 seconds)

        delay_midpoint = 600.0 # midpoint for sigmoid
        steepness = 0.02 # steepness for sigmoid


        delay = max(0, (order['platform_order_time'] + delivery_time) - order['estimate_arrived_time'])
        
        if delay > 1200:#it is an unsuscessful order
            return -self.base_reward
            
        sigmoid_penalty = 1.0 / (1.0 + np.exp(-steepness*(delay-delay_midpoint))) #is sigmoid penalty

        reward = reward * (1-sigmoid_penalty)
        
        norm_distance = state_features[0]
        # Distance-based modifiers
        if norm_distance > 0.5 and action == 1: #reward for long-distance splits
            reward += self.base_reward * 0.1
        elif norm_distance < 0.2 and action == 1:
            # Penalty for unnecessary short-distance splits
            reward -= self.base_reward * 0.5
                
        return reward