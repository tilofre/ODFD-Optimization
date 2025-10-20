# ppo_agent.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import logging
import os
import h3
from collections import deque
import numpy as np
#import torch
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


        self.entropy_start = 0.1   # Anfangswert (viel Exploration)
        self.entropy_end = 0.005    # Zielwert (wenig Exploration)
        self.total_episodes = total_episodes
        self.current_episode = 0    # wird bei jedem Training erhöht

        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_actor,
            clipnorm=1.0  # Ein guter Standardwert, der Explosionen verhindert
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_critic,
            clipnorm=1.0  # Ein guter Standardwert
        )
        

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(128, activation='tanh')(inputs)
        x = layers.Dense(128, activation='tanh')(x)
        outputs = layers.Dense(self.action_dim, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    def _build_critic(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='tanh')(inputs)
        x = layers.Dense(256, activation='tanh')(x)
        outputs = layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs)

    def store_transition(self, state, action, reward, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get_action(self, state, deterministic=False):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor(state_tensor)
        
        if deterministic:
            # Wähle die Aktion mit der höchsten Wahrscheinlichkeit (für Inferenz)
            action = tf.argmax(action_probs, axis=1).numpy()[0]
        else:
            # Wähle eine Aktion basierend auf der Wahrscheinlichkeitsverteilung (für Training)
            action = tf.random.categorical(tf.math.log(action_probs), 1).numpy().flatten()[0]
        
        log_prob = tf.math.log(action_probs[0, action] + 1e-10)
        return action, action_probs.numpy()[0], log_prob.numpy()

    # In ppo_agent.py in der Klasse PPOAgent

    def select_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        action_probs = self.actor(state)
        action = np.random.choice(self.action_dim, p=np.squeeze(action_probs))
        log_prob = tf.math.log(action_probs[0, action])
        # GEÄNDERT: Gib die gesamte Wahrscheinlichkeitsverteilung zurück
        return action, log_prob, action_probs
    
    def _calculate_advantages(self, rewards, dones, values):
        last_value = values[-1]
        values = values[:-1]
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t]+self.gamma*last_value*mask-values[t]
            last_adv = delta + self.gamma * self.gae_lamda * last_adv * mask
            advantages[t] = last_adv
            last_value = values[t]
        returns = advantages + values
        return advantages, returns
        
    
    def _get_entropy_coeff(self):
    # Basis-Decay
        fraction = min(1.0, self.current_episode / self.total_episodes)
        coeff = self.entropy_start + fraction * (self.entropy_end - self.entropy_start)

        # Curriculum-Resets
        if self.current_episode == 100:  # Start Phase 2
            coeff = 0.08
        elif self.current_episode == 150:  # Start Phase 3
            coeff = 0.06
        elif self.current_episode == 200:  # Start Phase 4
            coeff = 0.04

        return coeff

    def save_models(self, directory="ppo_models"):
        """Speichert die Gewichte des Actor- und Critic-Netzwerks."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.actor.save_weights(os.path.join(directory, "ppo_actor.weights.h5"))
        self.critic.save_weights(os.path.join(directory, "ppo_critic.weights.h5"))
        print(f"PPO-Modelle erfolgreich im Verzeichnis '{directory}' gespeichert.")

    def load_models(self, directory="ppo_models"):
        """Lädt die Gewichte für den Actor und Critic."""
        try:
            self.actor.load_weights(os.path.join(directory, "ppo_actor.weights.h5"))
            self.critic.load_weights(os.path.join(directory, "ppo_critic.weights.h5"))
            print(f"PPO-Modelle erfolgreich aus dem Verzeichnis '{directory}' geladen.")
        except (FileNotFoundError, OSError):
            print(f"❗️ Fehler: Gewichte im Verzeichnis '{directory}' nicht gefunden. Agent startet ohne trainiertes Wissen.")

    def train(self):
    # Konvertiere Listen in Arrays / Tensoren
        states_arr = np.array(self.states)
        actions_arr = np.array(self.actions)
        rewards_arr = np.array(self.rewards)
        dones_arr = np.array(self.dones)
        log_probs_arr = tf.convert_to_tensor(self.log_probs, dtype=tf.float32)

        # Werte vom Critic
        states_with_last = np.append(states_arr, [self.states[-1]],axis=0)
        values = self.critic.predict(states_with_last, verbose=0)
        values = np.squeeze(values)

        # Vorteile & Returns berechnen
    
        advantages, returns = self._calculate_advantages(rewards_arr, dones_arr, values)

        # Normalisierung der Vorteile (wichtig!)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Tensor-Konvertierung
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        # returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Debug: Skalen prüfen
        #logging.debug(f"[DBG] returns mean={np.mean(returns):.3f}, std={np.std(returns):.3f}")
        #logging.debug(f"[DBG] advantages mean={np.mean(advantages):.6f}, std={np.std(advantages):.6f}")

        # Hyperparameter für PPO-Update
        batch_size = len(states_arr)
        minibatch_size = max(64, batch_size // 4)
        target_kl = 0.02  # Early-Stopping bei zu großem KL

        actor_losses, critic_losses, entropies = [], [], []

        
        entropy_coeff = self._get_entropy_coeff()

        for ep in range(self.epochs):
            idxs = np.arange(batch_size)
            np.random.shuffle(idxs)

            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_entropies = []
            epoch_kls = []

            for start in range(0, batch_size, minibatch_size):
                mb_idx = idxs[start:start+minibatch_size]

                s_mb = states_arr[mb_idx]
                a_mb = actions_arr[mb_idx]
                adv_mb = tf.gather(advantages, mb_idx)
                ret_mb = tf.gather(returns, mb_idx)
                old_logp_mb = tf.gather(log_probs_arr, mb_idx)

                with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                    current_probs = self.actor(s_mb, training=True)
                    entropy = -tf.reduce_sum(current_probs * tf.math.log(current_probs + 1e-8), axis=1)

                    action_indices = tf.stack([tf.range(len(a_mb)), a_mb], axis=1)
                    current_action_probs = tf.gather_nd(current_probs, action_indices)

                    ratio = current_action_probs / (tf.exp(old_logp_mb) + 1e-8)

                    surr1 = ratio * adv_mb
                    surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_mb

                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2) + entropy_coeff * entropy)
                    critic_values = tf.squeeze(self.critic(s_mb, training=True))
                    critic_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(ret_mb, critic_values))

                # Gradienten berechnen
                actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)

                # Gradient Clipping
                actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)
                critic_grads, _ = tf.clip_by_global_norm(critic_grads, 0.5)

                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                # KL Divergenz Approximation
                new_logp_mb = tf.math.log(current_action_probs + 1e-8)
                approx_kl = tf.reduce_mean(old_logp_mb - new_logp_mb).numpy()

                epoch_actor_losses.append(actor_loss.numpy())
                epoch_critic_losses.append(critic_loss.numpy())
                epoch_entropies.append(tf.reduce_mean(entropy).numpy())
                epoch_kls.append(approx_kl)

            # Epoche fertig → Metriken mitteln
            actor_losses.append(np.mean(epoch_actor_losses))
            critic_losses.append(np.mean(epoch_critic_losses))
            entropies.append(np.mean(epoch_entropies))

            mean_kl = np.mean(epoch_kls)
            #logging.debug(f"[DBG] Epoch {ep+1}/{self.epochs} | KL={mean_kl:.5f} | ActorLoss={actor_losses[-1]:.5f}")

            # Early-Stopping, wenn KL zu groß → verhindert, dass Policy zu große Sprünge macht
            if mean_kl > 1.5 * target_kl:
                #logging.debug(f"[EARLY STOP] epoch={ep+1} mean_kl={mean_kl:.4f} > {target_kl*1.5:.4f}")
                break

        self.clear_memory()
        self.current_episode += 1

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropies)



class StateRepresentation:
    """
    Sie ist dafür verantwortlich, den Zustand Ihrer H3-Hexagon-Simulation
    in einen normalisierten Vektor umzuwandeln, den der RL-Agent versteht.
    """
    def __init__(self, couriers, min_lat, min_lon, max_lat, max_lon, window_size=10):
        """
        Initialisiert den State Handler.
        
        Args:
            couriers (list): Die Liste Ihrer Courier-Objekte.
            min_lat, min_lon, max_lat, max_lon (float): Die Grenzen Ihres
                                                        Operationsgebiets zur Normalisierung.
            window_size (int): Die Größe des Fensters für historische Features.
        """
        # Wir speichern keine grid-Liste, sondern die Grenzen des Gebiets.
        self.couriers = couriers
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon
        
        # Die Rolling Windows für historische Daten bleiben identisch.
        self.order_history = deque(maxlen=window_size)
        self.delivery_time_history = deque(maxlen=window_size)

    def normalize_location(self, h3_index):
        """
        NEUE FUNKTION: Nimmt einen H3-Index, wandelt ihn in Lat/Lon um
        und normalisiert diese Koordinaten auf den Bereich [0, 1].
        """
        if not h3_index or not isinstance(h3_index, str):
            return 0.0, 0.0
            
        try:
            # Der Kern der Anpassung: H3 -> Lat/Lon
            lat, lon = h3.cell_to_latlng(h3_index)
            
            # Die Normalisierungslogik ist danach identisch.
            norm_lat = (lat - self.min_lat) / (self.max_lat - self.min_lat)
            norm_lon = (lon - self.min_lon) / (self.max_lon - self.min_lon)
            return norm_lat, norm_lon
        except:
            # Fallback, falls ein ungültiger H3-Index übergeben wird
            return 0.0, 0.0

    def get_courier_features(self):
        """
        Extrahiert Kurier-Features. Die Logik ist identisch,
        ruft aber unsere neue Normalisierungsfunktion auf.
        """
        features = []
        for courier in self.couriers:
            # Nutzt unsere neue H3-Normalisierungsfunktion
            norm_lat, norm_lon = self.normalize_location(courier.position)
            
            # Der Rest ist identisch zum Beispiel
            norm_active_deliveries = min(courier.active_deliveries / 5.0, 1.0)
            
            # Anpassung für Ihre 'mandatory_stops' Struktur
            if courier.mandatory_stops:
                # Annahme: mandatory_stops ist eine Liste von [hex, typ, order]-Listen
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
        Extrahiert Auftrags-Features. Logik ist identisch, nutzt aber
        Ihre Datenfelder ('sender_h3') und unsere H3-Normalisierung.
        """
        features = []
        self.order_history.append(len(current_orders))
        
        # Die Logik für Auftragsdichte etc. bleibt identisch.
        order_density = sum(self.order_history) / self.order_history.maxlen if self.order_history else 0
        norm_order_density = min(order_density / 20.0, 1.0)
        
        max_orders = 10
        for i in range(min(len(current_orders), max_orders)):
            order = current_orders[i]
            
            # Hier greifen wir auf Ihre H3-Datenfelder zu
            rest_lat, rest_lon = self.normalize_location(order['sender_h3'])
            cust_lat, cust_lon = self.normalize_location(order['recipient_h3'])
            
            # Der Rest (Timestamps etc.) ist identisch.
            current_time = order['platform_order_time']
            prep_time = (order['estimate_meal_prepare_time'] - current_time) / 3600.0
            delivery_time = (order['estimate_arrived_time'] - current_time) / 3600.0
            
            features.extend([
                rest_lat, rest_lon, cust_lat, cust_lon,
                min(prep_time, 2.0) / 2.0, min(delivery_time, 2.0) / 2.0
            ])
            
        # Padding, falls weniger als max_orders vorhanden sind.
        num_features_per_order = 6
        padding_needed = (max_orders * num_features_per_order) - len(features)
        if padding_needed > 0:
            features.extend([0.0] * padding_needed)
            
        features.append(norm_order_density)
        return np.array(features)

    def get_system_features(self):
        """Diese Funktion ist grid-unabhängig und kann 1:1 übernommen werden."""
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
        """Diese Funktion ist ebenfalls identisch und fügt alles zusammen."""
        courier_features = self.get_courier_features()
        order_features = self.get_order_features(current_orders)
        system_features = self.get_system_features()
        
        return np.concatenate([courier_features, order_features, system_features])

    def update_delivery_time(self, delivery_time):
        """Diese Funktion ist identisch."""
        self.delivery_time_history.append(delivery_time)
    
    # def save_models(self, prefix="ppo_model"):
    #     """Speichert die Gewichte des Actor- und Critic-Netzwerks."""
    #     torch.save(self.actor.state_dict(), f"{prefix}_actor.pth")
    #     torch.save(self.critic.state_dict(), f"{prefix}_critic.pth")
    #     print(f"Modelle unter {prefix}_actor.pth und {prefix}_critic.pth gespeichert.")



class EnhancedStateHandler:
    """
    OPTIMIERTE VERSION: Diese Klasse trennt die Berechnung von globalen
    (einmal pro Zeitschritt) und auftragsspezifischen (einmal pro Auftrag)
    Zustandsmerkmalen, um die Simulation massiv zu beschleunigen.
    """
    def __init__(self, grid=None):
        # --- (Ihr gesamter __init__-Code bleibt unverändert) ---
        self.grid = grid
        max_h3_distance = 1346
        self.max_distance = max_h3_distance
        self.max_sqrt_distance = np.sqrt(self.max_distance)
        self.delivery_times = deque(maxlen=100)
        self.system_load = deque(maxlen=100)
        self.max_queue_length = 500.0
        self.max_couriers = 1284.0
        self.min_couriers = 700.0
        self.courier_range = self.max_couriers - self.min_couriers
        if self.courier_range == 0:
            self.courier_range = 1.0

    # --- (Alle Normalisierungs-Funktionen bleiben unverändert) ---
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
        
    # =================================================================
    # NEUE, SCHNELLE FUNKTIONEN
    # =================================================================
    
    def get_global_state_features(self, couriers, order_queue):
        """
        NEU: Berechnet Features, die für den GESAMTEN Zeitschritt gelten.
        Wird nur EINMAL pro Zeitschritt aufgerufen.
        """
        # 1. Kurier-Auslastung (Courier Utilization)
        active_couriers = [c for c in couriers if c.state != 'INACTIVE']
        if not active_couriers:
            norm_courier_utilization = 0.0
        else:
            total_active_deliveries = sum(c.active_deliveries for c in active_couriers)
            courier_utilization = total_active_deliveries / len(active_couriers)
            norm_courier_utilization = min(courier_utilization / 3.0, 1.0)
            
        # 2. Länge der Warteschlange
        norm_queue_length = self.normalize_queue_length(len(order_queue))
        
        # 3. Anzahl der Kuriere
        norm_courier_count = self.normalize_courier_count(len(active_couriers))
        
        # Metriken aktualisieren
        self.system_load.append(norm_courier_utilization)
        
        # Gibt die 3 globalen Features zurück
        return np.array([norm_courier_utilization, norm_queue_length, norm_courier_count])

    def get_order_specific_feature(self, order):
        """
        NEU: Berechnet das EINE Feature, das vom Auftrag abhängt.
        Sehr schnell, wird für jeden Auftrag in der Schleife aufgerufen.
        """
        distance_in_hex = abm3.get_hex_distance(order['sender_h3'], order['recipient_h3'])
        norm_distance = self.normalize_distance(distance_in_hex)
        return norm_distance

    def update_delivery_time(self, delivery_time):
        """Diese Funktion bleibt identisch."""
        self.delivery_times.append(delivery_time)

class EnhancedRewardCalculator:
    """
    Diese Klasse berechnet eine differenzierte Belohnung.
    NEU: Sie bestraft das Auftreten einer Verspätung stärker als deren Dauer.
    """
    def __init__(self, base_reward=1.0, max_h3_distance=1346, 
                 binary_penalty=5.0, proportional_penalty_weight=2.0):
        """
        Initialisiert den Reward Calculator.

        Args:
            base_reward (float): Der Grundwert für die Belohnung.
            binary_penalty (float): Eine fixe Strafe, die bei jeder Verspätung > 0 angewendet wird.
            proportional_penalty_weight (float): Das maximale Gewicht für die proportionale Strafe.
        """
        self.base_reward = base_reward
        self.max_distance = max_h3_distance
        self.binary_penalty = binary_penalty
        self.proportional_penalty_weight = proportional_penalty_weight
        

    def calculate_reward(self, success, order, delivery_time, action, state_features):
        """
        Berechnet eine klare, additive und stabile Belohnung.
        """
        reward = self.base_reward if success else -self.base_reward
        
        # Delay penalty (normalized by 5 minutes = 300 seconds)

        delay_midpoint = 600.0
        steepness = 0.02


        delay = max(0, (order['platform_order_time'] + delivery_time) - order['estimate_arrived_time'])
        
        if delay > 1200:
            return -self.base_reward
            
        sigmoid_penalty = 1.0 / (1.0 + np.exp(-steepness*(delay-delay_midpoint)))

        reward = reward * (1-sigmoid_penalty)
        
        norm_distance = state_features[0]
        # Distance-based modifiers
        if norm_distance > 0.5 and action == 1:
            reward += self.base_reward * 0.2
        elif norm_distance < 0.2 and action == 1:
            # Penalty for unnecessary short-distance splits
            reward -= self.base_reward * 0.5
                
        return reward