# Description: Enhanced RL agent with state handler and reward calculator

import numpy as np
from collections import deque
from collections import defaultdict
import copy
import h3
from collections import defaultdict
import q_utils.abmQ as abm3

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


class EnhancedStateHandler:
    """
    Diese Klasse ist eine 1:1-Adaption von 'EnhancedStateHandler' für
    ein H3-Hexagon-Grid-System.
    
    Sie behält die Logik der Feature-Extraktion und Diskretisierung bei,
    nutzt aber H3-spezifische Funktionen für die Berechnungen.
    """
    def __init__(self, n_distance_bins=10, n_courier_bins=10):
        """
        Initialisiert den State Handler.
        
        Args:
            n_distance_bins (int): Anzahl der Kategorien für die Distanz.
            n_courier_bins (int): Anzahl der Kategorien für die Kurier-Verfügbarkeit.
            max_h3_distance (int): Die maximal erwartete Distanz in Hexagons,
                                   wird zur Normalisierung auf [0,1] genutzt.
        """
        self.n_distance_bins = n_distance_bins
        self.n_courier_bins = n_courier_bins
        
        # ANPASSUNG: Wir definieren eine maximale Distanz in H3-Hexagons.
        max_h3_distance = 1346
        self.max_distance = max_h3_distance
        self.max_sqrt_distance = np.sqrt(self.max_distance)
        # Die Logik für die Erstellung der Bins bleibt identisch.
        #self.distance_bins = [-0.001, 0.164, 0.239, 0.302, 0.363, 0.421, 0.479, 0.54, 0.617, 0.741, 1.0]        
        self.courier_bins = np.linspace(0, 1, n_courier_bins + 1)
        self.distance_bins = np.linspace(0, 1, n_courier_bins + 1)
        
        # Die Rolling Windows für Metriken bleiben identisch.
        self.delivery_times = deque(maxlen=100)
        self.system_load = deque(maxlen=100)

        self.raw_utilization_log = [] 

    def normalize_distance(self, h3_distance):
        if h3_distance == float('inf'):
            return 1.0
        sqrt_distance = np.sqrt(h3_distance)
        norm_sqrt_distance = sqrt_distance / self.max_sqrt_distance
        return min(norm_sqrt_distance, 1.0)

    # In Q-Learning.ipynb -> EnhancedStateHandler
    # In Q-Learning.ipynb -> EnhancedStateHandler
    def get_state_features(self, order, couriers):
        """
        Extracts and normalizes state features with a corrected,
        scaled calculation for system utilization.
        """
        # In EnhancedStateHandler -> get_state_features (FINALE VERSION)

    def get_state_features(self, order, couriers):
        """
        Extrahiert und normalisiert Zustandsmerkmale mit einer datengestützten,
        kalibrierten Skalierung der Systemauslastung.
        """
        # 1. Distanzmerkmal (unverändert)
        distance_in_hex = abm3.get_hex_distance(
            order['sender_h3'], order['recipient_h3']
        )
        norm_distance = self.normalize_distance(distance_in_hex)

        # 2. FINALE, KALIBRIERTE SKALIERUNG DER AUSLASTUNG
        
        # Die Werte aus Ihrer Grafik!
        MIN_REAL_UTILIZATION = 0.15
        MAX_REAL_UTILIZATION = 0.40

        total_couriers = len(couriers)
        if total_couriers == 0:
            scaled_utilization = 1.0
        else:
            # Berechne die rohe, proportionale Auslastung
            total_active_deliveries = sum(c.active_deliveries for c in couriers)
            max_possible_deliveries = total_couriers * 3
            raw_utilization = (total_active_deliveries / max_possible_deliveries) if max_possible_deliveries > 0 else 0
                
            # Bilde den ECHTEN Betriebsbereich auf den vollen Bereich [0, 1] ab.
            if raw_utilization <= MIN_REAL_UTILIZATION:
                scaled_utilization = 0.0
            elif raw_utilization >= MAX_REAL_UTILIZATION:
                scaled_utilization = 1.0
            else:
                scaled_utilization = (raw_utilization - MIN_REAL_UTILIZATION) / \
                                    (MAX_REAL_UTILIZATION - MIN_REAL_UTILIZATION)

        norm_availability = 1 - scaled_utilization
        
        return norm_distance, norm_availability

    def discretize_state(self, state_features):
        """
        Diese Funktion wandelt die normalisierten Features in diskrete Bins um.
        Sie ist rein mathematisch und bleibt daher 1:1 identisch.
        """
        norm_distance, norm_availability = state_features
        
        # `np.digitize` ordnet die Werte (z.B. 0.75) dem passenden Bin zu.
        distance_bin = np.digitize(norm_distance, self.distance_bins) - 1
        courier_bin = np.digitize(norm_availability, self.courier_bins) - 1
        
        return (distance_bin, courier_bin)

    def get_state(self, order, couriers):
        """
        Dies ist die Hauptfunktion, die alles zusammenführt.
        Sie ruft die Feature-Extraktion und die Diskretisierung auf.
        """
        features = self.get_state_features(order, couriers)
        discretized_state = self.discretize_state(features)
        return discretized_state

    def update_delivery_time(self, delivery_time):
        """Diese Funktion bleibt 1:1 identisch."""
        self.delivery_times.append(delivery_time)
    

class QLearningAgent:
    def __init__(self, actions_n=2, learning_rate=0.2, discount_factor=0.95, epsilon=0.5):
        self.q_table = defaultdict(default_q_value)
        self.actions = actions_n
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_counts = defaultdict(default_action_counts)
        
    def get_action(self, state, couriers=None):
        # === FIX: Initialisierung bei Erstkontakt ===
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
    
    def learn(self, state, action, reward, next_state):
        # === FIX: Initialisierung bei Erstkontakt (für state und next_state) ===
        if isinstance(self.q_table[state], float):
            self.q_table[state] = np.zeros(self.actions)
        if isinstance(self.q_table[next_state], float):
            self.q_table[next_state] = np.zeros(self.actions)

        # Now the following access is safe
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
    
    def decrease_epsilon(self, decay=0.99):
        self.epsilon = max(0.05, self.epsilon * decay)
    
    def get_copy(self):
        return copy.deepcopy(self)

class EnhancedRewardCalculator:
    """
    Diese Klasse ist eine 1:1-Adaption von 'EnhancedRewardCalculator' für
    ein H3-Hexagon-Grid-System.

    Sie berechnet eine differenzierte Belohnung, die den Erfolg, die Lieferzeit,
    die gewählte Strategie (Split/Direct) und die aktuelle Systemauslastung
    berücksichtigt.
    """
    def __init__(self, base_reward=1.0, max_h3_distance=1346):
        """
        Initialisiert den Reward Calculator.

        Args:
            base_reward (float): Der Grundwert für die Belohnung.
            max_h3_distance (int): Die maximale Distanz in Hexagons, die zur
                                   Normalisierung verwendet wird (sollte mit dem
                                   Wert im H3StateHandler übereinstimmen).
        """
        self.base_reward = base_reward
        self.max_distance = max_h3_distance
        #self.binary_penalty = binary_penalty
        #self.proportional_penalty_weight = proportional_penalty_weight

    def calculate_reward(self, success, order, delivery_time, action, system_load, state_features):
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
        if norm_distance >= 0.5 and action == 1:
            reward += self.base_reward * 0.0
        elif norm_distance <= 0.2 and action == 1:
            # Penalty for unnecessary short-distance splits
            reward -= self.base_reward * 0.5
                
        return reward
    

def default_q_value():
    """Gibt den Standardwert für einen neuen Q-Wert zurück."""
    return 0.0

def default_action_counts():
    """Gibt ein NumPy-Array mit Nullen als Standardwert zurück."""
    return np.zeros(2)