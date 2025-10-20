# ## 1. Library Imports and Initial Setup
import pandas as pd
import numpy as np
import math
import random
import h3
from itertools import permutations
from collections import defaultdict
import q_utils.split2 as split2
import q_utils.rejection as rejection2
import q_utils.repositioning as repositioning2
from collections import deque




class Courier:
    """
    Here we create our courier class, which represents a single courier agent in our hexagon-grid simulation.
    start_hex is an h3 index that represents the couriers starting point.    
    
    """
    def __init__(self, courier_id, start_hex):
        self.id = courier_id #Couriers unique ID
        self.base = start_hex # The courier's home base or starting hexagon H3 index.
        self.position = start_hex # The courier's current location, stored as an H3 index string, beginning with the starting point and is updated through the process
        self.state = 'IDLE' # The current state. Can be 'IDLE' (available) or 'BUSY' (on a task).
        self.arrival_time = None # The simulation timestamp when the courier should arrive at their current target. (None as long as courier is idle)
        self.rejected_orders = [] # A list of order_ids that this courier has rejected, to prevent them from being offered the same order again within the same assignment attempt.
        self.mandatory_stops = []
        self.active_deliveries = 0
        
        self.was_repositioned = False #Only for plotting
        self.completed_orders = [] #Only for plotting
        self.route = {0: start_hex} #Only for plotting

def get_hex_distance(start_hex, end_hex):
    """
    Berechnet die Distanz mit der schnellen h3.grid_distance Funktion
    UND nutzt einen Cache, um wiederholte Berechnungen zu vermeiden.
    """
  
    try:
        distance = h3.grid_distance(start_hex, end_hex)
        # 3. Speichere das Ergebnis im Cache für die Zukunft
        return distance
    except (h3.H3FailedError, TypeError):
        return float('inf')
    
def calculate_travel_time(start_hex, end_hex, SPEED_HEX_PER_STEP, steps):
    """Hilfsfunktion zur Berechnung der Reisezeit in Sekunden für das Batching."""
    distance = get_hex_distance(start_hex, end_hex)
    if distance == float('inf'): 
        return float('inf')
    # Reisezeit = (Anzahl der Schritte) * (Dauer eines Schritts)
    travel_steps = math.ceil(distance / SPEED_HEX_PER_STEP)
    return travel_steps * steps



def initiate_couriers(total_couriers_to_create, data_source_df):    
    fleet = []
    """
    We need to initialize our fleet of couriers by placing them at their first known 
    historical starting positions (start hexagon) from the provided dataset.

    """

    # To find the first known position of each real courier, we sort all orders by time
    # and then keep only the first entry for each unique 'courier_id'
    start_positions_df = data_source_df.sort_values('platform_order_time').drop_duplicates('courier_id') 
    
    # we create a list of all hexagon locations where couriers started.
    # We drop any potential missing values (.dropna()) to ensure the list is clean.
    real_start_hexes_list  = start_positions_df['grab_h3'].dropna().tolist()
    
    # Create couriers and distribute them across the starting locations based on their starting amount
    if not real_start_hexes_list:
        return fleet

    for i in range(total_couriers_to_create):
        start_hex = real_start_hexes_list[i % len(real_start_hexes_list)]
        #We have a list with all starting points and the number of couriers are iterating through. 
        # If courier fleet is scaled down, the starting points differ
        
        courier = Courier(courier_id=i, start_hex=start_hex)
        
        fleet.append(courier)
            
    return fleet


def move_couriers(couriers, timestart, metrics, delivered_order_ids, SPEED_HEX_PER_STEP, steps):
    """Arbeitet die 'mandatory_stops'-Liste für jeden Kurier ab."""
    delay_inc, _, success, success_delay = metrics
    for courier in [c for c in couriers if c.state == 'BUSY' and c.arrival_time is not None and timestart >= c.arrival_time]:
        if not courier.mandatory_stops:
            courier.state = 'IDLE'; courier.arrival_time = None
            continue
        done_stop = courier.mandatory_stops.pop(0)
        courier.position = done_stop[0]
        courier.route[timestart] = courier.position
        stop_type, order = done_stop[1], done_stop[2]
        departure_time = courier.arrival_time
        if stop_type == 'R': 
            departure_time = max(courier.arrival_time, order['estimate_meal_prepare_time'])
        if stop_type == 'C':
            delivered_order_ids.add(order['order_id'])
            courier.completed_orders.append(order)
            delay = courier.arrival_time - order['estimate_arrived_time']
            courier.active_deliveries = max(0, courier.active_deliveries - 1)
            if delay > 0: delay_inc += delay; success_delay += 1
            else: success += 1
        elif stop_type == 'B':
            courier.was_repositioned = True
        if courier.mandatory_stops:
            next_stop = courier.mandatory_stops[0]
            travel_time = calculate_travel_time(courier.position, next_stop[0], SPEED_HEX_PER_STEP, steps)
            courier.arrival_time = departure_time + travel_time
        else:
            courier.state = 'IDLE'; courier.arrival_time = None
    return couriers, (delay_inc, 0, success, success_delay), delivered_order_ids

def move_idle_couriers_randomly(couriers, timestart):
    """
    Just a function where idle couriers are moved randomly to adjacent hexagons.
    For realistic behaviour.
    """
    for courier in couriers:
        if courier.state == 'IDLE':
            try:
                neighbors = h3.grid_ring(courier.position, 1) #grid_ring for adjacent hexagons (distance 1)
                new_position = random.choice(list(neighbors))
                
            
                courier.position = new_position
                
                courier.route[timestart] = new_position
                
            except Exception:
                #If grid_ring fails, then do nothing
                pass


def find_best_free_courier(order, free_couriers):
    """Find the closest courier"""
    best_courier = None
    min_dist = float('inf')
    restaurant_hex = order['sender_h3']
    for courier in free_couriers:
        dist_hex = get_hex_distance(courier.position, restaurant_hex)
        if dist_hex < min_dist:
            min_dist = dist_hex
            best_courier = courier
    return best_courier

def calculate_total_distance_in_hexes(couriers_list):
    """
    Calculates the total distance traveled by the entire fleet, measured in hexagons.
    """
    total_hex_distance = 0

    #Iterate through each courier in the final fleet
    for courier in couriers_list:
        sorted_route_items = sorted(courier.route.items())
        
        for i in range(1, len(sorted_route_items)):
            start_hex = sorted_route_items[i-1][1] 
            end_hex = sorted_route_items[i][1]            
            segment_distance = get_hex_distance(start_hex, end_hex) #calculate the distance of the segment in hexes
           
            if segment_distance != float('inf'):
                total_hex_distance += segment_distance
                
    return total_hex_distance

def move_couriers_new(couriers, timestart, metrics, delivered_order_ids, SPEED_HEX_PER_STEP, steps):
    """
    Arbeitet die 'mandatory_stops'-Liste für jeden Kurier ab.
    Diese Version kann zwischen echten Kunden und Split-Treffpunkten unterscheiden.
    """
    delay_inc, _, success, success_delay = metrics
    for courier in [c for c in couriers if c.state == 'BUSY' and c.arrival_time is not None and timestart >= c.arrival_time]:
        if not courier.mandatory_stops:
            courier.state = 'IDLE'; courier.arrival_time = None
            continue
            
        done_stop = courier.mandatory_stops.pop(0)
        courier.position = done_stop[0]
        courier.route[timestart] = courier.position
        
        # Unpack the stop information
        stop_type, order = done_stop[1], done_stop[2]
        
        departure_time = courier.arrival_time
        
        if stop_type == 'R': 
            if order is not None and 'estimate_meal_prepare_time' in order:
                departure_time = max(courier.arrival_time, order['estimate_meal_prepare_time'])
        
        # ========================================================
        # START DER KORREKTUR
        # ========================================================
        if stop_type == 'C':
                # NEU: Zuerst die Verspätung für JEDEN C-Stopp mit Deadline berechnen
                if order is not None and 'estimate_arrived_time' in order:
                    delay = courier.arrival_time - order['estimate_arrived_time']
                    if delay > 0:
                        delay_inc += delay
                        # Wichtig: success_delay wird nur bei echten Aufträgen erhöht
                        if 'order_id' in order:
                            success_delay += 1
                    else:
                        if 'order_id' in order:
                            success += 1

                # KORRIGIERT: Aktionen nur für ECHTE Kundenaufträge ausführen
                if order is not None and 'order_id' in order:
                    delivered_order_ids.add(order['order_id'])
                    courier.completed_orders.append(order)
                
                # Die Reduzierung der aktiven Lieferungen gilt für beide Fälle
                courier.active_deliveries = max(0, courier.active_deliveries - 1)
        # ========================================================
        # ENDE DER KORREKTUR
        # ========================================================

        elif stop_type == 'B':
            courier.was_repositioned = True

        if courier.mandatory_stops:
            next_stop = courier.mandatory_stops[0]
            travel_time = calculate_travel_time(courier.position, next_stop[0], SPEED_HEX_PER_STEP, steps)
            courier.arrival_time = departure_time + travel_time
        else:
            courier.state = 'IDLE'; courier.arrival_time = None
            
    return couriers, (delay_inc, 0, success, success_delay), delivered_order_ids

def handle_standard_assignment(order, attempts, couriers, timestart, constants, rejection_model, processed_ids_set):
    """
    Finds the best courier for a standard order, handles rejection and fallback logic, and assigns if accepted.
    Returns True if the order was fully assigned, False otherwise.
    """
    # Create a list of potential couriers for this specific order attempt.
    candidates_for_this_order = [c for c in couriers if c.active_deliveries < 3 and order['order_id'] not in c.rejected_orders]
    
    # --- Start the assignment-rejection loop for this order ---
    while candidates_for_this_order:
        best_courier, route_plan = None, None

        # --- "Second-Chance" Fallback Logic ---
        # If an order is "stuck" in the queue, assign it to the nearest IDLE courier without stacking.
        if attempts > constants['MAX_QUEUE_ATTEMPTS']:
            free_candidates = [c for c in candidates_for_this_order if c.state == 'IDLE']
            if free_candidates:
                best_courier = find_best_free_courier(order, free_candidates)
                if best_courier:
                    # Create a simple, non-stacked route
                    route_plan = [[order['sender_h3'], 'R', order], [order['recipient_h3'], 'C', order]]
        else:
            # --- Normal, Optimal Assignment Logic ---
            candidates_for_this_order.sort(key=lambda c: get_hex_distance(c.position, order['sender_h3']))
            top_candidates = candidates_for_this_order[:100]
            
            best_courier, route_plan, _ = find_best_assignment_new(
                order, top_candidates, timestart, 
                constants['SPEED_HEX_PER_STEP'], constants['steps'], constants['MAX_ACCEPTABLE_DELAY_SECONDS']
            )

        if not best_courier:
            # No suitable courier was found among the candidates
            return False, 0 # ✅ CORRECT: Return a tuple for failure

        # --- Rejection Check ---
        prob_rejection = rejection2.predict_rejection_probability(order, rejection_model)
        if random.random() < prob_rejection:
            best_courier.rejected_orders.append(order['order_id'])
            candidates_for_this_order.remove(best_courier) # Remove and try the next best
        else:
        # # ASSIGNMENT SUCCESSFUL
            best_courier.mandatory_stops = route_plan
            best_courier.active_deliveries = len([s for s in route_plan if s[1] == 'C' and s[2] is not None])
            if best_courier.state == 'IDLE':
                travel_time = calculate_travel_time(best_courier.position, route_plan[0][0], constants['SPEED_HEX_PER_STEP'], constants['steps'])
                best_courier.arrival_time = timestart + travel_time
                best_courier.state = 'BUSY'
            # CRUCIAL STEP: Calculate the final delivery time
            # Find the last stop in the route plan that is a customer delivery ('C')
            final_customer_stop = None
            for stop in reversed(route_plan):
                if stop[1] == 'C' and stop[2] is not None:
                    final_customer_stop = stop
                    break
        
        # You need a function to calculate the time to complete the whole route
        total_route_time = calculate_total_route_time(best_courier, route_plan, timestart, constants)
        final_delivery_timestamp = timestart + total_route_time
        delivery_duration_seconds = final_delivery_timestamp - order['platform_order_time']

        processed_ids_set.add(order['order_id'])
        
        # ✅ CORRECT: Return a tuple with True and the calculated delivery duration
        return True, delivery_duration_seconds

    # If the while loop finishes without assigning (all candidates rejected)
    return False, 0


def calculate_total_route_time(courier, route_plan, timestart, constants):
    """
    Calculates the total time in seconds required to complete a multi-stop route plan.
    """
    total_time = 0
    current_location = courier.position
    current_time = timestart

    for stop_hex, stop_type, order_info in route_plan:
        # Time to travel to the next stop
        travel_time = calculate_travel_time(
            current_location,
            stop_hex,
            constants['SPEED_HEX_PER_STEP'],
            constants['steps']
        )
        current_time += travel_time
        total_time += travel_time

        # Account for meal preparation time at restaurants ('R')
        if stop_type == 'R' and order_info is not None:
            # The courier has to wait if they arrive before the meal is ready
            wait_time = max(0, order_info['estimate_meal_prepare_time'] - current_time)
            total_time += wait_time
            current_time += wait_time # Update the time for the next leg of the journey

        # Update location for the next leg of the journey
        current_location = stop_hex

    return total_time

def find_best_assignment_new(order, candidate_couriers, timestart, SPEED_HEX_PER_STEP, steps, MAX_ACCEPTABLE_DELAY_SECONDS):
    """
    OPTIMIERTE VERSION: Nutzt eine "Greedy Insertion"-Heuristik anstelle von
    teuren Permutationen, um die beste Route zu finden.
    """
    best_option = {"courier": None, "route": None, "lateness": float('inf')}

    for courier in candidate_couriers:
        current_stops = courier.mandatory_stops
        new_pickup = [order['sender_h3'], 'R', order]
        new_dropoff = [order['recipient_h3'], 'C', order]
        
        min_lateness_for_courier = float('inf')
        best_route_for_courier = None

        # --- HEURISTIK START ---
        # 1. Sortiere alle Pickups (Restaurants) nach ihrer Zubereitungszeit. Das ist immer optimal.
        all_restaurants = sorted(
            [s for s in current_stops if s[1] == 'R'] + [new_pickup],
            key=lambda s: s[2]['estimate_meal_prepare_time'] if s[2] is not None else 0
        )
        
        # 2. Probiere jede mögliche Einfügeposition für den neuen Kunden in der bestehenden Kunden-Route aus.
        current_customers = [s for s in current_stops if s[1] == 'C']
        for i in range(len(current_customers) + 1):
            # Erstelle eine Kandidaten-Route durch Einfügen
            perm = list(current_customers)
            perm.insert(i, new_dropoff)
            route_candidate = all_restaurants + perm
            
            # 3. Simuliere diese EINE Route (viel schneller als alle Permutationen)
            max_lateness_in_route = simulate_route_lateness(
                route_candidate, courier, timestart, 
                SPEED_HEX_PER_STEP, steps
            )
            
            if max_lateness_in_route < min_lateness_for_courier:
                min_lateness_for_courier = max_lateness_in_route
                best_route_for_courier = route_candidate        
        if best_route_for_courier and min_lateness_for_courier < best_option["lateness"]:
            best_option = {"courier": courier, "route": best_route_for_courier, "lateness": min_lateness_for_courier}

    if best_option["lateness"] <= MAX_ACCEPTABLE_DELAY_SECONDS:
        return best_option["courier"], best_option["route"], best_option["lateness"]
    
    return None, None, float('inf')


def simulate_route_lateness(route, courier, timestart, speed, steps):

    sim_time = courier.arrival_time if courier.state == 'BUSY' and courier.arrival_time is not None else timestart
    sim_pos = courier.position if courier.state == 'IDLE' else (courier.mandatory_stops[0][0] if courier.mandatory_stops else courier.position)
    max_lateness = -float('inf')

    for stop_hex, stop_type, order_info in route:
        travel_time = calculate_travel_time(sim_pos, stop_hex, speed, steps)
        sim_time += travel_time
        sim_pos = stop_hex
        
        if stop_type == 'R' and order_info is not None:
            sim_time = max(sim_time, order_info['estimate_meal_prepare_time'])
        
        if stop_type == 'C' and order_info is not None:
            lateness = sim_time - order_info['estimate_arrived_time']
            max_lateness = max(max_lateness, lateness)
            
    return max_lateness

def create_courier_schedule(n_couriers, simulation_duration, availability_changes_per_hour=2):
    schedule = defaultdict(list)
    n_changes = int(simulation_duration / 3600 * availability_changes_per_hour)
    for _ in range(n_changes):
        timestep = np.random.randint(0, simulation_duration)
        n_adjust = np.random.randint(1, max(2, int(n_couriers * 0.1)))
        adjustment = n_adjust if np.random.random() > 0.5 else -n_adjust
        schedule[timestep].append(adjustment)
    return schedule

def manage_courier_availability(couriers, timestart):
    for courier in couriers:
        # Remove stops on list
        if courier.mandatory_stops:
            remaining_stops = [stop for stop in courier.mandatory_stops if stop[2]['estimate_arrived_time'] > timestart]
            
            if len(remaining_stops) < len(courier.mandatory_stops):
                courier.mandatory_stops = remaining_stops
                courier.active_deliveries = len([s for s in remaining_stops if s[1] == 'C'])

        if not courier.mandatory_stops and courier.state == 'BUSY':
            courier.state = 'IDLE'
            courier.active_deliveries = 0
            
    return couriers # Gibt die aktualisierte Liste zurück

# Fügen Sie diese Funktionen zu Ihrer h3_abm_utils.py oder einer ähnlichen Datei hinzu
from collections import defaultdict
import h3

def build_courier_index(idle_couriers):
    """
    Erstellt ein blitzschnelles Verzeichnis, das H3-Indizes auf Listen von
    Kurieren abbildet, die sich dort befinden.
    """
    index = defaultdict(list)
    for courier in idle_couriers:
        index[courier.position].append(courier)
    return index

def find_nearby_couriers_indexed(order_hex, courier_index, search_radius=2000):
    """
    Nutzt den Index, um Kuriere in einem Radius extrem schnell zu finden.
    """
    nearby_couriers = []
    # h3.grid_disk gibt sofort alle Hexagons im Umkreis zurück
    search_hexes = h3.grid_disk(order_hex, search_radius)
    
    for hex_id in search_hexes:
        if hex_id in courier_index:
            nearby_couriers.extend(courier_index[hex_id])
            
    return nearby_couriers


def update_couriers_and_system(timestart, steps, couriers, delivered_orders, constants, demand_forecast,order_queue):
    """
    NEUE FUNKTION (Teil 1 von run_abm_step):
    Aktualisiert den Zustand aller Kuriere, bewegt sie und verarbeitet abgeschlossene Lieferungen.
    Gibt die aktualisierte Kurierliste und die Liste der zugestellten Aufträge zurück.
    """
    # Ruft deine bestehende Funktion move_couriers_new auf
    couriers, _, newly_delivered_ids = move_couriers_new(
        couriers, 
        timestart, 
        (0, 0, 0, 0), # Metriken werden jetzt extern getrackt
        delivered_orders,
        constants['SPEED_HEX_PER_STEP'], 
        constants['steps']
    )
    delivered_orders.update(newly_delivered_ids)
    
    USE_REPOSITIONING = True # Sie können dies als Parameter steuern
    if USE_REPOSITIONING and (timestart - constants['initial_timestart']) % constants['repositioning_interval'] == 0:
        
        current_bin_key = pd.to_datetime(timestart, unit='s').floor('15min') + pd.Timedelta(hours=8)
        dynamic_demand = demand_forecast.get(current_bin_key, {})
        if dynamic_demand:
            repositioning2.run_repositioning_strategy(
                couriers, dynamic_demand, timestart, order_queue,
                constants['SPEED_HEX_PER_STEP'], constants['steps'],
                constants['MACRO_RESOLUTION'], constants['WORK_RESOLUTION']
            )    
    return couriers, delivered_orders


def get_new_orders(timestart, steps, data):
    """
    NEUE FUNKTION (Teil 2 von run_abm_step):
    Holt alle neuen Aufträge, die im aktuellen Zeitschritt eingehen.
    """
    current_orders_df = data[
        (data['platform_order_time'] >= timestart) &
        (data['platform_order_time'] < timestart + steps)
    ]
    # Gibt eine Liste von Dictionaries zurück, was einfacher zu verarbeiten ist
    return [order.to_dict() for _, order in current_orders_df.iterrows()]

def execute_decision_for_order(order, action, couriers, timestart, constants, reward_calculator, state_handler, rejection_model, state_features):
    """
    Führt EINE einzelne Entscheidung für EINEN Auftrag aus und berechnet den daraus resultierenden Reward.
    """
    success, delivery_time = False, 0
    
    # Temporäre Sets für die Auftrags-Handler
    processed_ids_set = set()
    order_queue_placeholder = [] 

    if action == 0:  # Direkte Lieferung
        success, delivery_time = handle_standard_assignment(
            order, 0, couriers, timestart, constants, rejection_model, processed_ids_set
        )
    else:  # Split-Lieferung
        success, delivery_time = split2.try_split_assignment(
            order, couriers, timestart, constants, rejection_model, order_queue_placeholder, processed_ids_set
        )

    # KORREKTUR: Die Berechnung für 'system_load' wurde hier hinzugefügt.
    system_load = np.mean(state_handler.system_load) if state_handler.system_load else 0.0

    # Berechne den Reward für genau diese Aktion
    reward = reward_calculator.calculate_reward(
        success=success,
        order=order,
        delivery_time=delivery_time,
        action=action,
        state_features=state_features,
        system_load=system_load  # Jetzt ist die Variable definiert
    )
    
    done = False

    # Aktualisiere Metriken nach der Entscheidung
    if success:
        delay = max(0, (order['platform_order_time'] + delivery_time) - order['estimate_arrived_time'])
        state_handler.update_delivery_time(delay)
        
    return reward, done, success