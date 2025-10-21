# ## 1. Library Imports and Initial Setup
import pandas as pd
import numpy as np
import math
import random
import h3
from itertools import permutations
import abm_utils.rejection as rejection
from itertools import permutations



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
        self.was_part_of_split = False
        self.handled_orders_info = []

        self.is_on_initial_split_leg = False # TRUE für C1 auf dem Weg zum Treffpunkt
        self.is_on_final_split_leg = False   # TRUE für C2 auf dem Weg zum Kunden
        


def get_hex_distance(start_hex, end_hex):
    """
    Calculates the distance using the fast h3.grid_distance function.
    Uses a cache to avoid repeated calculations.
    """
    key = tuple(sorted((start_hex, end_hex)))

    # Calculate distance
    try:
        distance = h3.grid_distance(start_hex, end_hex) #basic h3 function
    
        return distance
    except (h3.H3FailedError, TypeError):
        return float('inf')
    
def calculate_travel_time(start_hex, end_hex, SPEED_HEX_PER_STEP, steps):
    """
    Calculated travel time between hexagons
    For assignment, courier movement ...
    
    """
    distance = get_hex_distance(start_hex, end_hex)
    if distance == float('inf'): 
        return float('inf')
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

def move_idle_couriers_randomly(couriers, timestart):
    """
    Just a function where idle couriers are moved randomly to adjacent hexagons.
    For realistic behaviour.
    Is deactivated in the current setup for global travelled distance calculations.
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
    """Find the closest courier, in case there is no best assignment possible"""
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

def find_best_assignment_new(order, candidate_couriers, timestart, SPEED_HEX_PER_STEP, steps, MAX_ACCEPTABLE_DELAY_SECONDS):
    """
    Retains detailed route calculation and adds strategic
    prioritisation for punctual stacking couriers.
    """
   # Two ‘competitions’: one for the best freelance couriers, one for the best employed couriers
    best_idle_option = {"courier": None, "route": None, "lateness": float('inf')}
    best_stacking_option = {"courier": None, "route": None, "lateness": float('inf')}

    for courier in candidate_couriers:
        current_stops = courier.mandatory_stops
        new_pickup = [order['sender_h3'], 'R', order]
        new_dropoff = [order['recipient_h3'], 'C', order]
        
        all_restaurants = sorted(
            [s for s in current_stops if s[1] == 'R'] + [new_pickup], 
            key=lambda s: s[2]['estimate_meal_prepare_time'] if s[2] is not None and 'estimate_meal_prepare_time' in s[2] else 0
        )
        all_customers = [s for s in current_stops if s[1] == 'C'] + [new_dropoff]
        
        best_route_for_courier = None
        min_lateness_for_courier = float('inf')

        for customer_permutation in permutations(all_customers):
            route_candidate = all_restaurants + list(customer_permutation)
            
            sim_time = timestart if courier.state == 'IDLE' else courier.arrival_time
            sim_pos = courier.position if courier.state == 'IDLE' else (current_stops[0][0] if current_stops else courier.position)
            
            max_lateness_in_route = -float('inf')
            
            for stop_hex, stop_type, order_info in route_candidate:
                travel_time = calculate_travel_time(sim_pos, stop_hex, SPEED_HEX_PER_STEP, steps)
                sim_time += travel_time
                sim_pos = stop_hex
                
                if stop_type == 'R' and order_info is not None:
                    if 'estimate_meal_prepare_time' in order_info:
                        sim_time = max(sim_time, order_info['estimate_meal_prepare_time'])

                if stop_type == 'C' and order_info is not None:
                    if order_info.get('is_handover'):
                        actual_handoff_time = sim_time
                        final_customer_hex = order_info['final_customer_hex']
                        final_order = order_info['final_customer_order']
                        c2_travel_time = calculate_travel_time(stop_hex, final_customer_hex, SPEED_HEX_PER_STEP, steps)
                        final_arrival_time = actual_handoff_time + c2_travel_time
                        lateness = final_arrival_time - final_order['estimate_arrived_time']
                        max_lateness_in_route = max(max_lateness_in_route, lateness)
                    elif 'estimate_arrived_time' in order_info:
                        lateness = sim_time - order_info['estimate_arrived_time']
                        max_lateness_in_route = max(max_lateness_in_route, lateness)

            if max_lateness_in_route < min_lateness_for_courier:
                min_lateness_for_courier = max_lateness_in_route
                best_route_for_courier = route_candidate
        
        # Assign the result to the respective ‘competition’
        if courier.state == 'IDLE':
            if min_lateness_for_courier < best_idle_option["lateness"]:
                best_idle_option = {"courier": courier, "route": best_route_for_courier, "lateness": min_lateness_for_courier}
        else: # 'BUSY'
            if min_lateness_for_courier < best_stacking_option["lateness"]:
                best_stacking_option = {"courier": courier, "route": best_route_for_courier, "lateness": min_lateness_for_courier}
  
    # Find the best overall option from both competitions
    final_best_option = best_idle_option
    if best_stacking_option['lateness'] < best_idle_option['lateness']:
        final_best_option = best_stacking_option

    if final_best_option["lateness"] <= MAX_ACCEPTABLE_DELAY_SECONDS:
        courier_type = "Stacking" if final_best_option['courier'].state == 'BUSY' else "Idle"
        return final_best_option["courier"], final_best_option["route"], final_best_option["lateness"]
    
    #If none of the options meet the deadline, give up.
    return None, None, float('inf')

def move_couriers_new(couriers, timestart, metrics, delivered_order_ids, SPEED_HEX_PER_STEP, steps):
    """
    Corrected version:
    - ALWAYS adds delivered orders to delivered_order_ids for the simulation loop.
    - ONLY updates performance metrics (delay, success) for 'tracked' orders.
    """
    delay_inc, _, success, success_delay, stacked_orders, rejected_orders = metrics
    
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
            if order is not None:
                departure_time = max(courier.arrival_time, order['estimate_meal_prepare_time'])
        
        if stop_type == 'C':
            if order is not None and order.get('is_handover'):
                courier.is_on_initial_split_leg = False
            if order is not None and 'order_id' in order:
                # Schritt 1: IMMER für die while-Schleife zählen
                delivered_order_ids.add(order['order_id'])
                courier.completed_orders.append(order)
                
                # Schritt 2: NUR für getrackte Aufträge die Performance-Metriken updaten
                if order.get('phase') == 'tracked':
                    delay = courier.arrival_time - order['estimate_arrived_time']
                    if delay > 0: 
                        delay_inc += delay  
                        success_delay += 1
                    else: 
                        success += 1
            
            if order is not None:
                courier.active_deliveries = max(0, courier.active_deliveries - 1)

        elif stop_type == 'B':
            courier.was_repositioned = True

        if courier.mandatory_stops:
            next_stop = courier.mandatory_stops[0]
            travel_time = calculate_travel_time(courier.position, next_stop[0], SPEED_HEX_PER_STEP, steps)
            courier.arrival_time = departure_time + travel_time
        else:
            courier.state = 'IDLE'; courier.arrival_time = None
            courier.is_on_initial_split_leg = False 
            courier.is_on_final_split_leg = False
            
    updated_metrics = (delay_inc, 0, success, success_delay, stacked_orders, rejected_orders)
    return couriers, updated_metrics, delivered_order_ids

def handle_standard_assignment(order, attempts, couriers, timestart, constants, rejection_model, processed_ids_set, metrics, assignment_log):
    """
    Finds the best courier, handles rejection, and logs assignment details for later analysis.
    Differentiates between 'tracked' and 'warmup' orders for metric collection.
    Returns a tuple: (Boolean for success, updated metrics, updated assignment_log).
    """
    delay_inc, _, success, success_delay, stacked_orders, rejected_orders = metrics
    
    #candidates_for_this_order = [c for c in couriers if c.active_deliveries < 3 and order['order_id'] not in c.rejected_orders]
    
    candidates_for_this_order = [
        c for c in couriers if
        c.active_deliveries < 3 and
        order['order_id'] not in c.rejected_orders and
        not c.is_on_initial_split_leg and  
        not c.is_on_final_split_leg           
    ]
    while candidates_for_this_order:
        best_courier, route_plan, estimated_lateness = None, None, float('inf')

        # Fallback logic for jobs stuck in the queue
        if attempts > constants['MAX_QUEUE_ATTEMPTS']:
            free_candidates = [c for c in candidates_for_this_order if c.state == 'IDLE']
            if free_candidates:
                best_courier = find_best_free_courier(order, free_candidates)
                if best_courier:
                    route_plan = [[order['sender_h3'], 'R', order], [order['recipient_h3'], 'C', order]]
                    t_time_to_rest = calculate_travel_time(best_courier.position, order['sender_h3'], constants['SPEED_HEX_PER_STEP'], constants['steps'])
                    t_arrival_at_rest = timestart + t_time_to_rest
                    t_departs_rest = max(t_arrival_at_rest, order['estimate_meal_prepare_time'])
                    t_time_to_cust = calculate_travel_time(order['sender_h3'], order['recipient_h3'], constants['SPEED_HEX_PER_STEP'], constants['steps'])
                    final_arrival = t_departs_rest + t_time_to_cust
                    estimated_lateness = final_arrival - order['estimate_arrived_time']
        else:
            # Normal, optimal allocation logic
            candidates_for_this_order.sort(key=lambda c: get_hex_distance(c.position, order['sender_h3']))
            top_candidates = candidates_for_this_order[:100]
            best_courier, route_plan, estimated_lateness = find_best_assignment_new(
                order, top_candidates, timestart, 
                constants['SPEED_HEX_PER_STEP'], constants['steps'], constants['MAX_ACCEPTABLE_DELAY_SECONDS']
            )

        if not best_courier:
            return False, metrics, assignment_log

        # --- Distinction between tracked and warmup ---
        if order.get('phase') == 'tracked':
            # For tracked only
            prob_rejection = rejection.predict_rejection_probability(order, rejection_model)
            if random.random() < prob_rejection:
                best_courier.rejected_orders.append(order['order_id'])
                candidates_for_this_order.remove(best_courier)
                rejected_orders += 1 # count rejection
            else:
                # successful assignment
                was_stacked = best_courier.state == 'BUSY'
                if was_stacked:
                    stacked_orders += 1 #count stacking
                
                # fill assignment log
                idle_couriers_at_assignment = len([c for c in couriers if c.state == 'IDLE'])
                assignment_log.append({
                    'timestart': timestart, 'was_stacked': was_stacked,
                    'idle_couriers_at_assignment': idle_couriers_at_assignment,
                    'estimated_lateness_seconds': estimated_lateness
                })
                
                # Assign order
                best_courier.mandatory_stops = route_plan
                best_courier.handled_orders_info.append(order)
                best_courier.active_deliveries = len([s for s in route_plan if s[1] == 'C' and s[2] is not None])
                if best_courier.state == 'IDLE':
                    travel_time = calculate_travel_time(best_courier.position, route_plan[0][0], constants['SPEED_HEX_PER_STEP'], constants['steps'])
                    best_courier.arrival_time = timestart + travel_time
                    best_courier.state = 'BUSY'
                
                for stop in route_plan:
                    if stop[1] == 'C' and stop[2] is not None and 'order_id' in stop[2]:
                        processed_ids_set.add(stop[2]['order_id'])
                
                updated_metrics = (delay_inc, 0, success, success_delay, stacked_orders, rejected_orders)
                return True, updated_metrics, assignment_log
        else:
            #warmup
            prob_rejection = rejection.predict_rejection_probability(order, rejection_model)
            if random.random() < prob_rejection:
                best_courier.rejected_orders.append(order['order_id'])
                candidates_for_this_order.remove(best_courier)
                # Not counting rejected order
            else:
                # successful assignment
                best_courier.mandatory_stops = route_plan
                best_courier.handled_orders_info.append(order)
                best_courier.active_deliveries = len([s for s in route_plan if s[1] == 'C' and s[2] is not None])
                if best_courier.state == 'IDLE':
                    travel_time = calculate_travel_time(best_courier.position, route_plan[0][0], constants['SPEED_HEX_PER_STEP'], constants['steps'])
                    best_courier.arrival_time = timestart + travel_time
                    best_courier.state = 'BUSY'
                
                for stop in route_plan:
                    if stop[1] == 'C' and stop[2] is not None and 'order_id' in stop[2]:
                        processed_ids_set.add(stop[2]['order_id'])
                
                # Unchanged metrics
                return True, metrics, assignment_log
            
    # End
    updated_metrics = (delay_inc, 0, success, success_delay, stacked_orders, rejected_orders)
    return False, updated_metrics, assignment_log
