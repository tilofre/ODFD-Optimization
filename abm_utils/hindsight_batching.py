
"""
This module contains all logic related to "hindsight analysis" for the ABM simulation.
Hindsight analysis allows the system to look back at recently assigned and unassigned
orders to find opportunities for "bringer" actions, where one courier (the bringer)
picks up a new order and hands it off to another courier (the target) who is
already busy, saving time and resources.
"""

import abm_utils.abm as abm
import pandas as pd
from itertools import permutations, product, combinations
import h3
import warnings

def handle_hindsight_analysis(timestart, order_queue, new_orders_this_step, couriers, constants, hindsight_cache):
    """
    Performs the entire hindsight analysis and action.
    This function is outsourced to keep the run_abm function clean.
    """
    HINDSIGHT_INTERVAL = 30
    HINDSIGHT_WINDOW = 300
    processed_order_ids_this_step = set()
    used_courier_ids_this_step = set()

    #Trigger condition: Only execute at specific intervals
    is_triggered = (timestart - constants.get('initial_timestart', 0)) % HINDSIGHT_INTERVAL == 0 and timestart > constants.get('initial_timestart', 0)
    
    if not is_triggered:
        # If not triggered, return unchanged values.
        # IMPORTANT: “new_orders_this_step” must be in the return statement.
        return couriers, order_queue, new_orders_this_step, hindsight_cache, used_courier_ids_this_step, processed_order_ids_this_step
    # '
    assigned_orders, unassigned_orders = review_time_window(
        timestart, order_queue, new_orders_this_step, couriers, window_seconds=HINDSIGHT_WINDOW
    )
    
    potential_batches, hindsight_cache = hindsight_analysis(assigned_orders, unassigned_orders, couriers, timestart, constants, hindsight_cache)

    potential_batches.sort(key=lambda x: x['saving_minutes'], reverse=True)

    for batch_info in potential_batches:
        hindsight_batch_info = {'orders': [o['order_id'] for o in batch_info['orders']]}
        relevant_orders = assigned_orders + unassigned_orders
        
        bringer_result = evaluate_bringer_courier_option(hindsight_batch_info, relevant_orders, couriers, timestart, constants, used_courier_ids=used_courier_ids_this_step)

        if bringer_result and bringer_result.get('decision') == "Yes":
            bringer_id, target_id, unassigned_order = bringer_result['bringer_id'], bringer_result['target_id'], bringer_result['unassigned_order']
            
            bringer_courier = next((c for c in couriers if c.id == bringer_id), None)
            target_courier = next((c for c in couriers if c.id == target_id), None)

            if bringer_courier and target_courier:
                bringer_courier.mandatory_stops = bringer_result['bringer_route']
                target_courier.mandatory_stops = bringer_result['target_new_route']

                bringer_courier.active_deliveries = len([s for s in bringer_courier.mandatory_stops if s[1] in ['C', 'H']])
                target_courier.active_deliveries = len([s for s in target_courier.mandatory_stops if s[1] == 'C'])
                
                if bringer_courier.state == 'IDLE':
                    travel_time = abm.calculate_travel_time(bringer_courier.position, bringer_result['bringer_route'][0][0], constants['SPEED_HEX_PER_STEP'], constants['steps'])
                    bringer_courier.arrival_time = timestart + travel_time
                    bringer_courier.state = 'BUSY'
                
                used_courier_ids_this_step.add(bringer_id)
                used_courier_ids_this_step.add(target_id)
                processed_order_ids_this_step.add(unassigned_order['order_id'])
                
                # Remove order from list
                order_queue[:] = [o for o in order_queue if o[0]['order_id'] != unassigned_order['order_id']]
                new_orders_this_step[:] = [o for o in new_orders_this_step if o['order_id'] != unassigned_order['order_id']]

    return couriers, order_queue, new_orders_this_step, hindsight_cache, used_courier_ids_this_step, processed_order_ids_this_step

#Suppress potential H3 distance warnings if cells are not neighbors
warnings.filterwarnings("ignore", message=".*grid_path_cells may be inefficient.*")

def review_time_window(timestart, order_queue, new_orders_this_step, couriers, window_seconds=300):
    """
    NEW VERSION: Returns two separate lists for assigned and unassigned
    orders within the current time window.

    Args:
        timestart (int): The current simulation timestamp.
        all_orders_df (pd.DataFrame): DataFrame of all orders (not used here, but part of signature).
        order_queue (list): The list of pending orders: [(order, attempts), ...].
        couriers (list): The list of all Courier objects.
        window_seconds (int): The lookback window

    Returns:
        tuple: (active_assigned_orders, pending_orders)
               - active_assigned_orders (list[dict]): Orders currently being handled by BUSY couriers.
               - pending_orders (list[dict]): Orders from the order_queue.
    """
    window_start_time = timestart - window_seconds
    active_assigned_orders = []
    active_assigned_order_ids = set() # To prevent duplicates

    for courier in couriers:
        if courier.state == 'BUSY':
            for stop in courier.mandatory_stops:
                order_info = stop[2] 
                order = order_info[0] if isinstance(order_info, list) and len(order_info) > 0 else order_info
                
                if isinstance(order, pd.Series):
                    order = order.to_dict()

                if isinstance(order, dict) and 'order_id' in order and order['order_id'] not in active_assigned_order_ids:
                    if order.get('platform_order_time', 0) >= window_start_time:
                        order['assignment_status'] = 'assigned'
                        active_assigned_orders.append(order)
                        active_assigned_order_ids.add(order['order_id'])

    # Get unassigned orders from the queue
    pending_orders_queue = [
        o[0] for o in order_queue 
        if o[0].get('platform_order_time', 0) >= window_start_time
    ]
    
    # Get unassigned orders from the new orders list
    pending_orders_new = [
        o for o in new_orders_this_step
        if o.get('platform_order_time', 0) >= window_start_time
    ]
    
    # Combine both lists
    pending_orders = pending_orders_queue + pending_orders_new
    
    return active_assigned_orders, pending_orders


def hindsight_analysis(assigned_orders, unassigned_orders, couriers, timestart, constants, hindsight_cache):
    """
    Analyzes potential pairings between assigned and unassigned orders
    to find profitable "bringer" opportunities.

    Args:
        assigned_orders (list[dict]): List of orders currently assigned to busy couriers.
        unassigned_orders (list[dict]): List of orders waiting in the queue.
        couriers (list[list]): List of all Courier objects.
        timestart (int): Current simulation timestamp.
        constants (dict): Simulation constants.
        hindsight_cache (set): A set of (order_id, order_id) pairs that have already
                               been evaluated to avoid redundant calculations.

    Returns:
        tuple: (final_batches, hindsight_cache)
               - final_batches (list[dict]): A list of profitable batches to be evaluated.
               - hindsight_cache (set): The updated cache of evaluated pairs.
    """
    # If there's nothing to pair, exit early
    if not unassigned_orders or not assigned_orders:
        return [], hindsight_cache

    MIN_SAVING_MINUTES = constants.get('HINDSIGHT_MIN_SAVING_MINUTES', 5)
    MAX_DIST_BETWEEN_RESTAURANTS_HEX = constants.get('HINDSIGHT_MAX_RESTAURANT_DIST_HEX', 30)
    
    potential_batches = []
    idle_couriers = [c for c in couriers if c.state == 'IDLE']
    
    # If no one is available to be a "bringer", exit early
    if not idle_couriers:
        return [], hindsight_cache

    # Use itertools.product to create all possible pairs of (assigned, unassigned)
    for assigned_order, unassigned_order in product(assigned_orders, unassigned_orders):
        
        # Create a unique key for this pair, regardless of order
        pair_key = tuple(sorted((assigned_order['order_id'], unassigned_order['order_id'])))
        
        try:
            # Distance between both restaurants, as a heuristic
            dist_between_restos = abm.get_hex_distance(
                assigned_order['sender_h3'],
                unassigned_order['sender_h3']
            )
        except Exception as e:
            dist_between_restos = float('inf')

        if dist_between_restos > MAX_DIST_BETWEEN_RESTAURANTS_HEX:
             continue
        

        # Skip if this pair has already been analyzed
        if pair_key in hindsight_cache:
            continue
        hindsight_cache.add(pair_key)

        # Find the courier who is handling the 'assigned_order'
        target_courier = None
        for c in couriers:
            if c.state == 'BUSY':
                # Check all stops for this order_id
                # CORRECTION: Check for `s[2] is not None` to prevent ValueError
                if any(s[2] is not None and s[2].get('order_id') == assigned_order['order_id'] for s in c.mandatory_stops):
                    target_courier = c
                    break
        
        # If we couldn't find the courier (e.g., just finished) or they have no stops, skip
        if not target_courier or not target_courier.mandatory_stops:
            continue
        
        # --- HEURISTIC CALCULATION (QUICK CHECK) ---
        # This is a rough estimate to see if a full analysis is even worth it.
        
        # 1. Cost of separate delivery (using the *best* idle courier)
        best_idle_courier = min(idle_couriers, key=lambda c: abm.get_hex_distance(c.position, unassigned_order['sender_h3']))
        cost_separate_delivery, _ = calculate_best_route_for_order(best_idle_courier, unassigned_order, constants)
        
        # 2. Heuristic cost of a detour for the target courier
        current_pos = target_courier.position
        next_original_stop_pos = target_courier.mandatory_stops[0][0]
        new_resto_pos = unassigned_order['sender_h3']
        new_cust_pos = unassigned_order['recipient_h3']
        
        # Calculate time for the original path segment
        time_without_detour = abm.calculate_travel_time(current_pos, next_original_stop_pos, constants['SPEED_HEX_PER_STEP'], constants['steps'])
        
        # Calculate time for a naive detour (Current -> New Resto -> New Cust -> Original Stop)
        # This assumes the target courier does the pickup themselves
        time_with_detour = (abm.calculate_travel_time(current_pos, new_resto_pos, constants['SPEED_HEX_PER_STEP'], constants['steps']) +
                            abm.calculate_travel_time(new_resto_pos, new_cust_pos, constants['SPEED_HEX_PER_STEP'], constants['steps']) +
                            abm.calculate_travel_time(new_cust_pos, next_original_stop_pos, constants['SPEED_HEX_PER_STEP'], constants['steps']))
        
        cost_detour_heuristic = time_with_detour - time_without_detour
        
        # Potential saving is the difference
        potential_saving_seconds = cost_separate_delivery - cost_detour_heuristic
        
        # If the heuristic shows potential savings, add it for a full evaluation
        if (potential_saving_seconds / 60) > MIN_SAVING_MINUTES:
            batch_info = {
                'orders': [assigned_order, unassigned_order],
                'saving_minutes': potential_saving_seconds / 60,
                # Store the target courier's ID so we don't have to find them again
                'target_courier_id': target_courier.id 
            }
            potential_batches.append(batch_info)
    
    if not potential_batches:
        return [], hindsight_cache

    # Sort to evaluate the most promising batches first
    potential_batches.sort(key=lambda x: x['saving_minutes'], reverse=True)
    
    # --- Final Filtering ---
    # Ensure each *unassigned* order is only in ONE potential batch
    # to avoid assigning it to multiple bringers.
    final_batches = []
    processed_unassigned_ids = set()
    for batch in potential_batches:
        # Find the unassigned order in the batch
        current_unassigned_order = next((o for o in batch['orders'] if o.get('assignment_status') != 'assigned'), None)
        
        # If we found it and it hasn't been processed yet
        if current_unassigned_order is not None and current_unassigned_order['order_id'] not in processed_unassigned_ids:
            # Find the assigned order
            assigned_order = next((o for o in batch['orders'] if o.get('assignment_status') == 'assigned'), None)
            
            if assigned_order is not None:
                # Store the target courier ID on the order object for easier access
                # in the evaluation function.
                assigned_order['courier_id'] = batch['target_courier_id']
                final_batches.append(batch)
                processed_unassigned_ids.add(current_unassigned_order['order_id'])
            
    return final_batches, hindsight_cache


def evaluate_bringer_courier_option(hindsight_batch_info, relevant_orders, couriers, timestart, constants, used_courier_ids=None):
    """
    FINAL 3-STAGE LOGIC: Performs a detailed evaluation of a "bringer" action.

    1. STAGE 1: Check if the target courier "self-pickup" is good enough.
       - If YES (ETA shift for existing orders is small): REJECT bringer action.
       - If NO (ETA shift is large or causes lateness): Proceed to Stage 2.
    2. STAGE 2: Evaluate the "bringer" action for:
       a) Pure time savings (vs. a separate idle courier).
       b) Absolute on-time delivery for ALL orders (existing + new).
    3. STAGE 3: Only if the bringer action satisfies BOTH (2a) and (2b), APPROVE it.

    Args:
        hindsight_batch_info (dict): The batch info from `hindsight_analysis`.
        relevant_orders (list[dict]): All assigned and unassigned orders.
        couriers (list): List of all Courier objects.
        timestart (int): Current simulation timestamp.
        constants (dict): Simulation constants.
        used_courier_ids (set): Set of courier IDs already used in this step.

    Returns:
        dict: A decision dictionary ({"decision": "Yes" or "No", ...}).
    """
    if used_courier_ids is None:
        used_courier_ids = set()
    
    # Filter out couriers already used in this hindsight step
    available_couriers = [c for c in couriers if c.id not in used_courier_ids]

    # === SETUP: Identify Orders and Couriers ===
    
    # Helper function to safely check a courier's stops for an order
    def has_order(stop, assigned_order_id):
        order_info = stop[2]
        # CORRECTION: Check for None payload
        if order_info is None: 
            return False
            
        order = order_info[0] if isinstance(order_info, list) and len(order_info) > 0 else order_info
        
        # CORRECTION: Check for None order after list extraction
        if order is None:
            return False
            
        if isinstance(order, (dict, pd.Series)):
            # CORRECTION: Use .get() for safe access, avoids error if key missing
            return order.get('order_id') == assigned_order_id
        return False

    order_ids_in_batch = hindsight_batch_info['orders']
    
    # Find the unassigned order in the batch
    unassigned_order = next((o for o in relevant_orders if o['order_id'] in order_ids_in_batch and o.get('assignment_status') != 'assigned'), None)
    # CORRECTION: Explicit `is None` check
    if unassigned_order is None: 
        return {"decision": "No", "reason": "No unassigned order found in batch."}

    # Find the assigned order ID
    assigned_order_id = next((oid for oid in order_ids_in_batch if oid != unassigned_order['order_id']), None)
    if assigned_order_id is None: 
        return {"decision": "No", "reason": "No assigned order ID found in batch."}

    # Find the target courier from the available pool
    target_courier = next((c for c in available_couriers if c.state == 'BUSY' and any(has_order(s, assigned_order_id) for s in c.mandatory_stops)), None)
    if target_courier is None: 
        return {"decision": "No", "reason": "Target courier is no longer busy or not found."}

    # Check capacity of the target courier
    MAX_DELIVERIES = 3
    if target_courier.active_deliveries >= MAX_DELIVERIES: 
        return {"decision": "No", "reason": f"Target courier at max capacity ({target_courier.active_deliveries}/{MAX_DELIVERIES})."}

    # Find available idle couriers to act as bringers
    idle_couriers = [c for c in available_couriers if c.state == 'IDLE']
    # CORRECTION: Check `if not list` instead of `is None`
    if not idle_couriers: 
        return {"decision": "No", "reason": "No idle couriers available for bringer role."}

    # --- STAGE 1: EVALUATE "SELF-PICKUP" OPTION ---
    # Is it "good enough" for the target courier to just pick up the new order?
    
    ETA_SHIFT_THRESHOLD_SECONDS = constants.get('CRITICAL_ETA_SHIFT_MINUTES', 3) * 60
    original_stops = target_courier.mandatory_stops
    
    # Get baseline ETAs for all current customers
    original_etas = calculate_etas_for_route(timestart, target_courier.position, original_stops, constants)
    
    # Simulate the new route with self-pickup (add restaurant + customer)
    new_pickup = [unassigned_order['sender_h3'], 'R', unassigned_order]
    new_delivery = [unassigned_order['recipient_h3'], 'C', unassigned_order]
    route_self_pickup = find_best_insertion_in_route(original_stops, new_pickup, target_courier.position, constants)
    route_self_pickup = find_best_insertion_in_route(route_self_pickup, new_delivery, target_courier.position, constants)
    
    # Calculate new ETAs for this simulated route
    etas_self_pickup = calculate_etas_for_route(timestart, target_courier.position, route_self_pickup, constants)
    
    # Calculate the *maximum delay* caused for *existing* customers
    max_eta_shift = 0
    original_customer_orders = {s[2]['order_id']: s[2] for s in original_stops if s[1] == 'C' and s[2] is not None and 'order_id' in s[2]}
    
    for order_id in original_customer_orders:
        shift = etas_self_pickup.get(order_id, 0) - original_etas.get(order_id, 0)
        if shift > max_eta_shift:
            max_eta_shift = shift
    
    # Check if *any* order (existing or new) becomes late in this scenario
    is_any_order_late_in_self_pickup = False
    all_orders_in_self_pickup_route = {**original_customer_orders, unassigned_order['order_id']: unassigned_order}
    
    for order_id, eta in etas_self_pickup.items():
        if order_id in all_orders_in_self_pickup_route:
            order_deadline = all_orders_in_self_pickup_route[order_id]['estimate_arrived_time']
            if eta > order_deadline:
                is_any_order_late_in_self_pickup = True
                break
            
    # DECISION 1: Is self-pickup "good enough"?
    if max_eta_shift <= ETA_SHIFT_THRESHOLD_SECONDS and not is_any_order_late_in_self_pickup:
        # YES. The delay is acceptable and no one is late.
        # Hindsight should NOT intervene. The standard assignment logic
        # will handle this as a normal stacking assignment.
        return {"decision": "No", "reason": f"Self-pickup is efficient enough (ETA-Shift: {max_eta_shift/60:.1f}min)."}

    # --- STAGE 2 & 3: EVALUATE BRINGER CASE ---
    # We only get here if self-pickup is a *bad* option (causes delays/lateness)

    # a) Calculate cost of a separate, normal delivery for the new order
    best_idle_courier = min(idle_couriers, key=lambda c: abm.get_hex_distance(c.position, unassigned_order['sender_h3']))
    cost_separate_delivery, _ = calculate_best_route_for_order(best_idle_courier, unassigned_order, constants)

    # b) Calculate cost and punctuality of the "bringer" action
    bringer_courier = min(idle_couriers, key=lambda c: abm.get_hex_distance(c.position, unassigned_order['sender_h3']))
    intercept_point, _ = find_optimal_intercept_point(unassigned_order, bringer_courier, target_courier, timestart, constants)
    
    if intercept_point is None: 
        return {"decision": "No", "reason": "No intercept point found for bringer action."}

    # Bringer's route: Their Position -> Restaurant -> Intercept Point
    bringer_route = [[unassigned_order['sender_h3'], 'R', unassigned_order], [intercept_point, 'H', unassigned_order]]
    cost_bringer_part = calculate_route_travel_time(bringer_courier.position, bringer_route, constants)
    
    # Target's original cost
    cost_target_original = calculate_route_travel_time(target_courier.position, original_stops, constants)
    
    # Target's new route: Their Position -> ... -> Handoff -> ... -> New Customer -> ...
    handoff_stop = [intercept_point, 'H', unassigned_order]
    route_with_customer = find_best_insertion_in_route(original_stops, new_delivery, target_courier.position, constants)
    target_new_route = find_best_insertion_in_route(route_with_customer, handoff_stop, target_courier.position, constants)
    
    # Target's new cost
    cost_target_new = calculate_route_travel_time(target_courier.position, target_new_route, constants)
    
    # Calculate total cost and savings
    cost_target_detour = cost_target_new - cost_target_original
    total_cost_bringer_action = cost_bringer_part + cost_target_detour
    
    # Condition 1: Does it save time compared to a separate delivery?
    time_saving = cost_separate_delivery - total_cost_bringer_action
    is_time_saving_positive = time_saving > 600

    # Condition 2: Are all orders (existing + new) still on time?
    etas_bringer_scenario = calculate_etas_for_route(timestart, target_courier.position, target_new_route, constants)
    is_any_order_late = False
    all_orders_in_new_route = {**original_customer_orders, unassigned_order['order_id']: unassigned_order}
    
    for order_id, eta in etas_bringer_scenario.items():
        if order_id in all_orders_in_new_route:
            order_deadline = all_orders_in_new_route[order_id]['estimate_arrived_time']
            if eta > order_deadline:
                is_any_order_late = True
                break

    # FINAL DECISION: Only approve if BOTH conditions are met
    if is_time_saving_positive and not is_any_order_late:
        return {
            "decision": "Yes", 
            "saving_minutes": time_saving / 60, 
            "bringer_id": bringer_courier.id, 
            "target_id": target_courier.id, 
            "meeting_point": intercept_point, 
            "bringer_route": bringer_route, 
            "target_new_route": target_new_route, 
            "unassigned_order": unassigned_order
        }
    else:
        # Provide a reason for rejection
        reason = f"Bringer does not save time (Saving: {time_saving/60:.1f}min)." if not is_time_saving_positive else "Bringer action would cause a delay."
        return {"decision": "No", "reason": reason}


# --- Route and Time Calculation Helpers ---

def calculate_route_travel_time(start_pos, route, constants):
    """
    Calculates the total travel time (in seconds) for a given route.

    Args:
        start_pos (str): The starting H3 hexagon.
        route (list): A list of stops: [[hex, type, order], ...].
        constants (dict): Simulation constants.

    Returns:
        int: Total travel time in seconds.
    """
    if not route: 
        return 0
        
    total_time = 0
    current_pos = start_pos
    for stop in route:
        stop_hex = stop[0]
        # Use the ABM's core travel time function for each segment
        travel_time = abm.calculate_travel_time(current_pos, stop_hex, constants['SPEED_HEX_PER_STEP'], constants['steps'])
        total_time += travel_time
        current_pos = stop_hex
    return total_time


def calculate_best_route_for_order(courier, order, constants):
    """
    Calculates the optimal route and total travel time for a courier
    to deliver ONE single order.

    Args:
        courier (Courier): The courier object.
        order (dict): The order dictionary.
        constants (dict): Simulation constants.

    Returns:
        tuple: (total_travel_time, route_plan)
    """
    # The route for a single order is always Pickup, then Delivery.
    route_plan = [
        [order['sender_h3'], 'R', order],      # 'R' for Restaurant (Pickup)
        [order['recipient_h3'], 'C', order]   # 'C' for Customer (Delivery)
    ]

    # Calculate travel time from the courier's current position
    travel_time = calculate_route_travel_time(courier.position, route_plan, constants)

    return travel_time, route_plan


def find_best_route_for_batch(batch_orders, start_pos):
    """
    Finds the best route (minimum distance) for a batch of orders
    using brute-force permutation.
    
    NOTE: This is computationally expensive and only suitable for very small
    batches (2-3 orders). It also calculates DISTANCE, not TIME.
    """
    best_route = []
    min_dist = float('inf')
    
    # Create lists of all pickup and dropoff stops
    pickups = [[o['sender_h3'], 'R', o] for o in batch_orders]
    for p_perm in permutations(pickups):
        dropoffs = [[o['recipient_h3'], 'C', o] for o in batch_orders]
        for d_perm in permutations(dropoffs):
            # Route must be all pickups, then all dropoffs
            route_candidate = list(p_perm) + list(d_perm)
            current_dist = 0
            last_pos = start_pos
            for stop in route_candidate:
                # Use hex_distance for this heuristic
                current_dist += abm.get_hex_distance(last_pos, stop[0])
                last_pos = stop[0]
                
            if current_dist < min_dist:
                min_dist = current_dist
                best_route = route_candidate
    return best_route, min_dist


def find_optimal_intercept_point(order, bringer_courier, target_courier, timestart, constants):
    """
    Finds the best handoff point by checking every H3 cell
    on the target courier's *entire* future path.
    """
    restaurant_hex = order['sender_h3']
    best_intercept_point = None
    min_handoff_time = float('inf')

    # 1. Calculate when the bringer will have the food ready
    bringer_to_rest_time = abm.calculate_travel_time(bringer_courier.position, restaurant_hex, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    bringer_arrival_at_rest = timestart + bringer_to_rest_time
    # Bringer can't leave until they arrive AND the food is ready
    bringer_departs_rest = max(bringer_arrival_at_rest, order['estimate_meal_prepare_time'])

    # 2. Build the target courier's full path from their mandatory stops
    full_target_path = []
    current_pos = target_courier.position
    if not target_courier.mandatory_stops:
        return None, float('inf') # No path, no intercept

    for stop in target_courier.mandatory_stops:
        next_stop_hex = stop[0]
        try:
            # Use h3.grid_path_cells to find the sequence of hexes
            segment_path = h3.grid_path_cells(current_pos, next_stop_hex)
            full_target_path.extend(list(segment_path))
            current_pos = next_stop_hex
        except h3.H3CellError:
            continue # Skip if a path can't be calculated

    # CORRECTION: Check `if not list` (for an empty list)
    if not full_target_path:
        return None, float('inf') # No path was built

    # 3. Iterate through every H3 cell in the path as a potential intercept point
    for intercept_point_candidate in full_target_path:
        
        # Time for target to reach the candidate cell
        target_travel_time = abm.calculate_travel_time(target_courier.position, intercept_point_candidate, constants['SPEED_HEX_PER_STEP'], constants['steps'])
        target_arrival_at_intercept = timestart + target_travel_time
        
        # Time for bringer to reach the candidate cell (after getting food)
        bringer_travel_time = abm.calculate_travel_time(restaurant_hex, intercept_point_candidate, constants['SPEED_HEX_PER_STEP'], constants['steps'])
        bringer_arrival_at_intercept = bringer_departs_rest + bringer_travel_time
        
        # The handoff can only happen when *both* are present
        handoff_time = max(target_arrival_at_intercept, bringer_arrival_at_intercept)
        
        # If this is the fastest handoff so far, save it
        if handoff_time < min_handoff_time:
            min_handoff_time = handoff_time
            best_intercept_point = intercept_point_candidate

    return best_intercept_point, min_handoff_time


def find_best_insertion_in_route(existing_route, new_stop, start_pos, constants):
    """
    Finds the best position to insert a new stop into an existing route
    by checking every possible insertion index.
    """
    # Start by assuming the best route is just appending the stop
    best_route = existing_route + [new_stop]
    min_travel_time = calculate_route_travel_time(start_pos, best_route, constants)
    
    # Iterate through all possible insertion points (including index 0)
    for i in range(len(existing_route) + 1):
        candidate_route = existing_route[:i] + [new_stop] + existing_route[i:]
        current_travel_time = calculate_route_travel_time(start_pos, candidate_route, constants)
        
        if current_travel_time < min_travel_time:
            min_travel_time = current_travel_time
            best_route = candidate_route
            
    return best_route


def calculate_etas_for_route(start_time, start_pos, route, constants):
    """
    Calculates the estimated time of arrival (ETA) for every
    customer stop ('C') in a given route.
    """
    etas = {}
    
    # Filter for customer stops that have valid order info
    # CORRECTION: Check `stop[2] is not None` to prevent ValueError
    customer_stops_with_info = [
        stop for stop in route 
        if stop[1] == 'C' and stop[2] is not None and 'order_id' in stop[2]
    ]

    for stop in customer_stops_with_info:
        stop_pos, _, order_info = stop
        
        # Find the path from the start to this specific stop
        path_to_stop = []
        temp_route = list(route) # Create a copy to iterate over
        
        for next_stop in temp_route:
            path_to_stop.append(next_stop)
            
            # Stop when we reach the target customer stop
            # CORRECTION: Add safety checks for `next_stop[2]`
            if (next_stop[0] == stop_pos and 
                next_stop[2] is not None and 
                'order_id' in next_stop[2] and 
                next_stop[2]['order_id'] == order_info['order_id']):
                break
        
        # Calculate the travel time for this partial path
        travel_time = calculate_route_travel_time(start_pos, path_to_stop, constants)
        eta = start_time + travel_time
        etas[order_info['order_id']] = eta
            
    return etas