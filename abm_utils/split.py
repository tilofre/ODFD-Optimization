import h3
import abm_utils.abm as abm
import abm_utils.rejection as rejection
import random

def find_mid_meeting_point(start_hex, end_hex):
    """
    This version finds a point roughly in the middle of the path. (Thus not optimal, but will used at another calculation)
    """
    try:
        path = h3.grid_path_cells(start_hex, end_hex)
        if len(path) > 2:
            midpoint_index = len(path) // 2
            return path[midpoint_index]
        else:
            # If the path is very short, the restaurant is the best meeting point
            return start_hex
    except:
        return start_hex # Fallback



def find_dynamic_meeting_point_hex(order, courier1, courier2, timestart, constants):
    """
    Finds the best meeting point hexagon by checking the path between the restaurant
    and customer to minimize the handoff time for the two selected couriers.
    """
    restaurant_hex = order['sender_h3']
    customer_hex = order['recipient_h3']
    
    best_meeting_point = find_mid_meeting_point(restaurant_hex, customer_hex) # Start with the simple midpoint
    min_handoff_time = float('inf')

    try:
        path = h3.grid_path_cells(restaurant_hex, customer_hex)
    except:
        path = [best_meeting_point] # Fallback if path fails

    # Calculate time for courier 1 to get to the restaurant and wait for food
    c1_to_rest_time = abm.calculate_travel_time(courier1.position, restaurant_hex, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c1_arrival_at_rest = timestart + c1_to_rest_time
    c1_departs_rest = max(c1_arrival_at_rest, order['estimate_meal_prepare_time'])

    # Check each hexagon on the path as a potential meeting point
    for point in path:
        # Time for C1 to reach this point from the restaurant
        rest_to_meet_time = abm.calculate_travel_time(restaurant_hex, point, constants['SPEED_HEX_PER_STEP'], constants['steps'])
        c1_arrival_at_meet = c1_departs_rest + rest_to_meet_time
        
        # Time for C2 to reach this point directly
        c2_to_meet_time = abm.calculate_travel_time(courier2.position, point, constants['SPEED_HEX_PER_STEP'], constants['steps'])
        c2_arrival_at_meet = timestart + c2_to_meet_time
        
        # Handoff happens when the LATER of the two couriers arrives
        handoff_time = max(c1_arrival_at_meet, c2_arrival_at_meet)
        
        if handoff_time < min_handoff_time:
            min_handoff_time = handoff_time
            best_meeting_point = point
            
    return best_meeting_point

def process_split_delivery(order, idle_couriers, timestart, constants):
    """
    Finds a good pair of couriers for a split delivery using a fast heuristic
    instead of a brute-force search.
    """
    max_delay_threshold = constants.get('MAX_ACCEPTABLE_DELAY_SECONDS', 1800)
    
    if len(idle_couriers) < 2:
        return None, None, None, None

    restaurant_hex = order['sender_h3']
    customer_hex = order['recipient_h3']

    # --- HEURISTIC: Find the best couriers for each leg independently ---
    
    # Sort all couriers by their distance to the restaurant
    couriers_by_rest_dist = sorted(idle_couriers, key=lambda c: abm.get_hex_distance(c.position, restaurant_hex))
    c1 = couriers_by_rest_dist[0]

    # Sort all couriers by their distance to the customer
    couriers_by_cust_dist = sorted(idle_couriers, key=lambda c: abm.get_hex_distance(c.position, customer_hex))
    
    # Find the best courier for the second leg, making sure it's not the same as the first
    c2 = None
    for courier in couriers_by_cust_dist:
        if courier.id != c1.id:
            c2 = courier
            break
    
    # If we couldn't find two different couriers, we can't do a split
    if not c2:
        return None, None, None, None

    # --- Now, PLAN the delivery for this single, promising pair ---
    
    # Find the optimal meeting point for this pair
    meeting_point = find_dynamic_meeting_point_hex(order, c1, c2, timestart, constants)
    
    # Calculate the full timeline and final delay
    c1_to_rest_time = abm.calculate_travel_time(c1.position, restaurant_hex, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c1_arrival_at_rest = timestart + c1_to_rest_time
    c1_departs_rest = max(c1_arrival_at_rest, order['estimate_meal_prepare_time'])
    rest_to_meet_time = abm.calculate_travel_time(restaurant_hex, meeting_point, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c1_arrival_at_meet = c1_departs_rest + rest_to_meet_time

    c2_to_meet_time = abm.calculate_travel_time(c2.position, meeting_point, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c2_arrival_at_meet = timestart + c2_to_meet_time

    handoff_time = max(c1_arrival_at_meet, c2_arrival_at_meet)
    meet_to_cust_time = abm.calculate_travel_time(meeting_point, customer_hex, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    final_delivery_time = handoff_time + meet_to_cust_time
    
    total_delay = max(0, final_delivery_time - order['estimate_arrived_time'])

    # If the plan for this pair is acceptable, return their routes
    if total_delay <= max_delay_threshold:
        meeting_point_order = {
            'is_handover': True,  
            'handover_deadline': handoff_time, 
            'final_customer_hex': customer_hex, # target hex of c2
            'final_customer_order': order # complete order
        }
        route1 = [[restaurant_hex, 'R', order], [meeting_point, 'C', meeting_point_order]]
        route2 = [[meeting_point, 'R', None], [customer_hex, 'C', order]]
        return c1, c2, route1, route2
    
    return None, None, None, None

def execute_split_assignment(order, c1, c2, r1, r2, timestart, constants, rejection_model, processed_ids_set, next_queue,metrics):
    """
    Executes the assignment for a pre-planned split delivery.
    Offers the task sequentially to courier 1 and courier 2, handling rejections.
    Returns True if the order was processed (either fully assigned or Part 2 queued), False if C1 rejected.
    """

    delay_inc, _, success, success_delay, stacked_orders, rejected_orders = metrics
    # Offer task to Courier 1
    prob_rejection1 = rejection.predict_rejection_probability(order, rejection_model)
    if random.random() < prob_rejection1:
        c1.rejected_orders.append(order['order_id'])
        if order.get('phase') == 'tracked':
            rejected_orders += 1 # Increment rejection counter for C1
            updated_metrics = (delay_inc, 0, success, success_delay, stacked_orders, rejected_orders)
            return False, updated_metrics # C1 rejected, the entire plan fails for this cycle

    # C1 ACCEPTS

    # C1 ACCEPTS: Assign their route and dispatch them immediately
    c1.was_part_of_split = True
    #c1.is_on_initial_split_leg = True #its commented because if true, then no stacking is allowed (which might be good, but results are better with)
    c1.mandatory_stops = r1
    c1.handled_orders_info.append(order)
    c1.active_deliveries = 1
    travel_time1 = abm.calculate_travel_time(c1.position, r1[0][0], constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c1.arrival_time = timestart + travel_time1
    c1.state = 'BUSY'
    

    """
    We comment the rejection for part 2, as the code is differing the process and improve the results.
    Reason: First, c2 is the courier near by the restaurant. If declined, the meeting point is still the 
    handoff point. If c2 declines, the part2 is queued and a courier with a better position to the meeting 
    point can take the order. Its a problem of the handling and a chicken and egg problem discovered. Otherwise
    we have to implement a cost function to find the global optimum. Thus right now we let c2 pass..
    
    """
    # --- Offer task to Courier 2 ---
    # prob_rejection2 = rejection.predict_rejection_probability(order, rejection_model)
    # if random.random() < prob_rejection2:
    #     c2.rejected_orders.append(order['order_id'])
    #     #rejected_orders += 1
    #     # C1 is already moving, so we must queue Part 2
    #     order['assignment_status'] = 'pending_part2'
    #     order['sender_h3'] = r2[0][0] # The meeting point from the planned route
    #     order['estimate_meal_prepare_time'] = 0
    #     next_queue.append((order, 0))
    #     print(f"  [SPLIT PART 2 REJECTED] Order {order['order_id']} Part 1 assigned to C{c1.id}. Part 2 queued.")
    # else:
        # C2 ACCEPTS: Assign their route and dispatch them
    c2.was_part_of_split = True
    #c2.is_on_final_split_leg = True
    c2.mandatory_stops = r2
    c2.handled_orders_info.append(order) 
    c2.active_deliveries = 1
    travel_time2 = abm.calculate_travel_time(c2.position, r2[0][0], constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c2.arrival_time = timestart + travel_time2
    c2.state = 'BUSY'
    processed_ids_set.add(order['order_id'])
    print(f"  [SPLIT SUCCESS] Order {order['order_id']} planned and assigned to C{c1.id} and C{c2.id}")
    
    updated_metrics = (delay_inc, 0, success, success_delay, stacked_orders, rejected_orders)
    return True, updated_metrics # The order has been processed for this cycle

def normalize_distance(distance, min_dist, max_dist):
    if max_dist == min_dist:
        return 0.5 

    return (distance - min_dist) / (max_dist - min_dist)


def try_split_assignment(order, idle_couriers, timestart, constants, rejection_model, next_queue, processed_ids_set):
    """
    This is the main wrapper function for split deliveries.
    1. It plans the best split using 'process_split_delivery'.
    2. If a plan is found, it executes it using 'execute_split_assignment'.
    3. It returns the (success, delivery_time) tuple needed for the RL agent.
    """
    # Step 1: PLAN the delivery. We need the final delivery time from the plan.
    # NOTE: We will slightly modify process_split_delivery to return this time.
    c1, c2, r1, r2, planned_delivery_time = process_split_delivery(order, idle_couriers, timestart, constants)

    # If no valid plan could be made, the split fails immediately.
    if not c1:
        return False, 0

    # Step 2: EXECUTE the assignment for the found plan.
    was_processed = execute_split_assignment(
        order, c1, c2, r1, r2, timestart, constants,
        rejection_model, processed_ids_set, next_queue
    )

    # Step 3: Determine the final outcome for the RL agent.
    # The assignment is only a 'success' for the RL if it was fully assigned in this step
    # (i.e., not partially queued) AND the initial execution call was successful.
    if was_processed and order['order_id'] in processed_ids_set:
        # The delivery time is the one we calculated during the planning phase.
        delivery_duration = planned_delivery_time - order['platform_order_time']
        return True, delivery_duration
    else:
        # This covers cases where C1 rejected, or C2 rejected and the order was queued.
        # From the RL agent's perspective for this timestep, it was not a success.
        return False, 0