# split.py (Updated)

import h3
import math
import q_utils.abmQ as abm3
import q_utils.rejection as rejection2
import random

def find_optimal_meeting_point(start_hex, end_hex):
    """
    Finds an optimal meeting point along the path between two hexagons.
    This version finds a point roughly in the middle of the path.
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
    
    best_meeting_point = find_optimal_meeting_point(restaurant_hex, customer_hex) # Start with the simple midpoint
    min_handoff_time = float('inf')

    try:
        path = h3.grid_path_cells(restaurant_hex, customer_hex)
    except:
        path = [best_meeting_point] # Fallback if path fails

    # Calculate time for courier 1 to get to the restaurant and wait for food
    c1_to_rest_time = abm3.calculate_travel_time(courier1.position, restaurant_hex, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c1_arrival_at_rest = timestart + c1_to_rest_time
    c1_departs_rest = max(c1_arrival_at_rest, order['estimate_meal_prepare_time'])

    # Check each hexagon on the path as a potential meeting point
    for point in path:
        # Time for C1 to reach this point from the restaurant
        rest_to_meet_time = abm3.calculate_travel_time(restaurant_hex, point, constants['SPEED_HEX_PER_STEP'], constants['steps'])
        c1_arrival_at_meet = c1_departs_rest + rest_to_meet_time
        
        # Time for C2 to reach this point directly
        c2_to_meet_time = abm3.calculate_travel_time(courier2.position, point, constants['SPEED_HEX_PER_STEP'], constants['steps'])
        c2_arrival_at_meet = timestart + c2_to_meet_time
        
        # Handoff happens when the LATER of the two couriers arrives
        handoff_time = max(c1_arrival_at_meet, c2_arrival_at_meet)
        
        if handoff_time < min_handoff_time:
            min_handoff_time = handoff_time
            best_meeting_point = point
            
    return best_meeting_point

# def process_split_delivery(order, couriers, timestart, constants, rejection_model, processed_ids_set, next_queue):
#     """
#     Processes a split delivery in a hexagon environment by adapting the logic from
#     the coordinate-based process_split_delivery function.
    
#     Returns True if the order processing is complete for this cycle, False if it needs to be requeued.
#     """
#     idle_couriers = [c for c in couriers if c.state == 'IDLE' and order['order_id'] not in c.rejected_orders]
#     if len(idle_couriers) < 1:
#         return False # Not enough couriers, requeue whole order

#     # --- Find and assign first courier (Restaurant to Meeting Point) ---
#     idle_couriers.sort(key=lambda c: abm.get_hex_distance(c.position, order['sender_h3']))
#     courier1 = idle_couriers[0]
    
#     prob_rejection1 = rejection.predict_rejection_probability(order, rejection_model)
#     if random.random() < prob_rejection1:
#         courier1.rejected_orders.append(order['order_id'])
#         return False # Part 1 rejected, requeue whole order
    
#     # Courier 1 ACCEPTS: Update state using simulation's format
#     print(f"  [SPLIT PART 1] Order {order['order_id']} Part 1 assigned to Courier {courier1.id}.")
#     courier1.was_part_of_split = True
#     # NOTE: A temporary route is assigned here, which will be updated if a second courier is found.
#     # We use a simple midpoint for this temporary assignment.
#     temp_meeting_point = find_optimal_meeting_point(order['sender_h3'], order['recipient_h3'])
#     route1 = [[order['sender_h3'], 'R', order], [temp_meeting_point, 'C', None]]
#     courier1.mandatory_stops = route1
#     courier1.active_deliveries = 1
#     travel_time1 = abm.calculate_travel_time(courier1.position, route1[0][0], constants['SPEED_HEX_PER_STEP'], constants['steps'])
#     courier1.arrival_time = timestart + travel_time1
#     courier1.state = 'BUSY'

#     # --- Find and assign second courier (Meeting Point to Customer) ---
#     part2_assigned = False
#     remaining_idle = [c for c in idle_couriers if c.id != courier1.id]
#     if remaining_idle:
#         # Find a candidate for courier 2
#         simple_midpoint = find_optimal_meeting_point(order['sender_h3'], order['recipient_h3'])
#         remaining_idle.sort(key=lambda c: abm.get_hex_distance(c.position, simple_midpoint))
#         courier2 = remaining_idle[0]
        
#         prob_rejection2 = rejection.predict_rejection_probability(order, rejection_model)
        
#         if random.random() >= prob_rejection2: # Courier 2 ACCEPTS
#             # Calculate the DYNAMIC meeting point now that we have both couriers
#             meeting_point_hex = find_dynamic_meeting_point_hex(order, courier1, courier2, timestart, constants)
            
#             # Update Courier 1's route with the new, more optimal meeting point
#             route1 = [[order['sender_h3'], 'R', order], [meeting_point_hex, 'C', None]]
#             courier1.mandatory_stops = route1
            
#             # Assign Courier 2 with the optimal route
#             courier2.was_part_of_split = True
#             route2 = [[meeting_point_hex, 'R', None], [order['recipient_h3'], 'C', order]]
#             courier2.mandatory_stops = route2
#             courier2.active_deliveries = 1
#             travel_time2 = abm.calculate_travel_time(courier2.position, route2[0][0], constants['SPEED_HEX_PER_STEP'], constants['steps'])
#             courier2.arrival_time = timestart + travel_time2
#             courier2.state = 'BUSY'
            
#             processed_ids_set.add(order['order_id'])
#             part2_assigned = True
#         else:
#             courier2.rejected_orders.append(order['order_id'])

#     if part2_assigned:
#         print(f"  [SPLIT SUCCESS] Order {order['order_id']} fully split with Courier {courier2.id} for Part 2.")
#         return True
#     else:
#         # If Part 2 was rejected or no courier was available, queue it up.
#         # Courier 1 continues to the simple midpoint as their task is already set.
#         print(f"  [SPLIT PART 2 PENDING] Order {order['order_id']} Part 2 queued.")
#         order['assignment_status'] = 'pending_part2'
#         order['sender_h3'] = temp_meeting_point # The pickup is now the temp meeting point
#         order['estimate_meal_prepare_time'] = 0 # No prep time for a handoff
#         next_queue.append((order, 0))
#         return True # Order processed for this cycle
    

# In the cell with your helper functions, replace the planning function with this one.

def process_split_delivery(order, idle_couriers, timestart, constants):
    """
    Finds a good pair of couriers for a split delivery using a fast heuristic
    instead of a brute-force search.
    """
    max_delay_threshold = constants.get('MAX_ACCEPTABLE_DELAY_SECONDS', 1800)
    
    if len(idle_couriers) < 2:
        return None, None, None, None, 0

    restaurant_hex = order['sender_h3']
    customer_hex = order['recipient_h3']

    # --- HEURISTIC: Find the best couriers for each leg independently ---
    
    # Sort all couriers by their distance to the restaurant
    couriers_by_rest_dist = sorted(idle_couriers, key=lambda c: abm3.get_hex_distance(c.position, restaurant_hex))
    c1 = couriers_by_rest_dist[0]

    # Sort all couriers by their distance to the customer
    couriers_by_cust_dist = sorted(idle_couriers, key=lambda c: abm3.get_hex_distance(c.position, customer_hex))
    
    # Find the best courier for the second leg, making sure it's not the same as the first
    c2 = None
    for courier in couriers_by_cust_dist:
        if courier.id != c1.id:
            c2 = courier
            break
    
    # If we couldn't find two different couriers, we can't do a split
    if not c2:
        return None, None, None, None, 0

    # --- Now, PLAN the delivery for this single, promising pair ---
    
    # 1. Find the optimal meeting point for this pair
    meeting_point = find_dynamic_meeting_point_hex(order, c1, c2, timestart, constants)
    
    # 2. Calculate the full timeline and final delay
    c1_to_rest_time = abm3.calculate_travel_time(c1.position, restaurant_hex, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c1_arrival_at_rest = timestart + c1_to_rest_time
    c1_departs_rest = max(c1_arrival_at_rest, order['estimate_meal_prepare_time'])
    rest_to_meet_time = abm3.calculate_travel_time(restaurant_hex, meeting_point, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c1_arrival_at_meet = c1_departs_rest + rest_to_meet_time

    c2_to_meet_time = abm3.calculate_travel_time(c2.position, meeting_point, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c2_arrival_at_meet = timestart + c2_to_meet_time

    handoff_time = max(c1_arrival_at_meet, c2_arrival_at_meet)
    meet_to_cust_time = abm3.calculate_travel_time(meeting_point, customer_hex, constants['SPEED_HEX_PER_STEP'], constants['steps'])
    final_delivery_time = handoff_time + meet_to_cust_time
    
    total_delay = max(0, final_delivery_time - order['estimate_arrived_time'])

    # 3. If the plan for this pair is acceptable, return their routes
    # if total_delay <= max_delay_threshold:
    #     route1 = [[restaurant_hex, 'R', order], [meeting_point, 'C', None]]
    #     route2 = [[meeting_point, 'R', None], [customer_hex, 'C', order]]
    #     return c1, c2, route1, route2, final_delivery_time

    if total_delay <= max_delay_threshold:
        meeting_point_order = {
            'estimate_arrived_time': handoff_time
        }
        route1 = [
            [order['sender_h3'], 'R', order],
            [meeting_point, 'C', meeting_point_order] # <-- Der Fake-Auftrag wird hier genutzt
        ]
        route2 = [
            [meeting_point, 'R', None], # <-- C2 holt 'nichts' ab, nur Übergabe
            [order['recipient_h3'], 'C', order]
        ]
        return c1, c2, route1, route2, final_delivery_time
    
    return None, None, None, None, final_delivery_time

def execute_split_assignment(order, c1, c2, r1, r2, timestart, constants, rejection_model, processed_ids_set, next_queue):
    """
    Executes the assignment for a pre-planned split delivery.
    Offers the task sequentially to courier 1 and courier 2, handling rejections.
    Returns True if the order was processed (either fully assigned or Part 2 queued), False if C1 rejected.
    """
    # --- Offer task to Courier 1 ---
    prob_rejection1 = rejection2.predict_rejection_probability(order, rejection_model)
    if random.random() < prob_rejection1:
        c1.rejected_orders.append(order['order_id'])
        return False # C1 rejected, the entire plan fails for this cycle

    # C1 ACCEPTS: Assign their route and dispatch them immediately
    c1.was_part_of_split = True
    c1.mandatory_stops = r1
    c1.active_deliveries = 1
    travel_time1 = abm3.calculate_travel_time(c1.position, r1[0][0], constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c1.arrival_time = timestart + travel_time1
    c1.state = 'BUSY'
    
    # --- Offer task to Courier 2 ---
    # prob_rejection2 = rejection2.predict_rejection_probability(order, rejection_model)
    # if random.random() < prob_rejection2:
    #     c2.rejected_orders.append(order['order_id'])
    #     # C1 is already moving, so we must queue Part 2
    #     order['assignment_status'] = 'pending_part2'
    #     order['sender_h3'] = r2[0][0] # The meeting point from the planned route
    #     order['estimate_meal_prepare_time'] = 0
    #     next_queue.append((order, 0))
    #     #print(f"  [SPLIT PART 2 REJECTED] Order {order['order_id']} Part 1 assigned to C{c1.id}. Part 2 queued.")
    # else: 
    #     # C2 ACCEPTS: Assign their route and dispatch them
    c2.was_part_of_split = True
    c2.mandatory_stops = r2
    c2.active_deliveries = 1
    travel_time2 = abm3.calculate_travel_time(c2.position, r2[0][0], constants['SPEED_HEX_PER_STEP'], constants['steps'])
    c2.arrival_time = timestart + travel_time2
    c2.state = 'BUSY'
    processed_ids_set.add(order['order_id'])
        #print(f"  [SPLIT SUCCESS] Order {order['order_id']} planned and assigned to C{c1.id} and C{c2.id}")
    
    return True # The order has been processed for this cycle

# Add this new function to your split.py file

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