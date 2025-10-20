import h3
import math
import abm_utils.abm as abm


def predict_courier_availability(couriers, MACRO_RESOLUTION):
    """
    Counts the free couriers based on the mandatory stops
    """
    macro_availability, idle_couriers = {}, []
    for courier in couriers:
        if courier.state == 'IDLE':
            h3_macro_index = h3.cell_to_parent(courier.position, MACRO_RESOLUTION)
            macro_availability[h3_macro_index] = macro_availability.get(h3_macro_index, 0) + 1
            idle_couriers.append(courier)
        elif courier.state == 'BUSY' and courier.mandatory_stops and courier.mandatory_stops[0][1] == 'B':
            target_hex = courier.mandatory_stops[0][0]
            h3_macro_index = h3.cell_to_parent(target_hex, MACRO_RESOLUTION)
            macro_availability[h3_macro_index] = macro_availability.get(h3_macro_index, 0) + 1
    return macro_availability, idle_couriers

def compute_surplus_deficit_per_zone(predicted_demand, available_couriers):

    """
    Calculates the surplus or deficit of couriers in our macro zones.
    The core calculation of the strategy, as the relative distribution of couriers is compared with the demand -
    which we predicted beforehand. To find imbalances in the fleet's deployment.
    Arg. predicted_demand: A dictionary of demand and hex_id for the current time bin (15 min)
    ->> in our case the pre_binned_demand 

    Arg. available couriers: A dictionary of couriers count and hex id for the current time bin (15 min)
    --> macro availability of predict_courier_availability function
    """

    total_predicted_demand = sum(predicted_demand.values()) #calculate percentage share
    total_available_couriers = sum(available_couriers.values()) #calculate percentage share

    if total_available_couriers == 0 or total_predicted_demand == 0:
        return {} #safety check

    all_zone_ids = set(predicted_demand.keys()) | set(available_couriers.keys()) #get a complete set of all unique zones and unify
    surplus_deficit_map = {}
    #we iterate through all zones
    for zone_id in all_zone_ids:
        demand_percentage = predicted_demand.get(zone_id, 0) / total_predicted_demand #calculate the share of demand
        supply_percentage = available_couriers.get(zone_id, 0) / total_available_couriers #calculate the share of couriers
        surplus_deficit_map[zone_id] = supply_percentage - demand_percentage #subtract demand share from supply share 
        # if positive = surplus, if negative = deficit

    return surplus_deficit_map #dict with hex ID and a relative score for surplus or deficit 

def identify_courier_demand_zones(surplus_deficit_map):
    
    """
    Processes the surplus / deficit to create a sorted list of all actionable zones
    
    """
    surplus_zones = [{'id': zid, 'value': val} for zid, val in surplus_deficit_map.items() if val > 0]
    deficit_zones = [{'id': zid, 'value': abs(val)} for zid, val in surplus_deficit_map.items() if val < 0]
    surplus_zones.sort(key=lambda x: x['value'], reverse=True)
    deficit_zones.sort(key=lambda x: x['value'], reverse=True) 
    return surplus_zones, deficit_zones

def execute_repositioning(idle_couriers, surplus_zones, deficit_zones, timestart, order_queue, speed_hex_per_step, steps, macro_resolution, work_resolution):
    """
    The execution of our repositioning strategy where we determine a max allowed travel time, 
    best source zones (based on Work resolution), best courier selection and a springboard logic
    for longer travels.

    """
    #Adjusting the maximum allowed travel time for repositioning move based on system stress (order queue)
    if len(order_queue) > 50:
        max_repo_time_seconds = 5 * 60
    elif len(order_queue) > 20:
        max_repo_time_seconds = 10 * 60
    else:
        max_repo_time_seconds = 15 * 60
    
    available_couriers = list(idle_couriers) #A list of all available couriers
    
    #we go through all deficit zones
    while deficit_zones and surplus_zones and available_couriers:
        deficit_zones.sort(key=lambda x: x['value'], reverse=True) 
        top_deficit = deficit_zones[0] #start with the most urgent zones
        
        #search for best source zone
        best_surplus_zone = None
        max_score = -float('inf')
        for s_zone in surplus_zones: #check if available couriers
            has_available_courier = any(h3.cell_to_parent(c.position, macro_resolution) == s_zone['id'] for c in available_couriers)
            if not has_available_courier: continue

            #high surplus and low distance = best
            dist = abm.get_hex_distance(s_zone['id'], top_deficit['id'])
            if dist == 0: continue
            score = s_zone['value'] / dist 
            if score > max_score:
                max_score = score
                best_surplus_zone = s_zone
        
        if best_surplus_zone:
            #find the best courier with shortest ETA
            courier_to_move = None
            min_eta_to_final_target = float('inf')
            final_target_hex = h3.cell_to_center_child(top_deficit['id'], work_resolution)
            
            candidate_couriers = [c for c in available_couriers if h3.cell_to_parent(c.position, macro_resolution) == best_surplus_zone['id']]
            for candidate in candidate_couriers: #we go through all couriers in the best surplus zone
                distance_hex = abm.get_hex_distance(candidate.position, final_target_hex)
                travel_steps = math.ceil(distance_hex / speed_hex_per_step)
                travel_time = travel_steps * steps
                if travel_time < min_eta_to_final_target: #find min ETA
                    min_eta_to_final_target = travel_time
                    courier_to_move = candidate
            
            actual_target_hex = final_target_hex
            actual_travel_time = min_eta_to_final_target

            # sometimes min ETA in best surplus zone is still to far away
            # Here we want to travel only 50% of the distance if its in our max repo time
            if actual_travel_time > max_repo_time_seconds: #max allowed travel time
                full_path = h3.grid_path_cells(courier_to_move.position, final_target_hex)
                if len(full_path) > 1:
                    midpoint_index = len(full_path) // 2
                    actual_target_hex = full_path[midpoint_index] #new target hex

                    # Test new distance
                    distance_hex = abm.get_hex_distance(courier_to_move.position, actual_target_hex)
                    travel_steps = math.ceil(distance_hex / speed_hex_per_step)
                    actual_travel_time = travel_steps * steps

            # We test the travel time (and also if too long the middistance with the max distance)
            if actual_travel_time <= max_repo_time_seconds:
                #print(f"  [REPOSITIONING] Sending courier {courier_to_move.id} to {actual_target_hex} (ETA: {actual_travel_time/60:.1f} min)")
                
                #Assign the task to the courier
                courier_to_move.state = 'BUSY'
                courier_to_move.arrival_time = timestart + actual_travel_time
                courier_to_move.mandatory_stops = [[actual_target_hex, 'B', None]]
                courier_to_move.was_repositioned = True

                #update the state for the next iteration of the while loop
                available_couriers.remove(courier_to_move)
                best_surplus_zone['value'] -= (1 / len(idle_couriers)) if idle_couriers else 0
                top_deficit['value'] -= (1 / len(idle_couriers)) if idle_couriers else 0
                if best_surplus_zone['value'] <= 0.01: surplus_zones.remove(best_surplus_zone)
                if top_deficit['value'] <= 0.01: deficit_zones.remove(top_deficit)
            else:
                deficit_zones.pop(0) #if even the best move is not feasible, this deficit zone cannot be served
        else:
            break

def run_repositioning_strategy(couriers, demand_map, timestart, order_queue, speed_hex_per_step, steps, macro_resolution, work_resolution):

    """
    Here the entire reposition strategy is triggered
    1. Assess the available couriers
    2. Calculate surplus and deficit balance
    3. Identify the zones
    4. Pass all the information into the execution function to perform matching and assignment
    """
    available_couriers, idle_couriers = predict_courier_availability(couriers, macro_resolution)
    if not idle_couriers:
        return
        
    surplus_deficit_map = compute_surplus_deficit_per_zone(demand_map, available_couriers)
    surplus_zones, deficit_zones = identify_courier_demand_zones(surplus_deficit_map)
    
    if surplus_zones and deficit_zones:
        execute_repositioning(idle_couriers, surplus_zones, deficit_zones, timestart, order_queue, speed_hex_per_step, steps, macro_resolution, work_resolution)