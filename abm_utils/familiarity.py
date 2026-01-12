import h3
import math
import pandas as pd
from collections import Counter
from functools import lru_cache

def create_courier_blueprints(df, threshold_ratio=0.6):

    blueprints = {}
    
    if 'platform_order_time' in df.columns:
        df = df.sort_values('platform_order_time')

    grouped = df.groupby('courier_id')

    for real_id, group in grouped:

        start_hex = group.iloc[0]['grab_h3']
        history_hexes = []
        for _, row in group.iterrows():
            if pd.notna(row.get('grab_h3')):
                history_hexes.append(row['grab_h3'])
            
            if row.get('is_courier_grabbed', 0) != 0:
                if pd.notna(row.get('sender_h3')):
                    history_hexes.append(row['sender_h3'])
                if pd.notna(row.get('recipient_h3')):
                    history_hexes.append(row['recipient_h3'])
        
        familiar_zone_set = get_adaptive_familiar_zone(history_hexes, threshold_ratio=threshold_ratio)
        
        blueprints[real_id] = {
            'start_hex': start_hex,
            'familiar_zone': familiar_zone_set
        }
        
    return blueprints, list(blueprints.keys())
                

@lru_cache(maxsize=50000)
def get_res8_path_ratio(start_res8, end_res8, familiar_zone_frozenset):
    try:
        path = h3.grid_path_cells(start_res8, end_res8)
        if not path: return 0.0
        
        total_cells = len(path)
        familiar_count = sum(1 for h in path if h in familiar_zone_frozenset)
        
        return familiar_count / total_cells
    except:
        hits = 0
        if start_res8 in familiar_zone_frozenset: hits += 1
        if end_res8 in familiar_zone_frozenset: hits += 1
        return hits / 2.0

def determine_familiarity_zone(hex_list, parent_res):
    if not hex_list: 
        return None
    try: 
        parents = [h3.cell_to_parent(h, parent_res) for h in hex_list if h]
        if not parents:
            return None
        return Counter(parents).most_common(1)[0][0]
    except Exception:
        return None
    

def get_adaptive_familiar_zone(history_hexes, threshold_ratio=0.6):
    if not history_hexes:
        return set()

    valid_hexes = [h for h in history_hexes if h and h3.is_valid_cell(h)]
    if not valid_hexes:
        return set()
    
    total_points = len(valid_hexes)
    
    parents_res7 = [h3.cell_to_parent(h, 7) for h in valid_hexes]
    if not parents_res7: return set()
    
    best_res7, count_res7 = Counter(parents_res7).most_common(1)[0]
    
    coverage_res7 = count_res7 / total_points
    
    if coverage_res7 >= threshold_ratio:
        return set(h3.cell_to_children(best_res7, 8))
    parents_res8 = [h3.cell_to_parent(h, 8) for h in valid_hexes]
    top_7_res8 = [item[0] for item in Counter(parents_res8).most_common(7)]

    return set(top_7_res8)

def get_lat_lon(h3_cell):
    try:
        lat, lon = h3.cell_to_latlng(h3_cell)
        if abs(lat) < 1.0: return None, None
        return lat, lon
    except:
        return None, None
    
# def calculate_travel_time(start_hex, end_hex, normal_speed, fast_speed, steps, familiar_zone_set):
#     try:
#         dist = h3.grid_distance(start_hex, end_hex)
#     except:
#         return float('inf')
        
#     if dist == 0: return 0

#     base_time = math.ceil(dist / normal_speed) * steps

#     if not familiar_zone_set:
#         return base_time
    
    

#     start_res8 = h3.cell_to_parent(start_hex, 8)
#     end_res8 = h3.cell_to_parent(end_hex, 8)

#     in_start = start_res8 in familiar_zone_set
#     in_end = end_res8 in familiar_zone_set

#     if in_start and in_end:
#         return math.ceil(dist / fast_speed) * steps

#     if not in_start and not in_end:
#         return base_time

#     try:
#         path = h3.grid_path_cells(start_hex, end_hex)
#         n = len(path)
#         low, high = 0, n - 1
#         boundary = 0

#         while low <= high:
#             mid = (low + high) // 2
#             mid_res8 = h3.cell_to_parent(path[mid], 8)
#             in_mid = (mid_res8 in familiar_zone_set)

#             if in_mid == in_start:
#                 boundary = mid
#                 low = mid + 1
#             else:
#                 high = mid - 1

#         steps_start_state = boundary
#         steps_end_state = (n - 1) - boundary

#         if in_start: 
#             time_steps = (steps_start_state / fast_speed) + (steps_end_state / normal_speed)
#         else: 
#             time_steps = (steps_start_state / normal_speed) + (steps_end_state / fast_speed)
#         return math.ceil(time_steps) * steps
        

#     except:
#         return base_time

def calculate_travel_time(start_hex, end_hex, normal_speed, fast_speed, steps, familiar_zone_set):
    try:
        total_dist_r13 = h3.grid_distance(start_hex, end_hex)
    except:
        return float('inf')
        
    if total_dist_r13 == 0: return 0

    if not familiar_zone_set or normal_speed == fast_speed:
        return math.ceil(total_dist_r13 / normal_speed) * steps

    start_res8 = h3.cell_to_parent(start_hex, 8)
    end_res8 = h3.cell_to_parent(end_hex, 8)

    in_start = start_res8 in familiar_zone_set
    in_end = end_res8 in familiar_zone_set

    if start_res8 == end_res8:
        speed = fast_speed if in_start else normal_speed
        return math.ceil(total_dist_r13 / speed) * steps

    if in_start and in_end:
        return math.ceil(total_dist_r13 / fast_speed) * steps

    if not in_start and not in_end:
        return math.ceil(total_dist_r13 / normal_speed) * steps

    frozen_zone = frozenset(familiar_zone_set)
    
    ratio_familiar = get_res8_path_ratio(start_res8, end_res8, frozen_zone)

    dist_fast = total_dist_r13 * ratio_familiar
    dist_slow = total_dist_r13 * (1.0 - ratio_familiar)

    time_steps = (dist_fast / fast_speed) + (dist_slow / normal_speed)
    
    return math.ceil(time_steps) * steps