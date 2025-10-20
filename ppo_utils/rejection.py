import pandas as pd
import h3

# Ihre 'get_hex_distance'-Funktion (bleibt unverändert)
DISTANCE_CACHE = {}
def get_hex_distance(start_hex, end_hex):
    key = tuple(sorted((start_hex, end_hex)))
    if key in DISTANCE_CACHE:
        return DISTANCE_CACHE[key]
    try:
        distance = h3.grid_distance(start_hex, end_hex)
        DISTANCE_CACHE[key] = distance
        return distance
    except (h3.H3FailedError, TypeError):
        return float('inf')


def predict_rejection_probability(order_object, model):
        
    try:
        
        # Das ist die korrekte Logik: H3-Index direkt aus dem Objekt auslesen
        sender_h3 = order_object['sender_h3']
        recipient_h3 = order_object['recipient_h3']
        order_time = pd.to_datetime(order_object['platform_order_time'], unit='s')
        # Features berechnen
        hex_dist = get_hex_distance(sender_h3, recipient_h3) # get_hex_distance ist ja im Notebook definiert
        is_weekend_val = 1 if order_time.dayofweek >= 5 else 0
        push_hour_val = order_time.hour
        
        features_for_prediction = {
            'is_weekend': is_weekend_val,
            'push_hour': push_hour_val,
            'hex_distance': hex_dist
        }

        df_for_prediction = pd.DataFrame([features_for_prediction])
        
        # Vorhersage
        prob = model.predict_proba(df_for_prediction)[:, 0]
                
        return prob[0]

    except Exception as e:
        print("\n---!!! FEHLER INNERHALB DER DEBUG-FUNKTION !!!---")
        print(f"Fehlertyp: {type(e)}")
        print(f"Fehlermeldung: {e}")
        print("Das 'order_object', das den Fehler verursacht hat, ist:")
        print(order_object)
        # Wir lösen den Fehler erneut aus, um den vollen Traceback zu sehen
        raise e