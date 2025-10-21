import pandas as pd
import abm_utils.abm as abm 

'''
Predict the probability for rejection nased on trained model
'''
def predict_rejection_probability(order_object, model):
        
    try:
        sender_h3 = order_object['sender_h3']
        recipient_h3 = order_object['recipient_h3']
        order_time = pd.to_datetime(order_object['platform_order_time'], unit='s')
        # Calculate features
        hex_dist = abm.get_hex_distance(sender_h3, recipient_h3)
        is_weekend_val = 1 if order_time.dayofweek >= 5 else 0 #weekday or weekend
        push_hour_val = order_time.hour #order time hour
        
        features_for_prediction = {
            'is_weekend': is_weekend_val,
            'push_hour': push_hour_val,
            'hex_distance': hex_dist
        }

        df_for_prediction = pd.DataFrame([features_for_prediction])
        
        # prediction
        prob = model.predict_proba(df_for_prediction)[:, 0]
                
        return prob[0]

    except Exception as e:
        print("Errrroorororororooro mh")
        print(order_object)
        raise e