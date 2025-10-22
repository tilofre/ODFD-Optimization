# ODFD-Optimization


1. **Repository klonen:**
    Open Terminal:
    git clone https://github.com/tilofre/ODFD-Optimization.git odfd
    cd odfd

2. **Virtual Environment**
    Create new environment (recommended)
    python -m venv venv
    source venv/bin/activate 

3. **Install requirements**
    Here we install all necessary libraries in their version needed
    pip install -r requirements.txt

4. **Important data set**
    We use files already prepared for all cases. Nevertheless, the files are based on the all_wabill_info_meituan_0322.csv.
    It contains the delivery data of 8 days and can be accessed here: https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge

5. **Hands-On**
    Everything is done. The files are structured as followed:
    For demand forecasting different models:
        - ARIMA.ipynb
        - GCN-LSTM.ipynb
        - LSTM.ipynb
        - RF.ipynb (The one used for creating predicted_values.parquet)
        - XGBoost.ipynb 
    These models are used for predicting demand to use for the proactive repositioning within the ABM environment.

    To use the ABM for testing and evaluate the strategies use ABM.ipynb. It uses the abm_utils in te folder named similarly.
    Within the util folder are followed files needed:
        - abm.py -> The mind behind managing the abm by initializing couriers, moving them, update states, assign orders and calculate delays
        - rejection.py -> We trained a Logistic regression model based on actual rejections. The helper function takes the values and calculates a probability to reject an order.
        - repositioning.py -> all algorithms needed for calculating deficits, searching for optimal couriers and executing the repositioning 
        - split.py -> the functions needed for using the split delivery triggered in the ABM. Finds two couriers and assigning the orders
        - hindsight_batching.py -> Another strategy used for the SPAB. Evaluates the option of bringing an order going through several steps and assigning the tasks

    PPO_train.ipynb is notebook to train the PPO agent to decide at what situation to split. The helpers are in ppo_utils
        - abm.py. rejection.py, repositioning.py and splitPPO.py are most of all the same as in the abm_util. Nevertheless there are some slight differences, thus these are adapted and modified for the PPO and Q-Learning
        - ppo_agent.py -> all classes and functions needed for the PPO with its agents formulation, statehandler, representations, reward calculation and so on
        - PPOAgentVisualizer.py and PPOSystemMetrics.py are both files with visualisations used in PPO_train.ipynb

    PPO_test.ipynb is the notebook to test the PPO agent within the ABM environment. It decides which order needs to be split based on the trained agent. (Uses the abm_utils and the ppo_agent.py file)

    Ql_train.ipynb is the notebook to the train the Q-Learning agent. The helpers are in q_utils and ppo_utils (I know, its a bit confusing, could have combined it)
        - The same helpers as the PPO but specified for the Q-Learning approach, as they differ in state handlings
    
    Ql_test.ipynb is the notebook to test the Ql-agent. Same as for the PPO agent. BUT: it cannot be used as long a new training is done by yourself, as the pickle files are only usable in my local environment (Idk why, but pickle is weird regarding this)