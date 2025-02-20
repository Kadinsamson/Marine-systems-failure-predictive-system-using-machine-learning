import asyncio
import websockets
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load and preprocess your dataset
data_path = r"navalplantmaintenance.csv"
data = pd.read_csv(data_path, sep=r'\s+', header=None)

data.columns = ['lever_position', 'ship_speed', 'gt_shaft', 'gt_rate', 'gg_rate',
                'sp_torque', 'pp_torque', 'hpt_temp', 'gt_c_i_temp', 'gt_c_o_temp',
                'hpt_pressure', 'gt_c_i_pressure', 'gt_c_o_pressure', 'gt_exhaust_pressure',
                'turbine_inj_control', 'fuel_flow', 'gt_c_decay', 'gt_t_decay']

# Print the DataFrame columns to check names
print("DataFrame Columns:", data.columns.tolist())

X = data.iloc[:, :-2]
Y1 = data['gt_c_decay']  # Compressor decay
Y2 = data['gt_t_decay']  # Turbine decay

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train RandomForest Regressors
def train_random_forest(X_train, Y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    return model

rfr_y1 = train_random_forest(X_scaled, Y1)
rfr_y2 = train_random_forest(X_scaled, Y2)

# Define a function to estimate time before failure based on decay prediction
def estimate_time_before_failure(predicted_decay):
    hours = max(0, (1 - predicted_decay) * 100)  # Adjust the factor as necessary
    return f"{int(hours * 60)} minutes"  # Convert hours to minutes

# Threshold conditions dictionary
threshold_conditions = {
    'gt_c_i_temp': (lambda x: x < 280 or x > 400, 280, 400),  # Inlet Temperature
    'gt_c_o_temp': (lambda x: x > 590, 590),                  # Outlet Temperature
    'gt_c_i_pressure': (lambda x: x < 0.90, 0.90),           # Inlet Pressure
    'gt_c_o_pressure': (lambda x: x < 5.00, 5.00),             # Outlet Pressure
    'fuel_flow': (lambda x: x < 0.06 or x > 1.5, 0.06, 1.5),  # Fuel Flow
    'turbine_inj_control': (lambda x: x < 4 or x > 7.5, 4, 7.5), # Turbine Injection Control
    'hpt_temp': (lambda x: x > 650, 650),                    # High-Pressure Turbine Temperature
    'gt_shaft_torque': (lambda x: x < 280 or x > 300, 280, 300), # GT Shaft Torque
    'gg_rate': (lambda x: x < 5000 or x > 7200, 5000, 7200)  # Gas Generator Rate
}

# Define a function to process incoming live data
def detect_fault_and_identify_critical_sensor(live_data):
    feature_columns = ['lever_position', 'ship_speed', 'gt_shaft', 'gt_rate',
                       'gg_rate', 'sp_torque', 'pp_torque', 'hpt_temp',
                       'gt_c_i_temp', 'gt_c_o_temp', 'hpt_pressure',
                       'gt_c_i_pressure', 'gt_c_o_pressure', 'gt_exhaust_pressure',
                       'turbine_inj_control', 'fuel_flow']

    live_data_df = pd.DataFrame([live_data], columns=feature_columns)
    live_data_scaled = scaler.transform(live_data_df)

    # Predict using the trained models
    y1_pred = rfr_y1.predict(live_data_scaled)
    y2_pred = rfr_y2.predict(live_data_scaled)

    result = {
        "Predicted_Compressor_Decay": y1_pred[0],
        "Predicted_Turbine_Decay": y2_pred[0],
        "Time_Before_Failure": {
            "Compressor": estimate_time_before_failure(y1_pred[0]),
            "Turbine": estimate_time_before_failure(y2_pred[0])
        },
        "Compressor_Fault_Detected": "Running smoothly, no fault detected",
        "Turbine_Fault": "Running smoothly, no fault detected",
        "Warnings": [],
        "Suggestions": []
    }

    # Check threshold conditions and generate warnings and suggestions
    for sensor, (condition, *thresholds) in threshold_conditions.items():
        if sensor in live_data_df.columns:  # Ensure the sensor exists in live data
            sensor_value = live_data_df[sensor].values[0]
            if condition(sensor_value):  # Evaluate the lambda function
                result["Warnings"].append(f"Warning: Value detected for '{sensor}': {sensor_value} (Threshold: {', '.join(map(str, thresholds))})")
                result["Compressor_Fault_Detected"] = True

                # Add suggestions based on the sensor condition
                result["Suggestions"].append(f"Maintain '{sensor}' (Threshold: {', '.join(map(str, thresholds))}) as its value is outside the threshold.")

    return result

# WebSocket server handler
async def handle_client(websocket, path):
    print(f"Client connected: {path}")
    try:
        async for message in websocket:
            # Parse incoming JSON data
            live_data = json.loads(message)
            # Perform fault detection and identification
            result = detect_fault_and_identify_critical_sensor(live_data)
            # Send result back to the client
            await websocket.send(json.dumps(result))
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Error: Connection closed unexpectedly. Details: {e}")
    except websockets.exceptions.ConnectionClosedOK:
        print("Client disconnected normally")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

# Start WebSocket server
async def start_server():
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

# Run the server
if __name__ == "__main__":
    asyncio.run(start_server())
