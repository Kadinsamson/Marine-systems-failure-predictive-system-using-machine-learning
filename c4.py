import asyncio
import websockets
import json
import pandas as pd
import time

# Load the dataset
data_path = 'fault_condition_data.csv'  # Path to your dataset
data = pd.read_csv(data_path, sep='\s+', header=None)  # Adjust separator if needed

# Define columns (as per your dataset structure)
data.columns = ['lever_position', 'ship_speed', 'gt_shaft', 'gt_rate', 'gg_rate',
                'sp_torque', 'pp_torque', 'hpt_temp', 'gt_c_i_temp', 'gt_c_o_temp',
                'hpt_pressure', 'gt_c_i_pressure', 'gt_c_o_pressure', 'gt_exhaust_pressure',
                'turbine_inj_control', 'fuel_flow']  # Ensure these columns match

async def send_data():
    uri = "ws://localhost:8765"  # Change if your server is hosted elsewhere
    async with websockets.connect(uri) as websocket:
        for index, row in data.iterrows():
            # Convert the row to a dictionary
            live_data = row.to_dict()

            # Send live data to the server
            await websocket.send(json.dumps(live_data))
            

            # Wait for a response from the server
            try:
                response = await websocket.recv()
                result = json.loads(response)
                print(f"Received result: {result}")
            except json.JSONDecodeError as e:
                print(f"Error receiving response: {e}")

            # Wait before sending the next data point
            await asyncio.sleep(1)  # Adjust the delay as needed

# Run the client
if __name__ == "__main__":
    asyncio.run(send_data())
