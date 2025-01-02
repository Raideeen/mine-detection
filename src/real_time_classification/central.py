import asyncio
from pathlib import Path

import joblib
import pandas as pd
from bleak import BleakClient, BleakScanner

DEVICE_NAME = "Magnetometer"
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHARACTERISTIC_UUID = "12345678-1234-5678-1234-56789abcdef1"

# Load the trained model and scaler
model_dir = Path(__file__).resolve().parent.parent.parent / "model"
model = joblib.load(model_dir / "model.joblib")
scaler = joblib.load(model_dir / "scaler.joblib")

# Expected feature columns (used during training)
FEATURE_COLUMNS = ["Mag_X", "Mag_Y", "Mag_Z"]


async def main():
    devices = await BleakScanner.discover()
    for device in devices:
        if device.name == DEVICE_NAME:
            async with BleakClient(device.address) as client:
                print(f"Connected to {DEVICE_NAME}")

                def notification_handler(sender, data):
                    try:
                        # Decode and parse the data
                        decoded_data = data.decode().split(",")
                        mag_x, mag_y, mag_z, _, _ = map(float, decoded_data)

                        # Create a DataFrame to retain feature names
                        features = pd.DataFrame(
                            [[mag_x, mag_y, mag_z]],
                            columns=FEATURE_COLUMNS,
                        )
                        features_scaled = scaler.transform(features)

                        # Predict with the model
                        prediction = model.predict(features_scaled)[0]
                        confidence = model.predict_proba(features_scaled)[0][prediction]

                        # Display result
                        result = "metallic" if prediction == 1 else "not metallic"
                        print(f"Detected: {result} (Confidence: {confidence:.2f})")

                    except Exception as e:
                        print(f"Error processing data: {e}")

                # Start receiving data
                await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
                await asyncio.sleep(60)  # Receive notifications for 60 seconds
                await client.stop_notify(CHARACTERISTIC_UUID)


asyncio.run(main())
