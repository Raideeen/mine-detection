import asyncio
import math
from bleak import BleakClient, BleakScanner
from datetime import datetime
import csv

DEVICE_NAME = "Magnetometer"
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHARACTERISTIC_UUID = "12345678-1234-5678-1234-56789abcdef1"

async def main():
    devices = await BleakScanner.discover()
    for device in devices:
        if device.name == DEVICE_NAME:
            async with BleakClient(device.address) as client:
                print(f"Connected to {DEVICE_NAME}")

                # Open file in append mode
                with open("dataset.csv", "a", newline="") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    
                    # Write header row only if the file is empty
                    if csv_file.tell() == 0:
                        csv_writer.writerow(["Timestamp", "Mag_X", "Mag_Y", "Mag_Z", "Pressure", "Altitude", "Magnitude", "Label"])

                    def notification_handler(sender, data):
                        try:
                            # Decode and parse the data
                            decoded_data = data.decode().split(",")
                            mag_x, mag_y, mag_z, pressure, altitude = map(float, decoded_data)
                            
                            # Calculate magnitude
                            magnitude = math.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
                            print(magnitude)
                            
                            # Annotate based on simple heuristic (e.g., magnitude threshold)
                            label = "0" if magnitude < 100 else "1"  # Adjust threshold based on observation
                            
                            # Add timestamp and save to CSV
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"Received data: {timestamp}, {decoded_data}, Label: {label}")
                            csv_writer.writerow([timestamp, mag_x, mag_y, mag_z, pressure, altitude, magnitude, label])
                        except Exception as e:
                            print(f"Error processing data: {e}")

                    # Start receiving data
                    await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
                    await asyncio.sleep(60)  # Receive notifications for 30 seconds
                    await client.stop_notify(CHARACTERISTIC_UUID)

asyncio.run(main())
