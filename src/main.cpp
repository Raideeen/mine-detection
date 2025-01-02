#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <Arduino_LPS22HB.h>

// Custom service and characteristic UUIDs
BLEService magnetometerService("12345678-1234-5678-1234-56789abcdef0");
BLECharacteristic magnetometerCharacteristic("12345678-1234-5678-1234-56789abcdef1",
                                             BLENotify, 100); // Adjust size for larger payload

float x, y, z, ledvalue;
float pressure, altitude;

void setup()
{
  Serial.begin(9600);

  if (!IMU.begin())
  {
    Serial.println("Failed to initialize IMU!");
    while (1)
      ;
  }

  if (!BLE.begin())
  {
    Serial.println("Starting BLE failed!");
    while (1)
      ;
  }

  if (!BARO.begin())
  {
    Serial.println("Failed to initialize pressure sensor!");
    while (1)
      ;
  }

  BLE.setLocalName("Magnetometer");
  BLE.setAdvertisedService(magnetometerService);
  magnetometerService.addCharacteristic(magnetometerCharacteristic);
  BLE.addService(magnetometerService);
  BLE.advertise();

  Serial.println("BLE Magnetometer is now broadcasting...");
}

void loop()
{
  BLEDevice central = BLE.central();

  if (central)
  {
    Serial.println("Connected to central");

    while (central.connected())
    {
      // Read magnetic field data
      IMU.readMagneticField(x, y, z);

      // Visual indication
      if (x < 0)
      {
        ledvalue = -(x);
      }
      else
      {
        ledvalue = x;
      }

      analogWrite(LED_BUILTIN, ledvalue);
      // Read altitude data from the barometer
      pressure = BARO.readPressure();                              // Pressure in kPa
      altitude = 44330 * (1 - pow(pressure / 101.325, 1 / 5.255)); // Altitude in meters

      // Create data packet
      char data[100];
      snprintf(data, sizeof(data), "%.2f,%.2f,%.2f,%.2f,%.2f", x, y, z, pressure, altitude);

      // Send data packet via BLE
      magnetometerCharacteristic.writeValue((const uint8_t *)data, strlen(data));
      Serial.println(data); // For debugging

      delay(100);
    }

    Serial.println("Disconnected from central");
  }
}
