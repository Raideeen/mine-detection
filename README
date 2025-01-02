# Landmine detection

This repository contains a project for real-time classification of magnetic field data using an Arduino Nano 33 BLE Sense. The project includes data collection, training a machine learning model, and real-time classification using Bluetooth Low Energy (BLE).

## Project Workflow

<p align="center">
    <img src="assets/workflow.png" />
</p>

## Components

### Arduino Code

The Arduino code is located in [main.cpp](/src/main.cpp). It reads magnetic field data from the `LSM9DS1` sensor and pressure data from the `LPS22HB` sensor, and sends the data via BLE. 

### Data Collection

The data collection script is located in [src/record/central.py](/src/record/central.py). It connects to the Arduino Nano 33 BLE Sense via BLE and records magnetic field data into a CSV file.

### Training

The training pipeline is implemented in [pipeline.py](/src/training/pipeline.py). 

It includes the following steps:
1. Load and clean data
2. Augment data
3. Split data into training and testing sets
4. Train the model
5. Evaluate the model
6. Save the model and scaler

The final model generally gives an accuracy of around 98%. One reason for this could be the experimental setup and the lack of a more convincing landmine that could indeed create a more interesting simulated setup. It is overfitting a bit but it works well.

```text
Classification Report:
             precision    recall  f1-score   support

           0       1.00      0.99      1.00      2571
           1       0.96      0.99      0.97       327

    accuracy                           0.99      2898
   macro avg       0.98      0.99      0.99      2898
weighted avg       0.99      0.99      0.99      2898
```

### Real-Time Classification

The real-time classification script is located in [central.py](/src/real_time_classification/central.py). It connects to the Arduino Nano 33 BLE Sense via BLE, receives magnetic field data, and classifies it using the trained SVM model.

## Setup

### Prerequisites

- [PlatformIO](https://platformio.org/)
- [Conda](https://anaconda.org/anaconda/conda)
- [Python 3.11+](https://www.python.org/)
- [Joblib](https://joblib.readthedocs.io/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Bleak](https://bleak.readthedocs.io/)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Raideeen/mine-detection.git
    cd mine-detection
    ```

3. Install [Conda](https://anaconda.org/anaconda/conda) 
4. Install Python dependencies:
    ```sh
    conda env create -f environment.yml
    ```
5. Activate the conda environment:
    ```sh
    conda activate landmine_detection
    ```

### Building and Uploading Arduino Code

1. Connect your Arduino Nano 33 BLE Sense to your computer.
2. Activate the conda environment:
    ```sh
    conda activate landmine_detection
    ```
3. Build and upload the code:
    ```sh
    pio run --target upload
    ```

### Running Data Collection

1. Run the data collection script:
    ```sh
    python src/record/central.py
    ```

### Training the Model

1. Run the training pipeline:
    ```sh
    python src/training/pipeline.py
    ```

### Running Real-Time Classification

1. Run the real-time classification script:
    ```sh
    python src/real_time_classification/central.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details. Do whatever you want with it ! :D

## Acknowledgements

- [PlatformIO](https://platformio.org/)
- [Arduino](https://www.arduino.cc/)
- [Scikit-learn](https://scikit-learn.org/)
- [Bleak](https://bleak.readthedocs.io/)

## Contact

For any questions or suggestions, please contact [adrien.djebar@proton.me](mailto:adrien.djebar@proton.me).