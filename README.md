
# XTraffic: A Dataset Where Traffic Meets Incidents with Explainability and More

## Authors
Xiaochuan Gou\*, Ziyue Li\*, Tian Lan, Junpeng Lin, Zhishuai Li, Bingyu Zhao, Chen Zhang, Di Wang, Xiangliang Zhang

*The authors contributed equally to this work.


## Introduction
Welcome to the XTraffic dataset repository! This dataset integrates traffic and incident data across a large-scale region, covering 16,972 traffic nodes over the entire year of 2023. XTraffic includes time-series data on traffic flow, lane occupancy, and average vehicle speed, along with spatiotemporally-aligned incident records across seven different classes. Each node also features detailed physical and policy-level meta-attributes of lanes. Our goal is to enhance the interpretability and practical applications of traffic management and safety analysis through this comprehensive dataset.

## How to Use
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/XTraffic-dataset.git
    cd XTraffic-dataset
    ```

2. **Install Dependencies:**
    Ensure you have the necessary dependencies installed. You can use the `requirements.txt` file to set up your environment.
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the Dataset:**
    The dataset is available for download at the [provided URLs](https://www.kaggle.com/datasets/gpxlcj/xtraffic/). Follow the links to download the required files.

4. **Load the Data:**
    <!-- Use the provided scripts to load and preprocess the data.
    ```python
    import numpy as np

    # Example to load traffic data for January
    january_data = np.load('data/January_traffic.npy')
    
    # Example to load incident data
    incident_data = np.load('data/incident_data.npy')
    ``` -->

5. **Run Examples:**
    Example scripts are provided to demonstrate how to use the dataset for various tasks like incident classification, and causal analysis.
    ***For incident classification***:
    ```bash
    python examples/traffic_forecasting.py
    python examples/incident_classification.py
    ```

    ***For global causal analysis***:



    ***For local causal analysis***:


## Meta Data
The meta data for the sensors and incidents can be accessed through the following URL:
[Meta Data URL](https://github.com/XAITraffic/XTraffic/blob/main/xtraffic-metadata.json)

Feel free to explore and contribute to the repository. If you have any questions or suggestions, please open an issue or contact the authors.
