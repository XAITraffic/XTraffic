# Data Process

In this section, we introduce two preprocess of data: traffic and incident matching ,and adjacency matrix generation 
---

## Traffic and Incident Matching

1. Download the data into the ***data*** directory.

***sensor_meta_feature.csv***, and the incident records of the coresponding year, like  ***incident_y2023.csv***

2. Run the script file
```
python traffic_incident_match.py --dis_threshold 0.5 
```

***dis_threshold*** is the threshold of the maximum distance between the closest sensor to the incident. If it's out of the threshold, we filter the sample.


## Adjacency Matrix Generation
1. Download the data into the ***data*** directory.

***sensor_meta_feature.csv***

2. Run the script file

```
python adj_generation.py --eps
```

***eps*** is the threshold to determine the first-order neighbor relations between two sensors. If the distance between the two sensors is larger than it, then the entry of corresponding position in the adjacency matrix is 0.