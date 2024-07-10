# Geo-CETRA



## Setup

Dependencies:

- PyTorch 1.6+ (https://pytorch.org/)
- requirements.txt






## Preprocess

get the trajectory samples and change the input path, specify the study area and run file to preprocess the trajectory and generate realistic constraints

```
python proprocess.py
```



## Training

after preprocess, should get the trajectories by day. The preprocess is also printing all necessary information. Run the app.py or change the parameter in the script to run the pipeline  


```
bash scripts/run_generator_time.sh
```


## code is continously updating 


