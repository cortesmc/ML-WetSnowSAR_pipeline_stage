## Projet de création d'un pipeline

Projet basé sur le projet : [ML-WetSnowSAR](https://github.com/Matthieu-Gallet/ML-WetSnowSAR)      
Projet de thésard Matthieu Gallet.

Notion du projet : [Stage LISTIC 2024](https://www.notion.so/Stage-LISTIC-2024-58c77ade8f224b1688b7884a6151fe54?pvs=4)

## how to set an experiment with Qanat

1. We need qanat installed:
to install: [Qanat installation](https://ammarmian.github.io/qanat/installation.html)     

2. On the folder pipeline, open the terminal. And init qanat: 
```
qanat init .
```

3. we need to initialize the results folder: 
```
[14:21:20] INFO     Initializing Qanat repertory.                     init.py:27
Where do you want to store the results? (./results): ./results
```

4. Once qanat has been initalized on the pipeline folder, we can create a new experiment:
```
qanat experiment new
```
And using the folowing informations, the importants values to use are the same path and executable feel free to chose the rest of variables: 
```
Name: pipeline
Description: pipeline
Path: evaluation
Executable: evaluation/learning_models.py
Execute command of the experiment (/usr/bin/bash): python
Tags: ['ML', 'benchmarking']
Datasets: []
Do you want to add this experiment? [y/n]: y
```

5.Once the experiment is created we can set our parameter_file.yml with all the values and deploy the experiment:
```
qanat experiment run pipeline --parameters_file "./parameter/config_pipeline.yml"
```
