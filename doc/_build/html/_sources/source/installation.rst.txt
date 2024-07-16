.. _installation:

Installation
============

This guide explains how to use and deploy an experiment using Qanat.

Pre-requirements
----------------
To be able to deploy the experiment, you need to have Qanat installed. 
Follow the installation instructions here: `Qanat Installation <https://ammarmian.github.io/qanat/installation.html>`__.

How to Deploy an Experiment with Qanat
--------------------------------------
1. **Initialize Qanat in the Pipeline Folder**:
  
  Open the terminal in the **pipeline folder** and run:
  ::
      qanat init .

2. **Initialize the Results Folder**:
  
  You will see the following prompt:
  ::
      [14:21:20] INFO Initializing Qanat repertory. init.py:27
      Where do you want to store the results? (./results): ./results

3. **Create a New Experiment**:
  
  - **Create with Bash**:
    
    Once Qanat is initialized in the pipeline folder, create a new experiment:
    ::
        qanat experiment new
  
    Use the following information. The important values are the same path and executable. Feel free to choose the rest of the variables:
    ::
        Name: <name_experiment>
        Description: <description_experiment>
        Path: evaluation
        Executable: evaluation/learning_models.py
        Execute command of the experiment (/usr/bin/bash): python
        Tags: ['ML', 'benchmarking']
        Datasets: []
        Do you want to add this experiment? [y/n]: y

  - **Create from YAML**:
    
    Using the following bash command, an experiment will be created directly with the correct paths:
    ::
        qanat experiment new -f ./parameter/qanat_experiment.yml

4. **Run the Experiment**:
  
  Once the experiment is created, set your `parameter_file.yml` with all the values and deploy the experiment:

  - For testing the groups:
    ::
        qanat experiment run <experiment_name> --param_file "./parameter/config_param.yml" --dry_run

  - For running the experiment:
    ::
        qanat experiment run <experiment_name> --param_file "./parameter/config_param.yml"

Deploy an Experiment Inside a Container
---------------------------------------
This deployment is managed by Qanat and it is done by using `--container` when running:
::
    qanat experiment run <experiment_name> [POSITIONAL ARGUMENTS] [OPTIONS] --container <container_path.sif> [--gpu True|False]

.. note::
  For more information, look into Qanat documentation `Running inside a container <https://ammarmian.github.io/qanat/usage/running/container.html>`__.

Deploy an Experiment Through a Job System
-----------------------------------------
This deployment is managed by Qanat and it is done by using `--runner` when running:
::
    qanat experiment run <experiment_name> --runner local|htcondor|slurm [POSITIONAL ARGUMENTS] [OPTIONS] [--submit_template yourtemplate]

.. note::
  For more information, look into Qanat documentation `Running through a job system <https://ammarmian.github.io/qanat/usage/running/runner.html>`__.

How to Run Actions with Qanat
-----------------------------
First, the actions need to be created.

.. note::
    If the experiment was created from the YAML file, it already has the actions installed and ready to be used.
    You can test this by using:
    ::
      qanat experiment status <name_experiment>

    Where you should be able to find two actions:
    * results: Get all results from all the groups
    * generate_maps: Generates maps with the trained models for comparison with the massif of Grand-Rousses

To add/create a new action to the experiment you need to:

1. **Update the experiment**:
  ::
      qanat experiment update <experiment_name>
2. **Select the option: Action and select add**.
3. **Set a name and a description**.
4. **Set the executable of the action (path from project root)**:
  ::
      ⚙ Executable of the action (path from project root): evaluation/generate_results.py
5. **Set the executable to**: python.
6. **Add the experiment**.

To use the actions, you need to have run the experiment at least once. Qanat generates new files with the names `run_N°`, where the N° is the index of the run. To use an action on a determined run, you need the index of the run, and you use it like this:

- In the case of the action `get_results`, which lets you regroup all the results of multiple groups in one file named `final_results`:
  ::
    qanat experiment action pipeline <action_name> n_run

- In the case of the action `generate_maps`, it generates estimated maps for comparison with the data of Grand-Rousses or any other massif that is assigned:
  ::
    qanat experiment action pipeline generate_maps n_run --config_file <path_to_config_file>
