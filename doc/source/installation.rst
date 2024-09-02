.. _installation:

Installation
============

This guide explains how to install Qanat and deploy an experiment using it.

Prerequisites
-------------
Before you can deploy an experiment, you need to have Qanat installed. Follow the installation instructions here: `Qanat Installation <https://ammarmian.github.io/qanat/installation.html>`__.

Creating an Experiment with Qanat
---------------------------------
1. **Initialize Qanat in the Pipeline Folder**:
  
   Open the terminal in your **pipeline folder** and run the following command:
   ::
      qanat init .

2. **Initialize the Results Folder**:
  
   Upon running the initialization command, you'll see the following prompt:
   ::
      [14:21:20] INFO Initializing Qanat repository. init.py:27
      Where do you want to store the results? (./results): ./results

   Specify the desired path for storing the results.

3. **Create a New Experiment**:
  
   - **Creating with Bash**:
    
     Once Qanat is initialized in the pipeline folder, create a new experiment by running:
     ::
        qanat experiment new
  
     You will be prompted to provide the following information. The critical values include the path and executable. You can freely choose the other variables:
     ::
        Name: <name_experiment>
        Description: <description_experiment>
        Path: evaluation
        Executable: evaluation/learning_models.py
        Execute command of the experiment (/usr/bin/bash): python
        Tags: ['ML', 'benchmarking']
        Datasets: []
        Do you want to add this experiment? [y/n]: y

   - **Creating from YAML**:
    
     You can also create an experiment directly with the correct paths using the following command:
     ::
        qanat experiment new -f ./parameter/qanat_experiment.yml

4. **Running the Experiment**:
  
   Once the experiment is created, configure your `parameter_file.yml` with all necessary values and deploy the experiment:

   - **Testing the Groups**:
     ::
        qanat experiment run <experiment_name> --param_file "./parameter/config_param.yml" --dry_run

   - **Running the Experiment**:
     ::
        qanat experiment run <experiment_name> --param_file "./parameter/config_param.yml"

Running Actions with Qanat
--------------------------
Before running actions, they must be created.

.. note::
    If the experiment was created using a YAML file, it already includes the actions, which are ready for use. You can verify this by running:
    ::
        qanat experiment status <name_experiment>

    This should display two actions:
    * results: Retrieves all results from all the groups

To add or create a new action for the experiment, follow these steps:

1. **Update the Experiment**:

  ::
      qanat experiment update <experiment_name>
  

2. **Select the option for Action and choose 'add'**.

3. **Provide a name and a description**.

4. **Specify the executable for the action (path from project root)**:

  ::
      ⚙ Executable of the action (path from project root): evaluation/generate_results.py

5. **Set the executable to**: python.

6. **Add the experiment**.

To use the actions, you need to have run the experiment at least once. Qanat generates new files named `run_N°`, where N° is the index of the run. To apply an action to a specific run, use the run index as follows:

- For the action `get_results`, which consolidates all the results from multiple groups into a single file named `final_results`:
  ::
      qanat experiment action pipeline <action_name> n_run

- For the action `generate_maps`, which generates estimated maps for comparison with the data from Grand-Rousses or another assigned massif:
  ::
      qanat experiment action pipeline generate_maps n_run --config_file <path_to_config_file>
