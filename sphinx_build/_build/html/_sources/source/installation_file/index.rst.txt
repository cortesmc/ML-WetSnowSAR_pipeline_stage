.. _installation:

Installation
============
Before creating a Qanat experiment, you need to be sure to follow the `Qanat installation <https://ammarmian.github.io/qanat/installation.html>`_ on your Python environment and have all the different libraries installed or have an Apptainer/Singularity container.

Here is a list of all the libraries needed and an example of a container that can be built with Apptainer: 

Requirements python
-------------------
.. toctree::
   :maxdepth: 2

   ./req_python

Creating an Experiment with Qanat
---------------------------------

1. **Initialize Qanat in the Pipeline Folder**:
  
   Open the terminal in your **pipeline folder** and run the following command:

   .. code-block:: bash

      qanat init .

2. **Initialize the Results Folder**:
  
   Upon running the initialization command, you'll see the following prompt:

   .. code-block:: bash

      [14:21:20] INFO Initializing Qanat repository. init.py:27
      Where do you want to store the results? (./results): ./results

   Specify the desired path for storing the results.

3. **Create a New Experiment**:
  
   - **Creating with Bash**:
    
     Once Qanat is initialized in the pipeline folder, create a new experiment by running:

     .. code-block:: bash

        qanat experiment new
  
     You will be prompted to provide the following information. The critical values include the path and executable. You can freely choose the other variables:

     .. code-block:: bash

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

     .. code-block:: bash

        qanat experiment new -f ./parameter/qanat_experiment.yml

4. **Running the Experiment**:
  
   Once the experiment is created, configure your `parameter_file.yml` with all necessary values and deploy the experiment:

   - **Testing the Groups**:

     .. code-block:: bash

        qanat experiment run <experiment_name> --param_file "./parameter/config_param.yml" --dry_run

   - **Running the Experiment**:

     .. code-block:: bash

        qanat experiment run <experiment_name> --param_file "./parameter/config_param.yml"

Create an actions with Qanat
----------------------------
Before running actions, they must be created.

.. note::
    If the experiment was created using a YAML file, it already includes the actions, which are ready for use. You can verify this by running:

    .. code-block:: bash

        qanat experiment status <name_experiment>

    This should display two actions:
    * results: Retrieves all results from all the groups

To add or create a new action for the experiment, follow these steps:

1. **Update the Experiment**:

   .. code-block:: bash

      qanat experiment update <experiment_name>
  

2. **Select the option for Action and choose 'add'**.

3. **Provide a name and a description**.

4. **Specify the executable for the action (path from project root)**:

   .. code-block:: bash

      ⚙ Executable of the action (path from project root): evaluation/generate_results.py

5. **Set the executable to**: python.

6. **Add the experiment**.