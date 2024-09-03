Experiment Configuration
=========================

This section explains the structure and purpose of the YAML configuration used to create an experiment pipeline.

Configuration Overview
-----------------------

Here is the YAML configuration used for the creation of the experiment:

.. code-block:: yaml

    name: pipeline
    description: Pipeline pour la validation et Benchmarking de Modèles de Machine Learning pour la Caractérisation de la Neige Humide par Imagerie.
    path: evaluation
    executable: evaluation/learning_models.py
    executable_command: python
    actions:
      - results:
          name: results
          executable: evaluation/generate_results.py
          executable_command: python
          description: Retrieves all the results and regenerate all the images and results.

Explanation of Fields
----------------------

- **name**: ``pipeline``  
  The name of the pipeline or experiment. This should be a unique identifier for the experiment and it is used to run the experiment.

- **description**: ``Pipeline pour la validation et Benchmarking de Modèles de Machine Learning pour la Caractérisation de la Neige Humide par Imagerie.``  
  A brief description of the pipeline, providing context about what this experiment is meant to achieve.

- **path**: ``evaluation``  
  Specifies the path where the core evaluation script is located. This is typically the directory or folder where the main executable is found.

- **executable**: ``evaluation/learning_models.py``  
  The main Python script that runs the experiment. This script usually contains the logic for training, validating, and testing machine learning models.

- **executable_command**: ``python``  
  The command used to execute the script specified in the ``executable`` field. In this case, the Python interpreter is used to run the script.

Actions
-------

The configuration allows for defining additional actions that should be executed as part of the experiment. Below is the detailed breakdown of the actions:

- **results**:
  
  - **name**: ``results``  
    The name of the action, which in this case, refers to the process of generating results. This name is also used to deploy the action qith qanat.

  - **executable**: ``evaluation/generate_results.py``  
    The script responsible for retrieving and regenerating all the results and images related to the experiment.

  - **executable_command**: ``python``  
    Similar to the main executable, this action also uses Python to run the script.

  - **description**: ``Retrieves all the results and regenerate all the images and results.``  
    A description of what this action does. This action is specifically used for post-processing, where it gathers all the results produced by the experiment and regenerates any necessary outputs, such as images or other result files.

Usage Notes
-----------

- Make sure that all paths specified are correct and point to the appropriate directories and files.
- Ensure that Python is correctly installed and accessible in the environment where this experiment will be executed.
- The actions can be extended, modified or added depending on additional steps or processes required in the pipeline.
