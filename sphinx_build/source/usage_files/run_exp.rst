Running an Experiment
=====================
First you need the framework to be totally installed, for that you can see on :ref:`installation`.
Once the framework and Qanat are initialized, everything is ready to start working.

1. **Setting the Parameters**: 

   The first step is to set the parameters using a specific template (`--param_file`) that Qanat uses to manage different groups. The template is divided into two main parts:

   * **fixed_args**: These are the arguments that remain consistent across all groups.
   * **varying_args**: These are the variables that change for each group.
   
   .. note::
      `fixed_args` are applied uniformly across all groups, whereas `varying_args` allow you to introduce variability within each group. This approach enables Qanat to create and run multiple experiments efficiently.
      
      For examples of templates, please refer to the :ref:`templates`.

   .. warning::
      Currently, the pipeline supports the creation of groups with different *seeds* and *ML models* only. 
      When creating different groups, be mindful that the results may not be directly comparable across these groups.

2. **Deploying the Experiment**: 

   Once your Qanat experiment is created and your `parameter_file.yml` is configured with all necessary values, you can deploy the experiment as follows:

   - **Testing the Groups**:
     ::
        qanat experiment run <experiment_name> --param_file "./parameter/config_param.yml" --dry_run

   - **Running the Experiment**:
     ::
        qanat experiment run <experiment_name> --param_file "./parameter/config_param.yml"

Deploying an Experiment Inside a Container
------------------------------------------

To deploy the experiment inside a container, use the `--container` option:
::
    qanat experiment run <experiment_name> [POSITIONAL ARGUMENTS] [OPTIONS] --container <container_path.sif> [--gpu True|False]

.. note::
   For more details, refer to the Qanat documentation: `Running inside a container <https://ammarmian.github.io/qanat/usage/running/container.html>`__.

Deploying an Experiment Through a Job System
--------------------------------------------

To deploy the experiment through a job system, use the `--runner` option:
::
    qanat experiment run <experiment_name> --runner local|htcondor|slurm [POSITIONAL ARGUMENTS] [OPTIONS] [--submit_template yourtemplate]

.. note::
   For more details, refer to the Qanat documentation: `Running through a job system <https://ammarmian.github.io/qanat/usage/running/runner.html>`__.

Running an action
=================


To use the actions, you need to have run the experiment at least once. Qanat generates new files named run_N°, where N° is the index of the run. To apply an action to a specific run, use the run index as follows:
For the action get_results, which consolidates all the results from multiple groups into a single file named final_results:
  .. code-block:: bash

      qanat experiment action pipeline <action_name> n_run

The get_results action will automatically regroup all the results, taking into consideration whether there is a single group or multiple groups. It then generates plots and creates a final results file where it averages all the results from models with the same name.