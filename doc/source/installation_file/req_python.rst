
.. _requirements_python:

Requirements python
===================
 
Here is a list of all the libraries that are needed for this project:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `plotly`
- `seaborn`
- `joblib`
- `imblearn`
- `h5py`
- `pyyaml`
- `tqdm`
- `htcondor`

Using Apptainer/Singularity
===========================

To streamline the setup process and ensure that all dependencies are correctly installed and managed, it is recommended to use an Apptainer/Singularity container. This approach not only simplifies the environment setup but also ensures consistency across different systems.

Below is a `.def` file, which can be compiled to create a fully functioning container specifically tailored for this project:

.. code-block:: bash

    Bootstrap: docker
    From: python:3.10

    %post
        apt-get update && apt-get install -y software-properties-common
        apt-get update

        # Install Python packages
        pip install --upgrade pip
        pip install numpy==1.26.4
        pip install pandas scikit-learn scipy
        pip install matplotlib plotly seaborn
        pip install joblib
        pip install imblearn
        pip install h5py 
        pip install pyyaml
        pip install tqdm
        pip install htcondor

    %runscript
        exec "$@"

    %help
        This container is a base container for pansharpening experiments.
        It includes all necessary libraries and dependencies to ensure
        smooth operation of the project’s Python environment.

How to Use the Container
========================
After compiling the `.def` file with Apptainer/Singularity, you can run the container using the following command:

.. code-block:: bash

    apptainer build <container_name>.sif python <your_script.def>

This command ensures that your script runs within the containerized environment, leveraging all the pre-installed libraries and dependencies.