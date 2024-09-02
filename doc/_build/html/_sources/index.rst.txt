.. wet_snow_detection documentation master file, created by
   sphinx-quickstart on Wed Jun 12 16:32:48 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Wet Snow Detection Documentation!
===============================================
.. image:: ./source/images/listic_logo.png
   :scale: 100%
   :align: center

Project References
------------------
- **Base Project**: `ML-WetSnowSAR <https://github.com/Matthieu-Gallet/ML-WetSnowSAR>`__
- **Thesis Project by**: Matthieu Gallet
- **Developped by**: Carlos Cortes and Matthieu Gallet
- **Enseignant chargé du suivi du stagiaire** : Trouve Emmanuel

Introduction
------------
This framework is designed to facilitate the creation of pipelines for benchmarking various machine learning models in the detection of wet snow. It leverages different technologies for data management, job processing, and experiment management.

Key Features
------------
- **Experiment Management**: Utilizes `Qanat <https://ammarmian.github.io/qanat/>`_ for managing experiments locally or on dedicated servers.
- **Benchmarking**: Deploy multiple experiments with various ML models to obtain results on the same dataset, allowing easy benchmarking and comparison.
- **Customization**: Offers multiple options for selecting balancing methods, labeling methods, and more. Users can create diverse tests, comparing models with different data, seeds, and data selection methods.

Getting Started
---------------
Check out the installation section for further information, including how to :ref:`installation` the project.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   ./source/usage_files/index
   ./source/installation_file/index.rst
   estimators
   evaluation
   utils.

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
