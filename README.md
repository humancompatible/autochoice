# humancompatible.autochoice
Auto-ML tuning for fairness toolkits

This toolkit provides an automated framework for testing, evaluating and exploring fairness in machine learning algorithms using different fairness metrics and algorithms. 
It serves as an automated solution that integrates a seamless benchmarking for evaluating the performance of fairness for various models under diverse fairness constraints.   

Features:

- Automated tracking of model parameters, without requiring a by-hand definition
- Identification of best model parameters to achieve fairness
- Integrates conformal prediction ranges (to identify the statistical significance of the fairness output)
- Support multiple fairness constraints and algorithms from AIF360 and AIX360


## Requirements:

The toolkit provides a Dockerfile that builds the image containing the necessary execution environment for the evaluation process as well as exposes a Web-UI to simplify the execution of a machine learning pipeline.

## Setting up MLFlow Tracking Server
Under the folder:

`autochoice/autochoicebackend`

edit the files:
- run_mlflow.py: Setup the IP and PORT variables regarding the MLFlow tracking server in lines 28 & 29
- run_experiments_parallel: Setup the IP and PORT variables regarding the MLFlow tracking server in lines 12 & 13


## Building the docker image:

`docker build -t mlflow-automl-experiment .`

This command will build the execution environment of the ML pipeline and will automate the tracking process.

## Running the web-UI

Under the folder:

`autochoice/autochoiceui`

run the following command:

`voila autofairui.ipynb --no-browser --port=8888 --Voila.ip=0.0.0.0`

