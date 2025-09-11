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
Simply edit the 

`config.yaml`

file and set the tracking_uri variable in the form of "http://IP:PORT"

e.g.
`tracking_uri: "http://192.168.1.151:5000"`


## Building the docker image:

`docker build -t mlflow-automl-experiment .`

This command will build the execution environment of the ML pipeline and will automate the tracking process.

## Building the web-UI

This step requires the installation of the following packages using the pip tool:

`pip3 install ipywidgets pandas matplotlib seaborn scikit-learn aif360[all]`

If the installation of "aif360[all]" fails due to setuptools, follow the next steps:


```bash 
pip3 install --upgrade pip setuptools wheel
sudo apt install r-base
pip3 install rpy2
pip3 install aif360[all]
```

## Running the web-UI

Under the folder:

`autochoice/autochoiceui`

run the following command:

`voila autofairui.ipynb --no-browser --port=8888 --Voila.ip=0.0.0.0`


## References

If you use the Autochoice toolkit in your work, please cite the following work:

```bibtex
TBC
```
