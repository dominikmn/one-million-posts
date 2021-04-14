# Setup

Requirements:

- pyenv with Python: 3.8.5

## Development Setup

Use the requirements file in this repo to create a new environment.

```BASH
make setup

#or

pyenv local 3.8.5
python -m venv .venv
pip install --upgrade pip
pip install -r requirements.txt
```

The MLFLOW URI is not stored on git. There are two options to set it. Either locally in the .mlflow_uri file:

```BASH
echo http://127.0.0.1:5000/ > .mlflow_uri
```

this will create a local file where the uri is stored. Alternatively, one can export it as an environment variable with

```bash
export MLFLOW_URI=http://127.0.0.1:5000/
```

The code in the [config.py](modeling/config.py) will try to read the uri locally and if the file doesn't exist will look in the env var.. IF that is not set the URI will be empty in the code.

### Usage of MLFlow

#### Creating an MLFlow experiment

Experiments can be created via the GUI or via [command line](https://www.mlflow.org/docs/latest/tracking.html#managing-experiments-and-runs-with-the-tracking-service-api) if one uses the local mlflow:

```bash
mlflow experiments create --experiment-name <name-of-experiment>
```

Check the local mlflow by running

```bash
mlflow ui
```

and opening the link [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Kill the gunicorn process

If the port is in use (and the local mlflow cannot be run with `mlflow ui`) the process can be killed with

```bash
ps -A | grep gunicorn
kill <PID>
```
