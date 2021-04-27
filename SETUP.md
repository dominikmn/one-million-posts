# Setup

## Setup - MLflow

We use MLflow to track our models. Therefore, it needs to be set up to run scripts in `./modeling`:

1. The MLFLOW URI has to be added manually (not stored on git).
    * Either set it locally in the .mlflow_uri file (which has to be done only once and will create a local file where the uri is stored):
    ```BASH
    echo http://127.0.0.1:5000/ > .mlflow_uri
    ```

    * or export it as an environment variable (which has to be repeated on restart of your machine):

    ```bash
    export MLFLOW_URI=http://127.0.0.1:5000/
    ```
    
    * The code in the [config.py](modeling/config.py) will try to read the uri locally and if the file doesn't exist will look in the env var. If that is not set the URI will be empty in the code.

2. Create an MLFlow experiment with the name that is set in config.py. **This has to be done only once.** We use the experiment name `nlp-trio` in the config.py. You can either use the GUI to create an experiment [MLflow Documentation](https://www.mlflow.org/docs/latest/tracking.html#managing-experiments-and-runs-with-the-tracking-service-api) or create a local experiment using the CLI:
  ```bash
  mlflow experiments create --experiment-name <name-of-experiment>
  ```

3. **Always** start the mlflow server in a separate terminal session, before executin a modeling script:

  ```bash
  mlflow server
  ```
  
  The UI can then be accessed with [http://127.0.0.1:5000](http://127.0.0.1:5000).
