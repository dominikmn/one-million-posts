# Setup

## Notebook files
The Jupyter notebooks are pushed as `.py` files in the _python percentage script_ format (we like meaningful diffs).  
To get the actual notebook experience open them via jupyter with the [jupytext](https://github.com/mwouts/jupytext) plugin (gets installed as part of `make setup`).

## MLflow

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

3. **Always** start the mlflow server in a separate terminal session before executing a modeling script:

  ```bash
  mlflow server
  ```
  
4. Access the UI via [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Dashboard
Starting the backend requires that a local model file is available.

Initial setup
1. Run `make dashboard`
2. Run `make backend`

Start backend
1. Open a dedicated terminal session.
2. Load the environment `source .venv_backend/bin/activate`
3. Start the backend via `uvicorn prediction_server:app --reload`

Start dashboard
1. Open a dedicated terminal session
2. Load the environment `source .venv_frontend/bin/activate` 
3. Start the dashboard via `python app.py` and copy the address+port that is displayed.
4. Open it in your browser like `http://127.0.0.1:8050/` (replace 8050 with the actual port displayed in step 3. above).

