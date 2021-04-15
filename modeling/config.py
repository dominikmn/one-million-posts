import logging
import parsenvy

logger = logging.getLogger(__name__)

# mlflow
# sometimes when saving links in text.. there is a new line .. strip removes that
try:
    TRACKING_URI = open(".mlflow_uri").read().strip()
except:
    TRACKING_URI = parsenvy.str("MLFLOW_URI")
#TRACKING_URI_DEV = "http://127.0.0.1:5000/"

EXPERIMENT_NAME = "nlp-trio"
