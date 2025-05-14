import os
import pickle
import click
import mlflow
import numpy as np

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "homework"
MODEL_NAME = "nyc-taxi-regressor"  # <-- This is your registry model name
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)



def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run() as run:
        new_params = {param: int(params[param]) for param in RF_PARAMS}
        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        val_pred = rf.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mlflow.log_metric("val_rmse", val_rmse)

        test_pred = rf.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        mlflow.log_metric("test_rmse", test_rmse)

        # Log model explicitly for registration
        mlflow.sklearn.log_model(rf, artifact_path="model")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # Get top N runs from hyperopt experiment
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    hpo_runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    for run in hpo_runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Now search best model from best-models experiment
    best_model_experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=best_model_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )

    if not runs:
        print(f"❌ No runs found in experiment '{EXPERIMENT_NAME}'")
        return

    best_run = runs[0]

    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from run_id={run_id} with test_rmse={best_run.data.metrics['test_rmse']}")

    # Register model

    mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)


if __name__ == '__main__':
    run_register_model()
