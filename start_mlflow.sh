#!/bin/bash
conda activate exp-tracking-env
cd 03-training/experiment_tracking/

# Start MLflow UI in background
#mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
#MLFLOW_PID=$!



# Run Jupyter notebook non-interactively
#jupyter nbconvert --to notebook --execute pipeline.ipynb --output pipeline1.ipynb

# Kill MLflow after notebook execution (optional)
#kill $MLFLOW_PID

# If you want to pass parameter 
:'


#!/bin/bash

input_file = "default.csv"
alpha = 0.01
conda activate exp-tracking-env
cd 03-training/experiment_tracking/

# Start MLflow UI in background
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
MLFLOW_PID=$!

# Run notebook with papermill and pass parameters
papermill your_notebook.ipynb output_notebook.ipynb \
    -p input_file "green_tripdata_2021-02.parquet" \
    -p alpha 0.1

# Kill MLflow after notebook execution (optional)
kill $MLFLOW_PID
'