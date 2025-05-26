import sys
import subprocess
import time
import mlflow
import mlflow.tracking
import pandas as pd
from joblib import Parallel, delayed

# Set MLflow tracking URI

# IP and PORT of MLFlow Tracking Server
IP = ""
PORT = "5000"

# MLflow Setup
MLFLOW_TRACKING_URI = f"http://{IP}:{PORT}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Get the search algorithm from CLI argument (tpe or random)
search_algo = sys.argv[1] if len(sys.argv) > 1 else "tpe"

# List of Docker commands
docker_commands = [
    f"docker run --rm --network=host mlflow-automl-experiment Reweighing EqOddsPostprocessing {search_algo}",
    f"docker run --rm --network=host mlflow-automl-experiment Reweighing CalibratedEqOddsPostprocessing {search_algo}",
    f"docker run --rm --network=host mlflow-automl-experiment Reweighing RejectOptionClassification {search_algo}"
]



# Function to stop only **experiment-related** Docker containers
def stop_experiment_containers():
    try:
        result = subprocess.run(["docker", "ps", "--filter", "ancestor=mlflow-automl-experiment", "-q"], stdout=subprocess.PIPE, text=True)
        container_ids = result.stdout.strip().split("\n")
        
        for container in container_ids:
            if container:
                subprocess.run(["docker", "stop", container])  # Stop container
                print(f"Stopped long-running experiment container: {container}")
    except Exception as e:
        print(f"Error stopping experiment containers: {e}")

# Function to run a Docker command with a strict timeout (50s)
def run_docker(command):
    print(f"Running: {command}")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Monitor execution time
        start_time = time.time()
        while process.poll() is None:
            time.sleep(1)
            if time.time() - start_time > 50:  # If more than 50 seconds, stop it
                print(f"Timeout reached! Killing: {command}")
                process.terminate()  # Kill the process
                stop_experiment_containers()  # Stop only experiment-related containers
                return

        print(f"Completed: {command}")
    except Exception as e:
        print(f"Error running command: {command}, Error: {e}")

# Run Docker commands in **small batches** to prevent memory overload
PARALLEL_LIMIT = 3  # Max concurrent executions
for i in range(0, len(docker_commands), PARALLEL_LIMIT):
    batch = docker_commands[i:i+PARALLEL_LIMIT]
    Parallel(n_jobs=len(batch))(delayed(run_docker)(cmd) for cmd in batch)

# Wait for MLflow logs to update
time.sleep(20)

# Get all completed runs from MLflow
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Default")  # Change if needed
experiment_id = experiment.experiment_id
runs = client.search_runs(experiment_id)

# Collect data into DataFrame
run_data = []
for run in runs:
    if run.info.status == "FINISHED":  # Only log completed runs
        run_data.append({
            "Run ID": run.info.run_id,
            "Preprocessing": run.data.params.get("preprocessing_algorithm", "Unknown"),
            "Postprocessing": run.data.params.get("postprocessing_algorithm", "Unknown"),
            "Accuracy": run.data.metrics.get("accuracy", 0),
            "Conformal Coverage": run.data.metrics.get("conformal_coverage", 0),
            "FLAML Best Run": run.data.metrics.get("flaml.best_run", 0)  # Capture FLAML best_run metric
        })

df = pd.DataFrame(run_data)
df.to_csv("mlflow_results.csv", index=False)

print("Experiment results saved to `mlflow_results.csv`")
print(df)

# Identify the best run (based on FLAML best_run)
if not df.empty:
    best_flaml_run = df.loc[df["FLAML Best Run"].idxmax()]
    print(f"Best Run (FLAML): {best_flaml_run}")
else:
    print("No successful runs were completed within the time limit.")

