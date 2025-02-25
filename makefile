.PHONY: prepare train evaluate clean mlflow_server all_with_ui

# Variables
TRAIN_DATA = churn-bigml-80.csv
TEST_DATA = churn-bigml-20.csv
OUTPUT = prepared_data.pkl
MODEL = model.pkl

# Start MLflow server in the background
mlflow_server:
	@echo "Starting MLflow server in the background..."
	@nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 > mlflow.log 2>&1 &
	@sleep 2    # Wait for the server to start

# Prepare data: this saves the processed training and test data to pickle files.
prepare: $(OUTPUT)

$(OUTPUT):
	@echo "Preparing data..."
	python3 main.py --mode prepare --train_data $(TRAIN_DATA) --test_data $(TEST_DATA) --output $(OUTPUT)

# Train model: reads the raw CSVs, processes them, trains the model, and saves it.
train: $(MODEL)

$(MODEL): $(OUTPUT)
	@echo "Training model..."
	python3 main.py --mode train --train_data $(TRAIN_DATA) --test_data $(TEST_DATA) --save $(MODEL)

# Evaluate model: loads the saved model and scaler, processes the test data using the saved scaler, and evaluates.
evaluate: $(MODEL)
	@echo "Evaluating model..."
	python3 main.py --mode evaluate --load $(MODEL) --test_data $(TEST_DATA)

# Clean: remove generated pickle and model files.
clean:
	rm -f $(OUTPUT) $(MODEL) *.pkl
	rm -rf ./artifacts ./mlruns

# Run all steps (prepare, train, evaluate)
all: prepare train evaluate

# Run all steps with MLflow server and UI
all_with_ui: mlflow_server all
	@echo "Pipeline complete. MLflow server is running in the background."
	@echo "Please manually open http://127.0.0.1:5000 in your browser."
