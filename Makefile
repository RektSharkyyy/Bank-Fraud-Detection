.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all help

# Default Python interpreter
VENV = .venv
PYTHON = $(VENV)\Scripts\python
MLFLOW_PORT ?= 5000

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  make install             - Install project dependencies and set up environment"
	@echo "  make data-pipeline       - Run the data pipeline"
	@echo "  make train-pipeline      - Run the training pipeline"
	@echo "  make streaming-inference - Run the streaming inference pipeline with the sample JSON"
	@echo "  make run-all             - Run all pipelines in sequence"
	@echo "  make clean               - Clean up artifacts"

# Install project dependencies and set up environment
install:
	@echo "Installing project dependencies..."
	@echo "Creating virtual environment..."
	@python -m venv .venv
	@echo "Installing dependencies..."
	@.venv\Scripts\python -m pip install --upgrade pip
	@.venv\Scripts\pip install -r requirements.txt
	@echo "Installation completed successfully!"
	@echo "To activate run: .\.venv\Scripts\Activate.ps1"

# Clean up
clean:
	@echo "Cleaning up artifacts..."
	rm -rf artifacts/models/*
	rm -rf artifacts/evaluation/*
	rm -rf artifacts/predictions/*
	rm -rf artifacts/encode/*
	rm -rf mlruns
	@echo "Cleanup completed!"

# Run data pipeline
data-pipeline:
	@echo "Start running data pipeline..."
	python pipelines/data_pipeline.py
	@echo "Data pipeline completed successfully!"

.PHONY: data-pipeline-rebuild
data-pipeline-rebuild:
	@$(PYTHON) -c "from pipelines.data_pipeline import data_pipeline; data_pipeline(force_rebuild=True)"

# Run training pipeline
train-pipeline:
	@echo "Running training pipeline..."
	python pipelines/model_training.py

# Run streaming inference pipeline with sample JSON
streaming-inference:
	@echo "Running streaming inference pipeline with sample JSON..."
	python pipelines/inference_pipeline.py

# Run all pipelines in sequence
run-all:
	@echo "Running all pipelines in sequence..."
	@echo "========================================"
	@echo "Step 1: Running data pipeline"
	@echo "========================================"
	@$(PYTHON) pipelines/data_pipeline.py
	@echo "\n========================================"
	@echo "Step 2: Running training pipeline"
	@echo "========================================"
	@$(PYTHON) pipelines/model_training.py
	@echo "\n========================================"
	@echo "Step 3: Running streaming inference pipeline"
	@echo "========================================"
	@$(PYTHON) pipelines/inference_pipeline.py
	@echo "\n========================================"
	@echo "All pipelines completed successfully!"
	@echo "========================================"

mlflow-ui:
	@echo "Launching MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop the server"
	@$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port $(MLFLOW_PORT)

# Stop all running MLflow servers
stop-all:
	@echo "Stopping all MLflow servers..."
	@echo "Finding MLflow processes on port $(MLFLOW_PORT)..."
	@-lsof -ti:$(MLFLOW_PORT) | xargs kill -9 2>/dev/null || true
	@echo "Finding other MLflow UI processes..."
	@-ps aux | grep '[m]lflow ui' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@-ps aux | grep '[g]unicorn.*mlflow' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@echo "✅ All MLflow servers have been stopped"