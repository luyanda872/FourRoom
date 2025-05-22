# Name of the virtual environment directory
ENV_NAME = venv

# Detect OS type
ifeq ($(OS),Windows_NT)
    ACTIVATE = $(ENV_NAME)\Scripts\activate
    PYTHON = $(ENV_NAME)\Scripts\python.exe
    PIP = $(ENV_NAME)\Scripts\pip.exe
else
    ACTIVATE = . $(ENV_NAME)/bin/activate
    PYTHON = $(ENV_NAME)/bin/python
    PIP = $(ENV_NAME)/bin/pip
endif

.PHONY: all setup run1 run2 run3 clean

all: setup

setup:
	@echo "Creating virtual environment..."
	python -m venv $(ENV_NAME)
	@echo "Upgrading pip safely..."
	$(PYTHON) -m pip install --upgrade pip
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Setup complete."

run1:
	@echo "Running Scenario 1..."
	$(PYTHON) Scenario1.py

run2:
	@echo "Running Scenario 2..."
	$(PYTHON) Scenario2.py

run3:
	@echo "Running Scenario 3..."
	$(PYTHON) Scenario3.py

run1-stoch:
	@echo "Running Scenario 1 (Stochastic)..."
	$(PYTHON) Scenario1.py --stochastic

run2-stoch:
	@echo "Running Scenario 2 (Stochastic)..."
	$(PYTHON) Scenario2.py --stochastic

run3-stoch:
	@echo "Running Scenario 3 (Stochastic)..."
	$(PYTHON) Scenario3.py --stochastic

clean:
	@echo "Removing virtual environment..."
	rm -rf $(ENV_NAME)
	rm *.png
