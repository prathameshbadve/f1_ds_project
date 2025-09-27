setup-env:
	pyenv shell 3.11.4
	source .venv/bin/activate

setup-logging: config/logging.py
	python -c "from config.logging import setup_logging; setup_logging(); print('Finished setting up logging.');"

test-logging-setup: tests/unit/test_logging_setup.py
	pytest tests/unit/test_logging_setup.py

test-utils-decorators: src/utils/logger.py
	pytest tests/unit/test_utils_decorators.py

test-fastf1: tests/unit/test_fastf1_client.py
	pytest tests/unit/test_fastf1_client.py

test-schedule-loader: tests/unit/test_schedule_loader.py
	pytest tests/unit/test_schedule_loader.py

test-session-loader: tests/unit/test_session_loader.py
	pytest tests/unit/test_session_loader.py

clean-logfiles:
	rm -rf monitoring/logs

clean-raw-data:
	rm -rf data/raw/2022

# Pipeline operations
.PHONY: run-full-ingestion run-ingestion-conservative run-ingestion-performance resume-ingestion list-failed-runs dry-run-ingestion

run-full-ingestion:
	@echo "Running full F1 data ingestion pipeline..."
	@echo "Usage: make run-full-ingestion SEASONS=2022-2024 SESSIONS=Q,R"
	$(PYTHON) scripts/data/run_full_ingestion.py $(if $(SEASONS),--seasons $(SEASONS),) $(if $(SESSIONS),--sessions $(SESSIONS),)

run-ingestion-conservative:
	@echo "Running conservative F1 data ingestion..."
	$(PYTHON) scripts/data/run_full_ingestion.py --preset conservative

run-ingestion-performance:
	@echo "Running high-performance F1 data ingestion..."
	$(PYTHON) scripts/data/run_full_ingestion.py --preset performance

resume-ingestion:
	@echo "Resuming failed data ingestion..."
	$(PYTHON) scripts/data/resume_failed_ingestion.py

list-failed-runs:
	@echo "Listing failed ingestion runs..."
	$(PYTHON) scripts/data/resume_failed_ingestion.py --list

dry-run-ingestion:
	@echo "Dry run of data ingestion (shows what would be processed)..."
	$(PYTHON) scripts/data/run_full_ingestion.py --dry-run $(if $(SEASONS),--seasons $(SEASONS),)