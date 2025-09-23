setup-env:
	pyenv shell 3.11.4
	source .venv/bin/activate

setup-logging: config/logging.py
	python -c "from config.logging import setup_logging; setup_logging(); print('Finished setting up logging.');"

test-logging-setup: tests/unit/test_logging_setup.py
	pytest tests/unit/test_logging_setup.py

test-utils-decorators: src/utils/logger.py
	pytest tests/unit/test_utils_decorators.py

clean-logfiles:
	rm -rf monitoring/logs

test-fastf1: tests/unit/test_fastf1_client.py
	pytest tests/unit/test_fastf1_client.py