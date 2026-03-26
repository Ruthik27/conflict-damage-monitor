# Tests
conda activate cdm
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
