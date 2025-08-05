# KnowWhat Makefile
# Shortcuts for common tasks

.PHONY: help viz analysis test clean install

help:
	@echo "Available commands:"
	@echo "  make install    - Install all dependencies"
	@echo "  make viz        - Show detailed participant data (moves, recognition, generation)"
	@echo "  make analysis   - Run full analysis pipeline"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean generated files"

install:
	pip install -r requirements.txt
	cd infer && pip install -r requirements.txt

# Run visualization
viz:
	python analysis/core/visualize_participant_details.py

# Run full analysis pipeline
analysis:
	python analysis/core/run_full_analysis.py

# Run tests
test:
	python -m pytest tests/ -v

# Clean up generated files
clean:
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	rm -rf analysis_results/ 