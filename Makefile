# AMRI Replication Package
# ========================
# Reproduce all results from:
#   "Adaptive Misspecification-Robust Confidence Intervals:
#    Near-Minimax Optimal Inference via Soft-Threshold Blending"
#
# Usage:
#   make install      Install dependencies
#   make quick        Quick validation (~5 min)
#   make full         Full reproduction (~2-4 hours)
#   make figures      Generate publication figures only
#   make test         Run unit tests
#   make paper        Compile LaTeX manuscript
#   make clean        Remove generated files
#   make all          Full pipeline: install + test + full + figures + paper

PYTHON ?= python
LATEX ?= pdflatex

.PHONY: all install quick full figures test paper clean help

help:
	@echo "AMRI Replication Package"
	@echo "========================"
	@echo "  make install   - Install Python dependencies"
	@echo "  make quick     - Quick validation (~5 min)"
	@echo "  make full      - Full reproduction (~2-4 hours)"
	@echo "  make figures   - Generate publication figures"
	@echo "  make test      - Run unit tests"
	@echo "  make paper     - Compile LaTeX manuscript"
	@echo "  make clean     - Remove generated files"
	@echo "  make all       - Full pipeline"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

quick:
	$(PYTHON) run_all.py --quick

full:
	$(PYTHON) run_all.py --full

figures:
	$(PYTHON) src/publication_figures.py

paper: paper/main.tex
	cd paper && $(LATEX) main.tex && $(LATEX) main.tex
	cd paper && $(LATEX) supplement.tex && $(LATEX) supplement.tex

clean:
	rm -rf results/*.csv
	rm -rf figures/*.png figures/*.pdf
	rm -rf paper/*.aux paper/*.log paper/*.out paper/*.toc
	rm -rf __pycache__ src/__pycache__ amri/__pycache__
	rm -rf *.egg-info build dist

all: install test full figures paper
	@echo "All steps completed successfully."
