PYTHON ?= python
OUTPUT_DIR ?= artifacts
DATA ?= data/titanic.csv

.PHONY: install check test extract clean train pipeline

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) -m compileall -q main.py pipeline.py tests

test:
	$(PYTHON) -m unittest discover -s tests -v

extract:
	$(PYTHON) main.py --stage extract --data $(DATA) --output-dir $(OUTPUT_DIR)

clean:
	$(PYTHON) main.py --stage clean --data $(DATA) --output-dir $(OUTPUT_DIR)

train:
	$(PYTHON) main.py --stage train --data $(DATA) --output-dir $(OUTPUT_DIR)

pipeline: check test extract clean train
