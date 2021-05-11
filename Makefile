SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage deps help lint push test doc build

coverage:  ## Run tests with coverage
	coverage erase
	coverage run -m unittest -v stree.tests
	coverage report -m

deps:  ## Install dependencies
	pip install -r requirements.txt

lint:  ## Lint and static-check
	black stree
	flake8 stree
	mypy stree

push:  ## Push code with tags
	git push && git push --tags

test:  ## Run tests
	python -m unittest -v stree.tests

doc:  ## Update documentation
	make -C docs --makefile=Makefile html

build:  ## Build package
	rm -fr dist/*
	python setup.py sdist bdist_wheel

doc-clean:  ## Update documentation
	make -C docs --makefile=Makefile clean

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
