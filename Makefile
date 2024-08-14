SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: audit coverage help lint test doc doc-clean build

coverage:  ## Run tests with coverage
	@coverage erase
	@coverage run -m unittest -v stree.tests
	@coverage report -m

lint:  ## Lint and static-check
	@black stree
	@flake8 stree

test:  ## Run tests
	@python -m unittest -v stree.tests

doc:  ## Update documentation
	@make -C docs --makefile=Makefile html

build:  ## Build package
	@rm -fr dist/*
	@rm -fr build/*
	@hatch build

doc-clean:  ## Clean documentation folders
	@make -C docs --makefile=Makefile clean

audit: ## Audit pip
	@pip-audit

help: ## Show this help message
	@IFS=$$'\n' ; \
	help_lines=(`grep -Fh "##" $(MAKEFILE_LIST) | grep -Fv fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
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
