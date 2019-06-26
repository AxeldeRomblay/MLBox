clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	cd tests/; \
		rm -fr .tox/; \
		rm -f .coverage; \
		rm -fr htmlcov/

test: ## run tests quickly with the default Python
	cd tests/; \
		pytest

coverage: ## check code coverage quickly with the default Python
	cd tests/; \
		coverage run -m --source=../mlbox/ pytest;\
		coverage html;\
		$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/mlbox.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ mlbox
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

release: clean ## package and upload a release
	python setup.py sdist upload
	python setup.py bdist_wheel upload

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

develop: clean ## install the package to the active Python's site-packages in developer mode
	python setup.py develop
