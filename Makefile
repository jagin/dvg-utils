clean:
	rm -rf output
	rm -rf .pytest_cache
	rm -rf dvg_utils.egg-info
	rm -rf dist
	rm -rf build
	rm -f *.log

init:
	pip install -r requirements.txt

test:
	python -m pytest

dist-test: clean
	python setup.py sdist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

dist: clean
	python setup.py sdist
	twine upload dist/*
