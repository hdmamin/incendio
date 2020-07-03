SRC = $(wildcard notebooks/*.ipynb)

all: lib docs

lib: $(SRC)
	nbdev_build_lib
	touch incendio

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

pypi: dist
	twine upload --repository pypi dist/*

dist: clean_dist
	python setup.py sdist bdist_wheel

clean_dist:
	rm -rf dist

clean:
	nbdev_clean_nbs

