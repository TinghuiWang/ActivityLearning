# Makefile
#
# Copyright (C) 2015 Tinghui Wang <tinghui.wang@wsu.edu>
# 
# License: 3-Clause BSD
#
# Package Setup and Compilation

CYTHON ?= cython
PYTHON ?= python
CTAGS ?= ctags

all: clean compile test

install: cython compile
	$(PYTHON) setup.py install --user

clean-ctags:
	rm -fv tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -rfv build

compile:
	$(PYTHON) setup.py build_ext -i

cython:
	find actlearn -name "*.pyx" -exec $(CYTHON) {} \;

test:
	echo 'Tests To Be Implemented'
