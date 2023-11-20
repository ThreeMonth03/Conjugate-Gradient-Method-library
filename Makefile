#Flags
CXX = g++
FLAGS = -O3 -g -m64 -Wall -shared -std=c++11 -fPIC 
PYBINCLUDE = $(shell python3-config --includes) $(shell python3 -m pybind11 --includes)

#Includes
MODULE_SHARE_OBJS = cpp/_cgpy$(shell python3-config --extension-suffix)
MODULE_TARGET = cpp/matrix/_matrix.o

#PATH
MODULE_SHARE_OBJS_DIR = $(shell pwd)/cpp
PYTHONPATH := $(MODULE_SHARE_OBJS_DIR):$(PYTHONPATH)
export PYTHONPATH

#Tests
TEST_FILE = tests/test_matrix.py

#Makefile
.PHONY: all demo test clean

all: clean $(MODULE_SHARE_OBJS)

$(MODULE_SHARE_OBJS): $(MODULE_TARGET)	
	$(CXX) $(FLAGS) $^ -o $@

$(MODULE_TARGET): %.o : %.cpp %.hpp
	$(CXX) $(FLAGS) $(PYBINCLUDE)  -c $< -o $@


demo: $(MODULE_SHARE_OBJS)
	mkdir -p demo/results
	python3 demo/demo_matrix.py | tee demo/results/performance.txt

test: $(MODULE_SHARE_OBJS)
	python3 -m pytest -v tests/test_matrix.py

clean:
	rm -rf *.so cpp/*.so cpp/*/*.so
	rm -rf cpp/*/*.o
	rm -rf cpp/*/__pycache__ tests/__pycache__
	rm -rf .pytest_cache tests/.pytest_cache demo/.pytest_cache
	rm -rf demo/results
