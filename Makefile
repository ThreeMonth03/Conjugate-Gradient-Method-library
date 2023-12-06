#Flags
CXX = g++
FLAGS = -O3 -g -m64 -Wall -shared -std=c++11 -fPIC -fopenmp
PYBINCLUDE = $(shell python3-config --includes) $(shell python3 -m pybind11 --includes)
FLAGS_DEP = -MMD -MP
DIRS = $(shell find $(shell pwd)/cpp/* -type d)
CXXINCLUDE := $(patsubst %,-I %,$(DIRS))
#PATH
MODULE_SHARE_OBJS_RLT_DIR = cpp
MODULE_SHARE_OBJS_ABS_DIR = $(shell pwd)/$(MODULE_SHARE_OBJS_RLT_DIR)
PYTHONPATH := $(MODULE_SHARE_OBJS_ABS_DIR):$(PYTHONPATH)
export PYTHONPATH

#Includes
CPP_FILE = $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/*/*.cpp)
MODULE_SHARE_OBJS = $(MODULE_SHARE_OBJS_RLT_DIR)/_cgpy$(shell python3-config --extension-suffix)
TARGET = $(CPP_FILE:.cpp=.o)

#dependency
DEPS = $(TARGET:.o=.d)
-include $(DEPS)

#Makefile
.PHONY: all demo test clean
default: all

all: $(MODULE_SHARE_OBJS)

$(MODULE_SHARE_OBJS): $(TARGET)
	$(CXX) $(FLAGS) $^ -o $@

$(TARGET): %.o : %.cpp
	$(CXX) $(FLAGS) $(FLAGS_DEP) $(PYBINCLUDE) $(CXXINCLUDE)  -c $< -o $@


demo: $(MODULE_SHARE_OBJS)
	mkdir -p demo/results
	python3 demo/demo_matrix.py | tee demo/results/performance.txt
	python3 tests/test_cg_method.py

test: $(MODULE_SHARE_OBJS)
	python3 -m pytest -v tests/test_matrix.py
#python3 -m pytest -v tests/test_cg_method.py

clean:
	rm -rf *.so cpp/*.so cpp/*/*.so
	rm -rf cpp/*/*.o
	rm -rf */__pycache__ cpp/*/__pycache__
	rm -rf .pytest_cache */.pytest_cache
	rm -rf demo/results
	rm -rf cpp/*/*.d
