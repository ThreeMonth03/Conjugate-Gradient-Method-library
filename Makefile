#Flags
CXX = g++
FLAGS = -O3 -g -m64 -Wall -shared -std=c++11 -fPIC 
PYBINCLUDE = $(shell python3-config --includes) $(shell python3 -m pybind11 --includes)

#PATH
MODULE_SHARE_OBJS_RLT_DIR = cpp
MODULE_SHARE_OBJS_ABS_DIR = $(shell pwd)/$(MODULE_SHARE_OBJS_RLT_DIR)
PYTHONPATH := $(MODULE_SHARE_OBJS_ABS_DIR):$(PYTHONPATH)
export PYTHONPATH

#Includes
CPP_FILE = $(wildcard $(MODULE_SHARE_OBJS_RLT_DIR)/*/*.cpp)
MODULE_SHARE_OBJS = $(MODULE_SHARE_OBJS_RLT_DIR)/_cgpy$(shell python3-config --extension-suffix)
PYBIND_TARGET = $(MODULE_SHARE_OBJS_RLT_DIR)/pybind/_pybind.o
TARGET = $(CPP_FILE:.cpp=.o)

#dependency
deps = $(TARGET:.o=.d)
FLAGS_DEP = -MMD -MP


#Makefile
.PHONY: all demo test clean
default: all

all: clean $(MODULE_SHARE_OBJS)

$(MODULE_SHARE_OBJS): $(TARGET)
	$(CXX) $(FLAGS) $^ -o $@

$(filter-out $(PYBIND_TARGET), $(TARGET)): %.o : %.cpp
	$(CXX) $(FLAGS) $(FLAGS_DEP) $(PYBINCLUDE)  -c $< -o $@

$(PYBIND_TARGET): %.o : %.cpp
	$(CXX) $(FLAGS) $(FLAGS_DEP) $(PYBINCLUDE)  -c $< -o $@

demo: $(MODULE_SHARE_OBJS)
#	mkdir -p demo/results
#	python3 demo/demo_matrix.py | tee demo/results/performance.txt
	python3 tests/test_cg_method.py

test: $(MODULE_SHARE_OBJS)
#	python3 -m pytest -v tests/test_matrix.py
	python3 -m pytest -v tests/test_cg_method.py

clean:
	rm -rf *.so cpp/*.so cpp/*/*.so
	rm -rf cpp/*/*.o
	rm -rf */__pycache__ cpp/*/__pycache__
	rm -rf .pytest_cache */.pytest_cache
	rm -rf demo/results
	rm -rf cpp/*/*.d

#dependency
-include $(deps)