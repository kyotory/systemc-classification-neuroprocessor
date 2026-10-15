CXX ?= g++
SYSTEMC_HOME ?=
SYSTEMC_INC_DIR ?= $(if $(SYSTEMC_HOME),$(SYSTEMC_HOME)/include,/usr/include)
SYSTEMC_LIB_DIR ?= $(if $(SYSTEMC_HOME),$(SYSTEMC_HOME)/lib-linux64,/usr/lib/x86_64-linux-gnu)
TARGET := neuro_accelerator
INCLUDE_DIR := include
SOURCE_DIR := src
BUILD_DIR := build

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS += -I$(INCLUDE_DIR) -I$(SYSTEMC_INC_DIR) -MMD -MP
LDLIBS += -L$(SYSTEMC_LIB_DIR) -lsystemc
LDFLAGS += -Wl,-rpath,$(SYSTEMC_LIB_DIR)
SOURCES := $(wildcard $(SOURCE_DIR)/*.cpp)
OBJECTS := $(patsubst $(SOURCE_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPFILES := $(OBJECTS:.o=.d)

.PHONY: all clean run
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

-include $(DEPFILES)

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) --pe 4

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
