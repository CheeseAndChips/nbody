USE_CUDA = false

SRCDIR := src
OBJDIR := bin
DEPDIR := $(OBJDIR)/deps
TARGET := $(OBJDIR)/nbody.so

CXX := g++
NVCC := /usr/local/cuda/bin/nvcc
CUDA_INCLUDE_DIR := /usr/local/cuda/include

SHAREDFLAGS := -fPIC -Wall -std=c++17 -O2 -shared
CPPFLAGS = $(SHAREDFLAGS) $(PY_INC)
NVCCFLAGS = -std=c++17 -O2 -lineinfo $(PY_INC) --compiler-options "$(SHAREDFLAGS)"
LDFLAGS := -shared
LDLIBS = $(PY_LD) -lavcodec -lavutil -lpthread

ifeq ($(USE_CUDA), true)
CPPFLAGS += -DUSING_CUDA -I$(CUDA_INCLUDE_DIR)
NVCCFLAGS += -DUSING_CUDA -I$(CUDA_INCLUDE_DIR)
LD := $(NVCC)
CU_LANG := cu
LDFLAGS := --compiler-options $(LDFLAGS)
else
NVCCFLAGS := $(CPPFLAGS)
LD := $(CXX)
CU_LANG := c++
NVCC := $(CXX)
endif

SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(patsubst $(SRCDIR)/%,$(OBJDIR)/%.cpp.o,$(basename $(SRCS)))
DEPS := $(patsubst $(SRCDIR)/%,$(DEPDIR)/%.cpp.d,$(basename $(SRCS)))
CUDA_SRCS := $(wildcard $(SRCDIR)/*.cu)
CUDA_OBJS := $(patsubst $(SRCDIR)/%,$(OBJDIR)/%.cu.o,$(basename $(CUDA_SRCS)))
CUDA_DEPS := $(patsubst $(SRCDIR)/%,$(DEPDIR)/%.cu.d,$(basename $(CUDA_SRCS)))
STUB := $(patsubst %.so,%.pyi,$(TARGET))

PY_VERSION := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
PY_VERSION_MAJOR := $(word 1,${PY_VERSION})
PY_VERSION_MINOR := $(word 2,${PY_VERSION})
PY_INC := -I/usr/include/python$(PY_VERSION_MAJOR).$(PY_VERSION_MINOR)/
PY_LD := -lpython$(PY_VERSION_MAJOR).$(PY_VERSION_MINOR) -l:libboost_python$(PY_VERSION_MAJOR)$(PY_VERSION_MINOR).so -l:libboost_numpy$(PY_VERSION_MAJOR)$(PY_VERSION_MINOR).so

# create directories
$(shell mkdir -p $(OBJDIR) >/dev/null)
$(shell mkdir -p $(DEPDIR) >/dev/null)

all: $(OBJDIR)/ec $(TARGET)

.PHONY: clean
clean:
	rm -f $(OBJS) $(CUDA_OBJS) $(DEPS) $(CUDA_DEPS) $(TARGET) $(STUB) $(OBJDIR)/ec

$(OBJDIR)/ec:
	@echo "Verifying endianness"
	@lscpu | grep Endian | grep -q "Little"
	@touch $@

$(TARGET): $(OBJS) $(CUDA_OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)
	stubgen -m nbody --search-path=$(OBJDIR) -o $(OBJDIR)

$(OBJDIR)/%.cpp.o: $(SRCDIR)/%.cpp
$(OBJDIR)/%.cpp.o: $(SRCDIR)/%.cpp $(DEPDIR)/%.cu.d
	$(CXX) -x c++ -MT $@ -MMD -MP -MF $(DEPDIR)/$*.cpp.Td $(CPPFLAGS) -c -o $@ $<
	mv -f $(DEPDIR)/$*.cpp.Td $(DEPDIR)/$*.cpp.d && touch $@

$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu $(DEPDIR)/%.cu.d
	$(NVCC) -x $(CU_LANG) -MT $@ -MMD -MP -MF $(DEPDIR)/$*.cu.Td $(NVCCFLAGS) -c -o $@ $<
	mv -f $(DEPDIR)/$*.cu.Td $(DEPDIR)/$*.cu.d && touch $@

.PRECIOUS: $(DEPDIR)/%.cpp.d
$(DEPDIR)/%.cpp.d: ;

-include $(DEPS)

.PRECIOUS: $(DEPDIR)/%.cu.d
$(DEPDIR)/%.cu.d: ;

-include $(CUDA_DEPS)