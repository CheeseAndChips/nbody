USE_CUDA = false

SRCS := $(wildcard *.cpp)
CUDA_SRCS := $(wildcard *.cu)
TARGET = $(OBJDIR)/nbody.so

PY_VERSION := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
PY_VERSION_MAJOR := $(word 1,${PY_VERSION})
PY_VERSION_MINOR := $(word 2,${PY_VERSION})
PY_INC := -I/usr/include/python$(PY_VERSION_MAJOR).$(PY_VERSION_MINOR)/
PY_LD := -lpython$(PY_VERSION_MAJOR).$(PY_VERSION_MINOR) -l:libboost_python$(PY_VERSION_MAJOR)$(PY_VERSION_MINOR).so -l:libboost_numpy$(PY_VERSION_MAJOR)$(PY_VERSION_MINOR).so

CXX = g++
NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE_DIR = /usr/local/cuda/include
DEPFLAGS = -MT $@ -MD -MP -MF $(DEPDIR)/$*.Td
CPPFLAGS = -fPIC -Wall -std=c++17 -O2 $(PY_INC)
NVCCFLAGS = $(PY_INC) -std=c++17 -lineinfo --compiler-options "-fPIC -Wall -std=c++17 -O2" -DUSING_CUDA
LDFLAGS :=
LDLIBS := $(PY_LD) -lavcodec -lavutil -lpthread
OBJDIR = bin
DEPDIR = $(OBJDIR)/deps

OBJS := $(patsubst %,$(OBJDIR)/%.cpp.o,$(basename $(SRCS)))
DEPS := $(patsubst %,$(DEPDIR)/%.cpp.d,$(basename $(SRCS)))

CUDA_OBJS := $(patsubst %,$(OBJDIR)/%.cu.o,$(basename $(CUDA_SRCS)))
CUDA_DEPS := $(patsubst %,$(DEPDIR)/%.cu.d,$(basename $(CUDA_SRCS)))

POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

ifeq ($(USE_CUDA), true)
CPPFLAGS += -DUSING_CUDA
NVCCFLAGS += -DUSING_CUDA -I$(CUDA_INCLUDE_DIR)
LD = $(NVCC)
LDFLAGS += --compiler-options "-shared"
else
LD = $(CXX)
LDFLAGS += -shared
endif

COMPILE.cc = $(CXX) -x c++ $(DEPFLAGS) $(CPPFLAGS) -c -o $@
COMPILE.cu = $(NVCC) -x cu $(DEPFLAGS) $(NVCCFLAGS) -c -o $@
LINK.o = $(LD) $(LDFLAGS) $(LDLIBS) -o $@
PRECOMPILE =
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

# create directories
$(shell mkdir -p $(OBJDIR) >/dev/null)
$(shell mkdir -p $(DEPDIR) >/dev/null)

# all: endiancheck depend $(TARGET)
all: $(TARGET)

.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS) $(TARGET) $(OBJDIR)/ec

# endiancheck: $(OBJDIR)/ec

# $(OBJDIR)/ec:
# 	@echo "Verifying endianness"
# 	@lscpu | grep Endian | grep -q "Little"
# 	@touch $@

$(TARGET): $(OBJS) $(CUDA_OBJS)
	$(LINK.o) $^

$(OBJDIR)/%.cpp.o: %.cpp
$(OBJDIR)/%.cpp.o: %.cpp $(DEPDIR)/%.cu.d
	$(PRECOMPILE)
	$(COMPILE.cc) $<
	$(POSTCOMPILE)

$(OBJDIR)/%.cu.o: %.cu
$(OBJDIR)/%.cu.o: %.cu $(DEPDIR)/%.cu.d
	$(PRECOMPILE)
	$(COMPILE.cu) $<
	$(POSTCOMPILE)

.PRECIOUS: $(DEPDIR)/%.cpp.d
$(DEPDIR)/%.cpp.d: ;

-include $(DEPS)

.PRECIOUS: $(DEPDIR)/%.cu.d
$(DEPDIR)/%.cu.d: ;

-include $(CUDA_DEPS)