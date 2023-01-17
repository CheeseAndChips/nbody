USE_CUDA = false

PY_VERSION := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
PY_VERSION_MAJOR := $(word 1,${PY_VERSION})
PY_VERSION_MINOR := $(word 2,${PY_VERSION})
PY_INC := -I/usr/include/python$(PY_VERSION_MAJOR).$(PY_VERSION_MINOR)/
PY_LD := -lpython$(PY_VERSION_MAJOR).$(PY_VERSION_MINOR) -l:libboost_python$(PY_VERSION_MAJOR)$(PY_VERSION_MINOR).so -l:libboost_numpy$(PY_VERSION_MAJOR)$(PY_VERSION_MINOR).so

CC = g++
NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE_DIR = /usr/local/cuda/include
CFLAGS = -fPIC -Wall -std=c++17 -O2 $(PY_INC)
NVCCFLAGS = -std=c++17 -O2 -lineinfo
LDFLAGS = $(PY_LD) -lavcodec -lavutil -lpthread
SRCDIR = src
BINDIR = bin

SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(shell echo $(SRCS:.cpp=.o) | sed 's|src/|bin/|g')
DEPS := $(OBJS:.o=.d)
TARGET = $(BINDIR)/nbody.so

CUDA_SRCS := $(wildcard $(SRCDIR)/*.cu)
CUDA_OBJS := $(shell echo $(CUDA_SRCS:.cu=.o) | sed 's|src/|bin/|g')
CUDA_DEPS := $(CUDA_OBJS:.o=.cd)

ifeq ($(USE_CUDA), true)
CFLAGS += -DUSING_CUDA -I$(CUDA_INCLUDE_DIR)
LD = $(NVCC)
LDFLAGS_EXTRA = $(NVCCFLAGS) --compiler-options "$(CFLAGS) -shared"
else
LD = $(CC)
LDFLAGS_EXTRA = $(CFLAGS) -shared
endif

.PHONY: clean depend all

all: endiancheck depend $(TARGET)

$(BINDIR):
	@mkdir -p $(BINDIR)

$(TARGET): $(OBJS) $(CUDA_OBJS)
	$(LD) $(LDFLAGS_EXTRA) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(BINDIR)/*

endiancheck: $(BINDIR)/ec

$(BINDIR)/ec:
	@echo "Verifying endianness"
	@lscpu | grep Endian | grep -q "Little"
	@touch $@

depend: $(DEPS) $(CUDA_DEPS)

$(DEPS): $(BINDIR)/%.d: $(SRCDIR)/%.cpp
	@rm -f "$@"
	$(CC) -x c++ $(CFLAGS) -MT $(shell echo $@ | sed 's|\.d|.o|g') -MM $< >> $@

ifeq ($(USE_CUDA), true)
$(CUDA_DEPS): $(BINDIR)/%.cd: $(SRCDIR)/%.cu
	@rm -f "$@"
	$(NVCC) -x cu $(NVCCFLAGS) --compiler-options "$(CFLAGS)" -MT $(shell echo $@ | sed 's|\.cd|.o|g') -MM $< >> $@
else
$(CUDA_DEPS): $(BINDIR)/%.cd: $(SRCDIR)/%.cu
	@rm -f "$@"
	$(CC) -x c++ $(CFLAGS) -MT $(shell echo $@ | sed 's|\.cd|.o|g') -MM $< >> $@
endif

ifeq ($(filter $(MAKECMDGOALS),clean),)
include $(DEPS)
include $(CUDA_DEPS)
endif

$(OBJS):
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

ifeq ($(USE_CUDA), true)
$(CUDA_OBJS):
	$(NVCC) -x cu $(NVCCFLAGS) --compiler-options "$(CFLAGS)" -c $< -o $@
else
$(CUDA_OBJS):
	$(CC) -x c++ $(CFLAGS) -c $< -o $@
endif