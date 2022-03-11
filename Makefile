USE_CUDA = true

CC = g++
NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -Wall -std=c++17 -O2
NVCCFLAGS = -std=c++17 -O2
LDFLAGS = -lavcodec -lavutil -lpthread
SRCDIR = src
BINDIR = bin

SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(shell echo $(SRCS:.cpp=.o) | sed 's|src/|bin/|g')
DEPS := $(OBJS:.o=.d)
TARGET = $(BINDIR)/nbody

CUDA_SRCS := $(wildcard $(SRCDIR)/*.cu)
CUDA_OBJS := $(shell echo $(CUDA_SRCS:.cu=_cuobj.o) | sed 's|src/|bin/|g')
CUDA_DEPS := $(CUDA_OBJS:_cuobj.o=.cd)
CUDA_DEPS_REGULAR := $(CUDA_DEPS:.cd=.d)

ifeq ($(USE_CUDA), true)
LD = $(NVCC)
LDFLAGS_EXTRA = $(NVCCFLAGS) --compiler-options "$(CFLAGS)"
else
LD = $(CC)
LDFLAGS_EXTRA = $(CFLAGS)
endif

# SRCS += $(CUDA_SRCS)
# OBJS += $(shell echo $(CUDA_SRCS:.cu=.o) | sed 's|src/|bin/|g')

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

# depend: $(DEPS) $(CUDA_DEPS) $(CUDA_DEPS_REGULAR)
depend: $(DEPS) $(CUDA_DEPS)

$(DEPS): $(BINDIR)/%.d: $(SRCDIR)/%.cpp
	@rm -f "$@"
	$(CC) -x c++ $(CFLAGS) -MT $(shell echo $@ | sed 's|\.d|.o|g') -MM $< >> $@

$(CUDA_DEPS): $(BINDIR)/%.cd: $(SRCDIR)/%.cu
	@rm -f "$@"
	$(NVCC) -x cu $(NVCCFLAGS) --compiler-options "$(CFLAGS)" -MT $(shell echo $@ | sed 's|\.cd|_cuobj.o|g') -MM $< >> $@

# $(CUDA_DEPS_REGULAR): $(BINDIR)/%.d: $(SRCDIR)/%.cu
#	@rm -f "$@"
#	$(CC) -x c++ $(CFLAGS) -MT $(shell echo $@ | sed 's|\.d|.o|g') -MM $< >> $(shell echo $@ | sed 's|\.cd|.d|g')

ifeq ($(filter $(MAKECMDGOALS),clean),)
include $(DEPS)
include $(CUDA_DEPS)
# include $(CUDA_DEPS_REGULAR)
endif

$(OBJS):
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

$(CUDA_OBJS):
	$(NVCC) -x cu $(NVCCFLAGS) --compiler-options "$(CFLAGS)" -c $< -o $@