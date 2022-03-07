CC = g++
CFLAGS = -Wall -std=c++17 -O2
LDFLAGS = -lavcodec -lavutil
SRCDIR = src
BINDIR = bin

SRCS := $(wildcard $(SRCDIR)/*.cpp)
HEADERS := $(wildcard $(SRCDIR)/*.h)
OBJS := $(shell echo $(SRCS:.cpp=.o) | sed 's|src/|bin/|g')
DEPS := $(OBJS:.o=.d)
TARGET = $(BINDIR)/nbody

.PHONY: clean depend all

all: endiancheck depend $(TARGET)

$(BINDIR):
	@mkdir -p $(BINDIR)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(BINDIR)/*

endiancheck: $(BINDIR)/ec

$(BINDIR)/ec:
	@echo "Verifying endianness"
	@lscpu | grep Endian | grep -q "Little"
	@touch $@

depend: $(DEPS)

$(DEPS): $(BINDIR)/%.d: $(SRCDIR)/%.cpp
	@rm -f "$@"
	@echo -n "$(BINDIR)/" > $@
	$(CC) $(CFLAGS) -MM $< >> $@

ifeq ($(filter $(MAKECMDGOALS),clean),)
include $(DEPS)
endif

$(OBJS):
	$(CC) $(CFLAGS) -c $< -o $@