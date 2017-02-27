# Compiler
CC=gcc
NVCC=nvcc

# Binary target
BIN=cudaGetDevicesProperties

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=.o)))

# Flags
NVCC_FLAGS=-Wno-deprecated-gpu-targets -O3

# Libraries
LIBS=-lm

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)
	
run: all 
	$(BINDIR)/$(BIN)

$(OBJDIR) $(BINDIR) $(OUTDIR): 
	mkdir -p $@

$(BINDIR)/$(BIN): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

-include $(OBJS:.o=.d)

$(OBJS): $(SRCS)
	$(NVCC) -M $(NVCC_FLAGS) $< -odir $(OBJDIR) > $(@:.o=.d)
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR)

rebuild: clean all

.PHONY: all run clean rebuild
