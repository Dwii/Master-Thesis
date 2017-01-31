# Compiler
CC=gcc
NVCC=nvcc

# Binary target
BIN=lbmFlowAroundCylinder

# Binary arguments
ITER=200000

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out

# Test output file
TST=test.out

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=.o)))

# Flags
NVCC_FLAGS=-arch=sm_30

# Libraries
LIBS=-lm

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)
	
run: all 
	$(BINDIR)/$(BIN) -i $(ITER)

test: all $(OUTDIR) 
	$(BINDIR)/$(BIN) -i $(ITER) -f > $(OUTDIR)/$(TST) 

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

.PHONY: all run test clean rebuild