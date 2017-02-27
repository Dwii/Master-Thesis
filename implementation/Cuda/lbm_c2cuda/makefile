include ../../lbm_common.mk

# Compiler
CC=gcc
NVCC=nvcc

# Binary target
BIN=lbmFlowAroundCylinder

# Binary arguments
NB_BLOCKS=1
NB_THREADS=0

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out
LIBDIR=../../C/libs

# Test output file
OUT=lbm.out

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=.o)))

# Flags
NVCC_FLAGS=-Wno-deprecated-gpu-targets -O3

# Non-standard libraries
LIBNAM=pgm array
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)
	
run: all $(OUTDIR) 
	$(BINDIR)/$(BIN) -p $(OUTDIR) -i $(ITER) -B $(NB_BLOCKS) -T $(NB_THREADS)

output: all $(OUTDIR)
	$(call timing, "$(BINDIR)/$(BIN) -i $(ITER) -f -B $(NB_BLOCKS) -T $(NB_THREADS) > $(OUTDIR)/$(OUT)", $(OUTDIR)/$(OUT).timing)

$(OBJDIR) $(BINDIR) $(OUTDIR): 
	mkdir -p $@

$(BINDIR)/$(BIN): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

-include $(OBJS:.o=.d)

$(OBJS): $(SRCS)
	$(NVCC) -M $(NVCC_FLAGS) $< -odir $(OBJDIR) $(INCS) > $(@:.o=.d) 
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $< $(INCS)

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR)

rebuild: clean all

.PHONY: all run output clean rebuild