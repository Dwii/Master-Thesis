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
LIBDIR=../../Libraries

# Binary default arguments
OUTPRE=lbm_

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=.o)))

# Flags
NVCC_FLAGS=-Wno-deprecated-gpu-targets -O3

# Non-standard libraries
LIBNAM=pgm timing
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

all: libraries $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

libraries: 
	@$(foreach library, $(LIBDEP), $(MAKE) -C $(library) | grep -v 'Nothing to be done' || $$true;)
	
# Targets addition arguments
images: TRGTARGS=-p
output: TRGTARGS=-f -L

# Targets defaut intervals
run output: INTERVAL?=0
images: INTERVAL?=100

# Targets timing file path (left unset if not required)
output: TIMING=$(OUTDIR)/$(OUTPRE)$(ITER).timing

run images output: all $(OUTDIR)
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) $(TRGTARGS) -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) $(BINARGS)", $(TIMING))

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

.PHONY: all run images output clean rebuild