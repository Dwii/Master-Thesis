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

# Binary default arguments
OUTPRE=lbm_

DEFAULT_IMG_INTERVAL=100
DEFAULT_OUT_INTERVAL=0

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
	
run: images

images: all $(OUTDIR)
ifeq ($(strip $(INTERVAL)),)
	$(eval INTERVAL=$(DEFAULT_IMG_INTERVAL))
endif
	$(BINDIR)/$(BIN) -p -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) 

output: all $(OUTDIR) 
ifeq ($(strip $(INTERVAL)),)
	$(eval INTERVAL=$(DEFAULT_OUT_INTERVAL))
endif
	$(call timing, "$(BINDIR)/$(BIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE)", $(OUTDIR)/$(OUTPRE)$(ITER).timing)

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