include ../../lbm_common.mk

# Compiler
NVCC=nvcc

# Binary target
BIN=lbmSimple

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out
LIBDIR=../../Libraries

# Binary default arguments
OUTPRE=lbm_
NX=100
NY=100
NZ=10

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=.o)))

# Flags
SSIZE_FLAG=$(if $(SHARED_MEM_SIZE),-DSHARED_MEM_SIZE=$(SHARED_MEM_SIZE),)
OPT_FLAG=-DPALABOS_COMPATIBLE
CFLAGS=-g -Wall -Wextra -pedantic -O3
NVCC_FLAGS=-Wno-deprecated-gpu-targets -O3 --ptxas-options=-v -Xptxas -dlcm=cg $(OPT_FLAG) $(SSIZE_FLAG)

# Non-standard libraries and headers
LIBNAM=pgm timing lbmmain3d_dynsize
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
outpal: TRGTARGS=-F

# Targets defaut intervals
run output outpal: INTERVAL?=0
images: INTERVAL?=100

# Targets timing file path (left unset if not required)
output: TIMING=$(OUTDIR)/$(OUTPRE)$(ITER).timing

run images output outpal: all $(OUTDIR)
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) $(TRGTARGS) -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -x $(NX) -y $(NY) -z $(NZ) $(BINARGS)", $(TIMING))

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

optimized: ;@:
	$(eval OPT_FLAG=)

.PHONY: all run images output clean rebuild
