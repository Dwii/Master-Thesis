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
LIBDIR=../../C/libs

# Binary default arguments
OUTPRE=lbm_
NX=100
NY=100
NZ=10

DEFAULT_IMG_INTERVAL=100
DEFAULT_OUT_INTERVAL=0

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=.o)))

# Flags
OPT_FLAG=-DPALABOS_COMPATIBLE
CFLAGS=-g -Wall -Wextra -pedantic -O3
NVCC_FLAGS=-Wno-deprecated-gpu-targets -O3 --ptxas-options=-v $(OPT_FLAG)


# Non-standard libraries and headers
LIBNAM=pgm timing lbmmain3d_dynsize
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

run: images

images: all $(OUTDIR)
	$(eval INTERVAL ?= $(DEFAULT_IMG_INTERVAL))
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) -p -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -x $(NX) -y $(NY) -z $(NZ) $(BINARGS)" )

output: all $(OUTDIR) 
	$(eval INTERVAL ?= $(DEFAULT_OUT_INTERVAL))
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -x $(NX) -y $(NY) -z $(NZ) -L $(BINARGS)", $(OUTDIR)/$(OUTPRE)$(ITER).timing)

outpal: all $(OUTDIR) 
	$(eval INTERVAL ?= $(DEFAULT_OUT_INTERVAL))
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) -F -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -x $(NX) -y $(NY) -z $(NZ) $(BINARGS)")

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
