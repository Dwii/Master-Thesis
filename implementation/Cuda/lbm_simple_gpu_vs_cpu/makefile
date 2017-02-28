include ../../lbm_common.mk

# Compiler
CC=gcc
NVCC=nvcc

# Binary target
GPUBIN=lbmSimple_gpu
CPUBIN=lbmSimple_cpu

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
GPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_gpu.o)))
CPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_cpu.o)))
OBJS= $(GPUOBJS) $(CPUOBJS)

# Flags
NVCC_FLAGS=-Wno-deprecated-gpu-targets -O3

# Non-standard libraries
LIBNAM=pgm array
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN)

run: images

images: all $(OUTDIR)
ifeq ($(strip $(INTERVAL)),)
	$(eval INTERVAL=$(DEFAULT_IMG_INTERVAL))
endif
	$(BINDIR)/$(GPUBIN) -p -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) 

output: all $(OUTDIR) 
ifeq ($(strip $(INTERVAL)),)
	$(eval INTERVAL=$(DEFAULT_OUT_INTERVAL))
endif
	$(eval GPUPRE=gpu_$(OUTPRE))
	$(eval CPUPRE=cpu_$(OUTPRE))
	$(call timing, "$(BINDIR)/$(GPUBIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(GPUPRE)", $(OUTDIR)/$(GPUPRE)$(ITER).timing)
	$(call timing, "$(BINDIR)/$(CPUBIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(CPUPRE)", $(OUTDIR)/$(CPUPRE)$(ITER).timing)

$(OBJDIR) $(BINDIR) $(OUTDIR): 
	mkdir -p $@

$(BINDIR)/$(GPUBIN): $(GPUOBJS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

$(BINDIR)/$(CPUBIN): $(CPUOBJS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

-include $(OBJS:.o=.d)

$(GPUOBJS): NVCC_XPU_FLAG=-DCOMPUTE_ON_GPU
$(CPUOBJS): NVCC_XPU_FLAG=-DCOMPUTE_ON_CPU
$(OBJS): $(SRCS)
	$(NVCC) -M $(NVCC_FLAGS) $(NVCC_XPU_FLAG) $< -odir $(OBJDIR) $(INCS) > $(@:.o=.d)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_XPU_FLAG) -o $@ -c $< $(INCS)

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR)

rebuild: clean all

.PHONY: all run images output clean rebuild