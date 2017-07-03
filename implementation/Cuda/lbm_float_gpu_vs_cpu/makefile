include ../../lbm_common.mk
include ../lbm_gpu_vs_cpu_common.mk

# Compiler
CC=gcc
NVCC=nvcc

# Binary target
GPUBIN=lbmFlowAroundCylinder_gpu
CPUBIN=lbmFlowAroundCylinder_cpu

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out
LIBDIR=../../C/libs

# Binary default arguments
OUTPRE=lbm_
GPUPRE=gpu_$(OUTPRE)
CPUPRE=cpu_$(OUTPRE)

DEFAULT_IMG_INTERVAL=100
DEFAULT_OUT_INTERVAL=0

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
GPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_gpu.o)))
CPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_cpu.o)))
OBJS= $(GPUOBJS) $(CPUOBJS)

# Flags
NVCC_FLAGS=-Wno-deprecated-gpu-targets -O3
NVCC_OTHER_FLAGS=-D USE_GPU_OPERATORS

# Non-standard libraries
LIBNAM=pgm timing
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

# Deltas vars
DLT_FILE=$(OUTDIR)/deltas.csv
DLT_ITERS=$(if $(subst 0,,$(INTERVAL)), $(shell seq $(INTERVAL) $(INTERVAL) $(ITER)), $(ITER))
OUT_SIZE=44226000 
# OUT_SIZE = NX * NY * F * Print digits (64+the dot) = 420*180*9*65

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN)

run: images

images: all $(OUTDIR)
	$(eval INTERVAL ?= $(DEFAULT_IMG_INTERVAL))
	$(BINDIR)/$(CPUBIN) -p -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) 

output: all $(OUTDIR)
	$(eval INTERVAL ?= $(DEFAULT_OUT_INTERVAL))
	$(MAKE) outcpu outgpu -j 2 INTERVAL=$(INTERVAL)

outcpu: 
	$(call execute, $(LIBDEP), "$(BINDIR)/$(CPUBIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(CPUPRE) -L", $(OUTDIR)/$(CPUPRE)$(ITER).timing)

outgpu: 
	$(call execute, $(LIBDEP), "$(BINDIR)/$(GPUBIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(GPUPRE) -L", $(OUTDIR)/$(GPUPRE)$(ITER).timing)

deltas: all $(OUTDIR)
	$(eval INTERVAL ?= 1)
	$(MAKE) outcpu outgpu outdlt -j 3 INTERVAL=$(INTERVAL)

outdlt: $(DLT_FILE) $(DLT_ITERS)

$(DLT_FILE): 
	printf "" > $@

$(DLT_ITERS): GPUOUT=$(OUTDIR)/$(GPUPRE)$@.out
$(DLT_ITERS): CPUOUT=$(OUTDIR)/$(CPUPRE)$@.out
$(DLT_ITERS): 
	$(call wait_output,$(GPUOUT),$(OUT_SIZE))
	$(call wait_output,$(CPUOUT),$(OUT_SIZE))
	printf "$@;%s\n" $$($(DLT_SCRIPT_SHELL) $(DLT_SCRIPT) $(GPUOUT) $(CPUOUT)) >> $(DLT_FILE)
	rm $(GPUOUT) $(CPUOUT)

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
	$(NVCC) -M $(NVCC_FLAGS) $(NVCC_OTHER_FLAGS) $(NVCC_XPU_FLAG) $< -odir $(OBJDIR) $(INCS) > $(@:.o=.d)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_OTHER_FLAGS) $(NVCC_XPU_FLAG) -o $@ -c $< $(INCS)

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR)

rebuild: clean all

.PHONY: all run images output outcpu outgpu deltas fill_deltas $(DLT_FILE) $(DLT_ITERS) clean rebuild 
