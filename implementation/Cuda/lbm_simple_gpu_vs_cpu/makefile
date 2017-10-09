include ../../lbm_common.mk
include ../lbm_gpu_vs_cpu_common.mk

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
LIBDIR=../../Libraries

# Binary default arguments
OUTPRE=lbm_
GPUPRE=gpu_$(OUTPRE)
CPUPRE=cpu_$(OUTPRE)

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
GPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_gpu.o)))
CPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_cpu.o)))
OBJS= $(GPUOBJS) $(CPUOBJS)

# Flags
NVCC_FLAGS=-Wno-deprecated-gpu-targets -O3

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

all: libraries $(OBJDIR) $(BINDIR) $(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN)

libraries: 
	@$(foreach library, $(LIBDEP), $(MAKE) -C $(library) | grep -v 'Nothing to be done' || $$true;)

# Targets addition arguments
images: TRGTARGS=-p
outcpu outgpu: TRGTARGS=-f -L

# Targets defaut intervals
run output outcpu outgpu: INTERVAL?=0
images: INTERVAL?=100
deltas: INTERVAL?=1

# Targets info file path (left unset if not required)
outcpu: INFO?=$(OUTDIR)/$(CPUPRE)$(ITER).info
outgpu: INFO?=$(OUTDIR)/$(GPUPRE)$(ITER).info

# Targets binary and prefix path
BIN=$(CPUBIN)
XPUPRE=$(CPUPRE)
outgpu: BIN=$(GPUBIN)
outgpu: XPUPRE=$(GPUPRE)

output: all $(OUTDIR)
	$(MAKE) outcpu outgpu -j 2 INTERVAL=$(INTERVAL)

deltas: all $(OUTDIR)
	$(MAKE) outcpu outgpu outdlt -j 3 INTERVAL=$(INTERVAL) INFO=/dev/null

run images outcpu outgpu: all $(OUTDIR)
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) $(TRGTARGS) -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(XPUPRE) $(BINARGS)", $(INFO))

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
	$(NVCC) -M $(NVCC_FLAGS) $(NVCC_XPU_FLAG) $< -odir $(OBJDIR) $(INCS) > $(@:.o=.d)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_XPU_FLAG) -o $@ -c $< $(INCS)

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR)

rebuild: clean all

.PHONY: all run images output outcpu outgpu deltas $(DLT_FILE) $(DLT_ITERS) clean rebuild 
