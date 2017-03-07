include ../../lbm_common.mk

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
TOOLDIR=../../Tools

# Binary default arguments
OUTPRE=lbm_
GPUPRE=gpu_$(OUTPRE)
CPUPRE=cpu_$(OUTPRE)

DEFAULT_IMG_INTERVAL=100
DEFAULT_OUT_INTERVAL=0
DEFAULT_DLT_INTERVAL=1000

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
GPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_gpu.o)))
CPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_cpu.o)))
OBJS= $(GPUOBJS) $(CPUOBJS)

# Flags
NVCC_FLAGS=-Wno-deprecated-gpu-targets -O3
NVCC_OTHER_FLAGS=-D USE_GPU_OPERATORS

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
	$(call timing, "$(BINDIR)/$(GPUBIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(GPUPRE)", $(OUTDIR)/$(GPUPRE)$(ITER).timing)
	$(call timing, "$(BINDIR)/$(CPUBIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(CPUPRE)", $(OUTDIR)/$(CPUPRE)$(ITER).timing)

# OUT_SIZE = NX * NY * F * Print digits (64+the dot) = 420*180*9*65
deltas: OUT_SIZE=44226000 
deltas: all $(OUTDIR)
ifeq ($(strip $(INTERVAL)),)
	$(eval INTERVAL=$(DEFAULT_DLT_INTERVAL))
endif
	$(BINDIR)/$(GPUBIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(GPUPRE) &
	$(BINDIR)/$(CPUBIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(CPUPRE) &

	for iter in $$(seq $(INTERVAL) $(INTERVAL) $(ITER)) ; do                                                                                       \
		gpu_out=$(OUTDIR)/$(GPUPRE)$$iter.out ;                                                                                                    \
		cpu_out=$(OUTDIR)/$(CPUPRE)$$iter.out ;                                                                                                    \
		while [[ ! -s $$gpu_out || ! -s $$cpu_out || $$(cat $$gpu_out | wc -c) -ne $(OUT_SIZE) || $$(cat $$cpu_out | wc -c) -ne $(OUT_SIZE) ]]; do \
			echo "wait for $$gpu_out and $$cpu_out" ;                                                                                              \
			sleep 2 ;                                                                                                                              \
		done;                                                                                                                                      \
		delta=$$(python3 $(TOOLDIR)/floats_delta.py $$gpu_out $$cpu_out) ;                                                                         \
		printf "$$iter;$$delta\n" >> $(OUTDIR)/deltas.csv ;                                                                                        \
		rm $$gpu_out $$cpu_out ;                                                                                                                   \
	done

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

.PHONY: all run images deltas output clean rebuild