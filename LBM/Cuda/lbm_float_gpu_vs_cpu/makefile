# Compiler
CC=gcc
NVCC=nvcc

# Binary target
GPUBIN=lbmFlowAroundCylinder_gpu
CPUBIN=lbmFlowAroundCylinder_cpu
BINS=

# Binary arguments
ITER=200000

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out

# Test output file
OUT=lbm.out

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
GPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_gpu.o)))
CPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_cpu.o)))
OBJS= $(GPUOBJS) $(CPUOBJS)
# Flags
NVCC_FLAGS=-Wno-deprecated-gpu-targets

# Libraries
LIBS=-lm

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN)

run: all 
	$(BINDIR)/$(BIN) -i $(ITER)

output: all $(OUTDIR) 
	$(BINDIR)/$(GPUBIN) -i $(ITER) -f > $(OUTDIR)/gpu_$(OUT)
	$(BINDIR)/$(CPUBIN) -i $(ITER) -f > $(OUTDIR)/cpu_$(OUT)

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
	$(NVCC) -M $(NVCC_FLAGS) $(NVCC_XPU_FLAG) $< -odir $(OBJDIR) > $(@:.o=.d)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_XPU_FLAG) -o $@ -c $<

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR)

rebuild: clean all

.PHONY: all run output clean rebuild