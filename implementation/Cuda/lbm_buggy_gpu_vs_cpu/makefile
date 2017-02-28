# Compiler
NVCC=nvcc

# Binary target
GPUBIN=lbmFlowAroundCylinder_gpu
CPUBIN=lbmFlowAroundCylinder_cpu

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin

# Test output file
OUT=lbm.out
ITER=2

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
GPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_gpu.o)))
CPUOBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=_cpu.o)))
OBJS= $(GPUOBJS) $(CPUOBJS)

# Flags
NVCC_FLAGS=-Wno-deprecated-gpu-targets

# Standard libraries
LIBS=-lm

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN)

run: all
	$(BINDIR)/$(CPUBIN) -i $(ITER)
	$(BINDIR)/$(GPUBIN) -i $(ITER)

$(OBJDIR) $(BINDIR): 
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
	rm -rf $(OBJDIR) $(BINDIR)

rebuild: clean all

.PHONY: all run clean rebuild