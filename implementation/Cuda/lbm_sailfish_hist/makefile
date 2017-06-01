include ../../lbm_common.mk
include ../lbm_gpu_vs_cpu_common.mk

# Compiler
CXX=g++
NVCC=nvcc

# Binary target
GPUBIN=lbm_gpu
CPUBIN=lbm_cpu

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

# Non-standard libraries
LIBNAM=pgm timing
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

CXXFLAGS=-O2

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN)

images: all $(OUTDIR)
	$(call execute, $(LIBDEP), "$(BINDIR)/$(GPUBIN) $(ITER) $(OUTDIR) $(GPUPRE)")

output: all $(OUTDIR) outcpu outgpu

outcpu: 
	$(call execute, $(LIBDEP), "$(BINDIR)/$(CPUBIN) $(ITER)", $(OUTDIR)/$(CPUPRE)$(ITER).timing)

outgpu: 
	$(call execute, $(LIBDEP), "$(BINDIR)/$(GPUBIN) $(ITER)", $(OUTDIR)/$(GPUPRE)$(ITER).timing)

$(OBJDIR)/%.o: %.cpp $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INCS)

$(BINDIR)/$(CPUBIN): lbm.cpp $(OBJDIR)/sim.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm $(INCS) $(LIBS)

$(BINDIR)/$(GPUBIN): lbm.cu $(OBJDIR)/sim.o 
	$(NVCC) $^ --use_fast_math -o $@ $(INCS) $(LIBS)

$(OBJDIR) $(BINDIR) $(OUTDIR): 
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR)
