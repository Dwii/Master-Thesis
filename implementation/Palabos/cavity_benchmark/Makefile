include ../../lbm_common.mk

##########################################################################
## Makefile.
##
## The present Makefile is a pure configuration file, in which 
## you can select compilation options. Compilation dependencies
## are managed automatically through the Python library SConstruct.
##
## If you don't have Python, or if compilation doesn't work for other
## reasons, consult the Palabos user's guide for instructions on manual
## compilation.
##########################################################################

# USE: multiple arguments are separated by spaces.
#   For example: projectFiles = file1.cpp file2.cpp
#                optimFlags   = -O -finline-functions

# Leading directory of the Palabos source code
palabosRoot  = ../palabos-master
# Name of source files in current directory to compile and link with Palabos
projectFiles = cavity3d.cpp

# Set optimization flags on/off
optimize     = true
# Set debug mode and debug flags on/off
debug        = true
# Set profiling flags on/off
profile      = false
# Set MPI-parallel mode on/off (parallelism in cluster-like environment)
MPIparallel  = false
# Set SMP-parallel mode on/off (shared-memory parallelism)
SMPparallel  = false
# Decide whether to include calls to the POSIX API. On non-POSIX systems,
#   including Windows, this flag must be false, unless a POSIX environment is
#   emulated (such as with Cygwin).
usePOSIX     = true

# Path to external source files (other than Palabos)
srcPaths =
# Path to external libraries (other than Palabos)
libraryPaths = ../../Libraries/lbmcuda ../../Libraries/timing
# Path to inlude directories (other than Palabos)
includePaths = ../../Libraries/lbmcuda ../../Libraries/timing
# Dynamic and static libraries (other than Palabos)
libraries    = -llbmcuda -ltiming

# Compiler to use without MPI parallelism
serialCXX    = g++
# Compiler to use with MPI parallelism
parallelCXX  = mpicxx
# General compiler flags (e.g. -Wall to turn on all warnings on g++)
compileFlags = -Wall -Wnon-virtual-dtor -DPLB_MAC_OS_X -Wno-deprecated
#compileFlags = -DPLB_MAC_OS_X -Wno-deprecated-gpu-targets --ptxas-options=-v
# General linker flags (don't put library includes into this flag)
linkFlags    =
# Compiler flags to use when optimization mode is on
optimFlags   = -O3
# Compiler flags to use when debug mode is on
debugFlags   = -g
# Compiler flags to use when profile mode is on
profileFlags = -pg

# Binary filename output by SCons (first project file without its extension)
OUTBIN=$(shell outbin="$(projectFiles)"; echo $${outbin%%.*})
# Final binaries names
CPUBIN=$(OUTBIN)_cpu
GPUBIN=$(OUTBIN)_gpu
# Directories
OUTDIR=out
BINDIR=bin

# Binary default arguments
N = 30
SNX = 16
SNY = 16
SNZ = 16
INTERVAL=300
OUTPRE=lbm_
BOUNDARY=true
BOUNDARY_FLAG=$(subst true,,$(subst $(BOUNDARY),-b,true))
XPU=cpu

# Makfile stuff
D=$$
noop=
space=$(noop) $(noop)
quote='
define newline


endef

##########################################################################
# All code below this line is just about forwarding the options
# to SConstruct. It is recommended not to modify anything there.
##########################################################################

SCons     = $(palabosRoot)/scons/scons.py -j 6 -f $(palabosRoot)/SConstruct

define SConsArgsDefinition
palabosRoot=$(palabosRoot)
projectFiles="$(projectFiles)"
optimize=$(optimize)
debug=$(debug)
profile=$(profile)
MPIparallel=$(MPIparallel)
SMPparallel=$(SMPparallel)
usePOSIX=$(usePOSIX)
serialCXX=$(serialCXX)
parallelCXX=$(parallelCXX)
compileFlags="$(compileFlags)"
linkFlags="$(linkFlags)"
optimFlags="$(optimFlags)"
debugFlags="$(debugFlags)"
profileFlags="$(profileFlags)"
srcPaths="$(srcPaths)"
libraryPaths="$(libraryPaths)"
includePaths="$(includePaths)"
libraries="$(libraries)"
endef

define SConsArgs
	$(shell printf '$(subst $(newline),\n,${SConsArgsDefinition})')
endef

compile: $(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN)

$(BINDIR)/$(CPUBIN): compileFlags += -DPLB_NO_CUDA
$(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN): $(projectFiles) | $(BINDIR)
	python $(SCons) $(SConsArgs) && mv $(OUTBIN) $@ 

rebuild: clean compile

clean:
	python $(SCons) -c $(SConsArgs)
	/bin/rm -vf `find $(palabosRoot) -name '*~'`
	/bin/rm -rf $(OUTDIR)
	/bin/rm -rf $(BINDIR)
	/bin/rm -f *.o

$(OUTDIR) $(BINDIR):
	@mkdir -p $@

setbin:
	$(eval BIN=$(if $(subst $(strip $(XPU)),,cpu),$(BIN),$(BINDIR)/$(CPUBIN)))
	$(eval BIN=$(if $(subst $(strip $(XPU)),,gpu),$(BIN),$(BINDIR)/$(GPUBIN)))
	$(if $(strip $(BIN)),,echo "Unset binary..."; exit 1)

run: BINARGS?=-p -e
run: compile setbin .stdout $(OUTDIR)
	$(call execute, $(libraryPaths), "$(BIN) -N $(N) -x $(SNX) -y $(SNY) -z $(SNZ) -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -L $(BINARGS) $(BOUNDARY_FLAG)")

.PHONY: compile rebuild clean setbin run 