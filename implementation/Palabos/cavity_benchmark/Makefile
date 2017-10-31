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

# Repeat transfert (send and receive)
RT?=1
# Repeat send
RS?=$(RT)
# Repeat receive
RR?=$(RT)
# Repeat collide and stream
RC?=1
# Repeat collide and stream from Palabos
RPC?=1

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
libraryPaths = ../../Libraries/timing
# Path to inlude directories (other than Palabos)
includePaths = ../../Libraries/timing
# Dynamic and static libraries (other than Palabos)
libraries    = -ltiming $(if $(filter $(shell uname),Linux),-lrt)

# Compiler to use without MPI parallelism
serialCXX    = g++
# Compiler to use with MPI parallelism
parallelCXX  = mpicxx
# General compiler flags (e.g. -Wall to turn on all warnings on g++)
compileFlags = -Wall -Wnon-virtual-dtor -DPLB_MAC_OS_X -Wno-deprecated -DREPEAT_SEND=$(RS) -DREPEAT_RECEIVE=$(RR) -DREPEAT_CNS=$(RC) -DREPEAT_PALABOS_CNS=$(RPC)
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
BENDIR=benchmarks

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

# Benchmarks
BENPRE=$(OUTPRE)$(XPU)_$(subst true,,$(subst $(BOUNDARY),boundary_,true))
BENCHMARKS=$(wildcard $(BENDIR)/*.ben)
BENCHDATAS=${subst $(BENDIR)/,$(OUTDIR)/$(BENPRE),$(BENCHMARKS:.ben=.dat)}
# Should measures energies be checked or not (true/false)
CHECK_NRG=true
# Measures per benchmark setup
MPS=2
# Measure extention
MEXT=out
PLOTSCRIPT=plot_benchmark.py

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

LBMCUDA_PATH=../../Libraries/lbmcuda 

compile: $(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN)

$(BINDIR)/$(CPUBIN): compileFlags += -DPLB_NO_CUDA
$(BINDIR)/$(GPUBIN): libraries += -llbmcuda
$(BINDIR)/$(GPUBIN): libraryPaths += $(LBMCUDA_PATH)
$(BINDIR)/$(GPUBIN): includePaths += $(LBMCUDA_PATH)
$(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN): $(projectFiles) | $(BINDIR)
	python $(SCons) $(SConsArgs) && mv $(OUTBIN) $@ 

rebuild: clean_bin compile

clean: clean_bin
	python $(SCons) -c $(SConsArgs)
	/bin/rm -vf `find $(palabosRoot) -name '*~'`
	/bin/rm -rf $(OUTDIR)

clean_bin:
	/bin/rm -rf $(BINDIR)
	/bin/rm -f *.o

$(OUTDIR) $(BINDIR):
	@mkdir -p $@

setbin:
	$(eval BIN=$(if $(subst $(strip $(XPU)),,cpu),$(BIN),$(BINDIR)/$(CPUBIN)))
	$(eval BIN=$(if $(subst $(strip $(XPU)),,gpu),$(BIN),$(BINDIR)/$(GPUBIN)))
	$(eval libraryPaths+=$(if $(subst $(strip $(XPU)),,gpu),,$(LBMCUDA_PATH)))
	$(if $(strip $(BIN)),,echo "Unset binary..."; exit 1)

run: BINARGS?=-p -e

info: INFO=$(OUTDIR)/$(OUTPRE)$(ITER).info
info: BINARGS?=-e

info run: compile setbin .stdout $(OUTDIR)
	@echo $(XPU) libs: $(libraryPaths)
	$(call execute, $(libraryPaths), "$(BIN) -N $(N) -x $(SNX) -y $(SNY) -z $(SNZ) -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -L $(BINARGS) $(BOUNDARY_FLAG)", $(INFO))


# Function:    measure
# Description: execute and store a measure (lups and energy) 
# Arguments:   NXZY, measure name, expected energy (optional)
define measure
	$(eval nxyz=$(shell printf ' -N %d -x %d -y %d -z %d' $(1)))
	$(eval measure_file=$(2).$(MEXT))
	$(call execute, $(libraryPaths), "$(BIN) -e -L $(BOUNDARY_FLAG) $(nxyz) -i $(ITER) -I 0 2> /dev/null | grep -E 'energy|lups'", $(measure_file))
	$(if $(strip $(3)),
		@cat $(measure_file) | sed -n 's/.*energy: \(.*\).*/\1/p' | grep $(3) > /dev/null || (echo "energy differ for $(measure_file)" && exit 1)
	)
endef

# Function:    measure
# Description: execute measure function X times
# Arguments:   NXZY separated with spaces, measures name, X, expected energy (optional)
define repeat_measure
	$(eval I=$(shell echo $$(seq $(3))))
	$(foreach i, $(I), $(call measure, $(1), $(2)_$(i), $(4)))
endef

# Function:    run_benchmark
# Description: execute benchmark measurements X times
# Arguments:   benchmark file, measures name, X, energy file (optional)
define run_benchmark
	$(eval I=$(shell seq $$(cat $(1) | wc -l)))
	@$(foreach i, $(I), $(call repeat_measure, $$(sed '$(i)q;d' $(1)), $(2)_$(i), $(3), $(if $(strip $(4)), $$(sed '$(i)q;d' $(4))))) 
endef

benchmark: INTERVAL=0
benchmark: setbin compile $(OUTDIR) $(BENCHDATAS)

.PRECIOUS: $(BENDIR)/%.nrg

$(BENDIR)/%.nrg: BIN=$(BINDIR)/$(CPUBIN)
$(BENDIR)/%.nrg: $(BENDIR)/%.ben 
	$(if $(subst true,,$(CHECK_NRG)),, \
		$(eval benchmark_name=${subst $(BENDIR),$(OUTDIR),$(<:.ben=)}_nrg) \
		$(call run_benchmark, $<, $(benchmark_name), 1) \
		@expected_count=$$(cat $< | wc -l); \
		nrg_count=$$(cat $(benchmark_name)*_1.out | grep energy | wc -l); \
		test $$nrg_count -eq $$expected_count || ( \
			printf "Wrong number of energies: %d/%d\n" $$nrg_count $$expected_count; \
			exit 1 \
		) \
		@cat $(benchmark_name)*_1.out | sed -n 's/.*energy: \(.*\).*/\1/p' > $@ \
	)

$(OUTDIR)/$(BENPRE)%.dat: $(BENDIR)/%.ben $(BENDIR)/%.nrg
	$(eval benchmark_name=$(@:.dat=))
	$(eval nrg_file=$(^:.ben=.nrg))
	$(call run_benchmark, $<, $(benchmark_name), $(MPS), $(subst true,,$(subst $(CHECK_NRG),$(nrg_file),true)))
	$(eval NI=$(shell cat $< | wc -l))
	$(eval I=$(shell seq $(NI)))

	@expected_count=$$(echo '$(NI) * $(MPS)' | bc ); \
	output_count=$$(grep -l -E "lups|energy" $(benchmark_name)*.out | wc -l); \
	test $$output_count -eq $$expected_count || ( \
		printf "Wrong number of valid output: %d/%d\n" $$output_count $$expected_count; \
		exit 1 \
	)

	@$(foreach i, $(I), echo "($$(cat $(benchmark_name)_$(i)*.$(MEXT) | sed -n 's/average lups: \(.*\).*/\1/p' | paste -s -d+ -))" / $(MPS) | bc >> $@; )


define get_dat_legend
	$(eval xpu=$(shell echo $(1) | grep -q gpu && echo "GPU" || echo "CPU")) \
	$(eval transfert=$(shell echo $(1) | grep -q boundary && echo "boundaries" || echo "full domain")) \
	'$(xpu) ($(transfert))'
endef

benchplot: 
	@$(foreach benchmark, $(BENCHMARKS), \
		$(eval image=${subst $(BENDIR)/,$(OUTDIR)/,$(benchmark:.ben=.pdf)}) \
		$(eval dats=$(shell echo ${subst $(BENDIR)/,$(OUTDIR)/*,$(benchmark:.ben=.dat)})) \
		python3 $(PLOTSCRIPT) $(benchmark) $(image) $(foreach dat, $(dats), $(dat) $(call get_dat_legend, $(dat)) ); \
	)	

.PHONY: compile rebuild clean clean_bin set_bin run benchmark