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
# Repeat duplicate overlaps for Palabos
RPD?=1

# Number of process for mpirun (> 1 imply MPIparallel = true)
NP=1
MPICMD=$(subst 1,mpirun -np $(NP),$(subst $(NP),,1))

# External libraries path to compile with their makefiles
LBMCUDA_PATH=../../Libraries/lbmcuda 
TIMING_PATH=../../Libraries/timing
LIBDEP=$(LBMCUDA_PATH) $(TIMING_PATH)

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
MPIparallel  = $(subst 1,true,$(subst $(NP),false,1))
# Set SMP-parallel mode on/off (shared-memory parallelism)
SMPparallel  = false
# Decide whether to include calls to the POSIX API. On non-POSIX systems,
#   including Windows, this flag must be false, unless a POSIX environment is
#   emulated (such as with Cygwin).
usePOSIX     = true

# Path to external source files (other than Palabos)
srcPaths =
# Path to external libraries (other than Palabos)
libraryPaths = $(TIMING_PATH)
# Path to inlude directories (other than Palabos)
includePaths = $(TIMING_PATH)
# Dynamic and static libraries (other than Palabos)
libraries    = -ltiming $(if $(filter $(shell uname),Linux),-lrt)

# Compiler to use without MPI parallelism
serialCXX    = g++
# Compiler to use with MPI parallelism
parallelCXX  = mpicxx
# General compiler flags (e.g. -Wall to turn on all warnings on g++)
compileFlags = -Wall -Wnon-virtual-dtor -Wno-deprecated 
# Add compatibility flag for on Darwin (Mac) systems
compileFlags += $(if $(filter $(shell uname),Darwin),-DPLB_MAC_OS_X)
# Set how many times certain parts of the code should be repeated 
compileFlags += -DREPEAT_SEND=$(RS) -DREPEAT_RECEIVE=$(RR) -DREPEAT_CNS=$(RC) -DREPEAT_PALABOS_CNS=$(RPC) -DREPEAT_PALABOS_DO=$(RPD)
#compileFlags += -Wno-deprecated-gpu-targets --ptxas-options=-v
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
# Binaries sufix (mpi/sec, repeat, ...)
SEQSUF=$(subst true,seq,$(subst $(MPIparallel),mpi,true))
RSSUF =$(subst 1,s$(RS),$(subst $(RS),,1))
RRSUF =$(subst 1,r$(RR),$(subst $(RR),,1))
RCSUF =$(subst 1,c$(RC),$(subst $(RC),,1))
RPCSUF=$(subst 1,pc$(RPC),$(subst $(RPC),,1))
RPDSUF=$(subst 1,pd$(RPD),$(subst $(RPD),,1))
REPSUF=$(RSSUF)$(RRSUF)$(RCSUF)$(RPCSUF)$(RPDSUF)
BINSUF=$(SEQSUF)$(if $(REPSUF),_repeat-$(REPSUF))
# Final binaries names
CPUBIN=$(OUTBIN)_cpu_$(BINSUF)
GPUBIN=$(OUTBIN)_gpu_$(BINSUF)
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
BENPRE=$(OUTPRE)$(XPU)_$(BINSUF)_$(subst true,,$(subst $(BOUNDARY),boundary_,true))
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

compile: libraries $(BINDIR)/$(GPUBIN) $(BINDIR)/$(CPUBIN)

libraries: 
	@$(foreach library, $(LIBDEP), $(MAKE) -C $(library) | grep -v 'Nothing to be done for $(library)' || $$true;)

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
	$(call execute, $(libraryPaths), "$(MPICMD) $(BIN) -N $(N) -x $(SNX) -y $(SNY) -z $(SNZ) -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -L $(BINARGS) $(BOUNDARY_FLAG)", $(INFO))


# Function:    measure
# Description: execute and store a measure (lups and energy) 
# Arguments:   NXZY, measure name, expected energy (optional)
define measure
	$(eval nxyz=$(shell printf ' -N %d -x %d -y %d -z %d' $(1)))
	$(eval measure_file=$(2).$(MEXT))
	@printf "Generate measure $(measure_file): ";
	$(call execute, $(libraryPaths), "$(MPICMD) $(BIN) -e -L $(BOUNDARY_FLAG) $(nxyz) -i $(ITER) -I 0 | grep -E 'energy|lups'", $(measure_file))
	$(if $(strip $(3)),
		@cat $(measure_file) | sed -n 's/.*energy: \(.*\).*/\1/p' | grep $(3) > /dev/null || (echo "energy differ for $(measure_file)" && exit 1)
	)
	@cat $(measure_file) | grep -q lups || (echo "no lups" && exit 1)
	@echo "Done";
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

benchmarks: INTERVAL=0
benchmarks: setbin compile $(OUTDIR) $(BENCHDATAS)

.PRECIOUS: $(BENDIR)/%.nrg

$(BENDIR)/%.nrg: BIN=$(BINDIR)/$(CPUBIN)
$(BENDIR)/%.nrg: $(BENDIR)/%.ben 
	$(eval expected_count=$(shell cat $< | wc -l))
	$(if $(subst true,,$(CHECK_NRG)),, \
		$(eval benchmark_name=${subst $(BENDIR),$(OUTDIR),$(<:.ben=)}_nrg) \
		$(call run_benchmark, $<, $(benchmark_name), 1) \
		printf "Generate nrg $@: "; \
		nrg_count=$$(cat $(benchmark_name)*_1.out | grep energy | wc -l); \
		test $$nrg_count -eq $(expected_count) || ( \
			printf "Wrong number of energies: %d/%d\n" $$nrg_count $$expected_count; \
			exit 1 \
		); \
		cat $(benchmark_name)*_1.out | sed -n 's/.*energy: \(.*\).*/\1/p' > $@; \
		echo "Done" \
	)

$(OUTDIR)/$(BENPRE)%.dat: $(BENDIR)/%.ben $(BENDIR)/%.nrg
	$(eval benchmark_name=$(@:.dat=))
	$(eval nrg_file=$(^:.ben=.nrg))
	$(call run_benchmark, $<, $(benchmark_name), $(MPS), $(subst true,,$(subst $(CHECK_NRG),$(nrg_file),true)))
	$(eval NI=$(shell cat $< | wc -l))
	$(eval I=$(shell seq $(NI))) 
	$(eval J=$(shell seq $(MPS))) 
	@printf "Generate $@: "
	@($(foreach i, $(I), \
		function join_by { local IFS="$$1"; shift; echo "$$*"; }; \
		all_lups=$$(echo $$($(foreach j, $(J), cat $(benchmark_name)_$(i)_$(j).$(MEXT) | sed -n 's/average lups: \(.*\).*/\1/p' | sort -g | head -n 1; ))); \
		nbr_lups=$$(echo $$all_lups | wc -w); \
		sum_lups=$$(echo $$(join_by + $$all_lups) | bc); \
		avg_lups=$$(echo $$sum_lups / $(MPS) | bc); \
		echo $$avg_lups; \
	)) > $@
	@echo "Done"

define get_dat_legend
	$(eval xpu=$(shell echo $(1) | grep -q gpu && echo "GPU" || echo "CPU")) \
	$(eval transfert=$(shell echo $(1) | grep -q boundary && echo "partiel" || echo "complet")) \
	$(eval mpi=$(shell echo $(1) | grep -q mpi && echo "with MPI ")) \
	'$(xpu) $(mpi)($(transfert))'
endef

benchplot: 
	@$(foreach benchmark, $(BENCHMARKS), \
		$(eval image=${subst $(BENDIR)/,$(OUTDIR)/$(OUTPRE),$(benchmark:.ben=.pdf)}) \
		$(eval dats=$(shell ls ${subst $(BENDIR)/,$(OUTDIR)/*,$(benchmark:.ben=.dat)} | grep -v repeat)) \
		python3 $(PLOTSCRIPT) $(benchmark) $(image) $(foreach dat, $(dats), $(dat) $(call get_dat_legend, $(dat)) ); \
	)	

.PHONY: compile rebuild clean clean_bin set_bin run benchmark