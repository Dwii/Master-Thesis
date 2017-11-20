include ../../lbm_common.mk

# Compiler
CC=gcc

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out
LIBDIR=../../Libraries
BENDIR=benchmarks

# Binary default arguments
OUTPRE=lbm_
NX=100
NY=100
NZ=10

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.c=.o)))

# Repeat transfert (send and receive)
RT?=1
# Repeat send (write)
RS?=$(RT)
# Repeat receive (read)
RR?=$(RT)
# Repeat collide and stream
RC?=1

# Binaries sufix (mpi/sec, repeat, ...)
RSSUF =$(subst 1,s$(RS),$(subst $(RS),,1))
RRSUF =$(subst 1,r$(RR),$(subst $(RR),,1))
RCSUF =$(subst 1,c$(RC),$(subst $(RC),,1))
REPSUF=$(RSSUF)$(RRSUF)$(RCSUF)
BINSUF=$(if $(REPSUF),_repeat-$(REPSUF))

# Binary target
BIN=lbmSimple$(BINSUF)

# Flags
CFLAGS=-g -Wall -Wextra -pedantic -std=gnu99 -O3
CFLAGS += -DREPEAT_WRITE=$(RS) -DREPEAT_READ=$(RR) -DREPEAT_CNS=$(RC)

# Non-standard libraries and headers
LIBNAM=pgm timing lbmcuda
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

# Benchmarks
BENPRE=$(OUTPRE)$(BINSUF)
BENCHMARKS=$(wildcard $(BENDIR)/*.ben)
BENCHDATAS=${subst $(BENDIR)/,$(OUTDIR)/$(BENPRE),$(BENCHMARKS:.ben=.dat)}
# Measures per benchmark setup
MPS=2
# Measure extention
MEXT=out

all: libraries $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

libraries: 
	@$(foreach library, $(LIBDEP), $(MAKE) -C $(library) | grep -v 'Nothing to be done' || $$true;)

# Targets addition arguments
images: TRGTARGS=-p
output: TRGTARGS=-f -L
outpal: TRGTARGS=-F

# Targets defaut intervals
run output outpal: INTERVAL?=0
images: INTERVAL?=100

# Targets timing file path (left unset if not required)
output: INFO=$(OUTDIR)/$(OUTPRE)$(ITER).info

run images output outpal: all $(OUTDIR)
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) $(TRGTARGS) -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -x $(NX) -y $(NY) -z $(NZ) $(BINARGS)", $(INFO))

$(OBJDIR) $(BINDIR) $(OUTDIR): 
	mkdir -p $@

$(BINDIR)/$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ -Bdynamic $(LIBS)

-include $(OBJS:.o=.d)

$(OBJS): $(SRCS)
	$(CC) -MMD $(CFLAGS) -o $@ -c $< $(INCS)

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR)

rebuild: clean all

# Function:    measure
# Description: execute and store a measure (lups and energy) 
# Arguments:   NXZY, measure name
define measure
	$(eval nxyz=$(shell printf ' -x %d -y %d -z %d' $(1)))
	$(eval measure_file=$(2).$(MEXT))
	@printf "Generate measure $(measure_file): ";
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) -i $(ITER) -I 0 -L $(nxyz) $(BINARGS) | grep lups", $(measure_file))
	@cat $(measure_file) | grep -q lups || (echo "no lups" && exit 1)
	@echo "Done";
endef

# Function:    measure
# Description: execute measure function X times
# Arguments:   NXZY separated with spaces, measures name, X
define repeat_measure
	$(eval I=$(shell echo $$(seq $(3))))
	$(foreach i, $(I), $(call measure, $(1), $(2)_$(i)))
endef

# Function:    run_benchmark
# Description: execute benchmark measurements X times
# Arguments:   benchmark file, measures name, X
define run_benchmark
	$(eval I=$(shell seq $$(cat $(1) | wc -l)))
	@$(foreach i, $(I), $(call repeat_measure, $$(sed '$(i)q;d' $(1)), $(2)_$(i), $(3))) 
endef

benchmarks: INTERVAL=0
benchmarks: all $(OUTDIR) $(BENCHDATAS)

$(OUTDIR)/$(BENPRE)%.dat: $(BENDIR)/%.ben
	$(eval benchmark_name=$(@:.dat=))
	$(call run_benchmark, $<, $(benchmark_name), $(MPS))
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

.PHONY: all run images output clean rebuild
