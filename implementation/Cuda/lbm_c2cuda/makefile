# Function:    timing
# Description: Measure the execution time of a command a put it in a file
# Argument $1: Command (in a string) to execute and time
# Argument $2: File where the timing should be stored
define timing
	@start=$$(date +%s); \
	eval $(1); \
	end=$$(date +%s); \
	T=$$(($$end-$$start)); \
	printf "%02d:%02d:%02d" "$$((T/3600%24))" "$$((T/60%60))" "$$((T%60))" > $(2);
endef

# Compiler
CC=gcc
NVCC=nvcc

# Binary target
BIN=lbmFlowAroundCylinder

# Binary arguments
ITER=200000

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out
LIBDIR=../../C/libs

# Test output file
OUT=lbm.out

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.cu=.o)))

# Flags
NVCC_FLAGS=-Wno-deprecated-gpu-targets

# Non-standard libraries
LIBNAM=pgm array
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)
	
run: all $(OUTDIR) 
	$(BINDIR)/$(BIN) -p $(OUTDIR) -i $(ITER)

output: all $(OUTDIR)
	$(call timing, "$(BINDIR)/$(BIN) -i $(ITER) -f > $(OUTDIR)/$(OUT)", $(OUTDIR)/$(OUT).timing)

$(OBJDIR) $(BINDIR) $(OUTDIR): 
	mkdir -p $@

$(BINDIR)/$(BIN): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

-include $(OBJS:.o=.d)

$(OBJS): $(SRCS)
	$(NVCC) -M $(NVCC_FLAGS) $< -odir $(OBJDIR) $(INCS) > $(@:.o=.d) 
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $< $(INCS)

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR)

rebuild: clean all

.PHONY: all run output clean rebuild