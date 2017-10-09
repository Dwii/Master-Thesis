include ../../lbm_common.mk

# Compiler
CC=gcc

# Binary target
BIN=lbmFlowAroundCylinder

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out
LIBDIR=../../Libraries

# Binary default arguments
OUTPRE=lbm_

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.c=.o)))

# Flags
CFLAGS=-g -Wall -Wextra -pedantic -std=gnu99 -O3

# Non-standard libraries
LIBNAM=pgm array timing
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

all: libraries $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

libraries: 
	@$(foreach library, $(LIBDEP), $(MAKE) -C $(library) | grep -v 'Nothing to be done' || $$true;)

# Targets addition arguments
images: TRGTARGS=-p
output: TRGTARGS=-f -L

# Targets defaut intervals
run output: INTERVAL?=0
images: INTERVAL?=100

# Targets info file path (left unset if not required)
output: INFO=$(OUTDIR)/$(OUTPRE)$(ITER).info

run images output: all $(OUTDIR)
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) $(TRGTARGS) -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) $(BINARGS)", $(INFO))

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

.PHONY: all run images output clean rebuild