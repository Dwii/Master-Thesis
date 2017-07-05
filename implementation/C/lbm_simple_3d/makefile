include ../../lbm_common.mk

# Compiler
CC=gcc

# Binary target
BIN=lbmSimple

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out
LIBDIR=../../C/libs

# Binary default arguments
OUTPRE=lbm_

DEFAULT_IMG_INTERVAL=100
DEFAULT_OUT_INTERVAL=0

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.c=.o)))

# Flags
CFLAGS=--std=gnu99 -O3

# Non-standard libraries
LIBNAM=pgm timing lbmmain3d
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

run: images

images: all $(OUTDIR)
	$(eval INTERVAL ?= $(DEFAULT_IMG_INTERVAL))
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) -p -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) $(BINARGS)")

output: all $(OUTDIR) 
	$(eval INTERVAL ?= $(DEFAULT_OUT_INTERVAL))
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -L $(BINARGS)", $(OUTDIR)/$(OUTPRE)$(ITER).timing)

$(OBJDIR) $(BINDIR) $(OUTDIR): 
	mkdir -p $@

$(BINDIR)/$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

-include $(OBJS:.o=.d)

$(OBJS): $(SRCS)
	$(CC) -MMD $(CFLAGS) -o $@ -c $< $(INCS)

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR)

rebuild: clean all

.PHONY: all run images output clean rebuild 
