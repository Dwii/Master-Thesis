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
LIBDIR=../libs

# Binary default arguments
OUTPRE=lbm_

DEFAULT_IMG_INTERVAL=100
DEFAULT_OUT_INTERVAL=0

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.c=.o)))

# Flags
CFLAGS=-g -Wall -Wextra -pedantic -std=gnu99 -O3

# Non-standard libraries
LIBNAM=pgm array
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

run: images

images: all $(OUTDIR)
ifeq ($(strip $(INTERVAL)),)
	$(eval INTERVAL=$(DEFAULT_IMG_INTERVAL))
endif
	$(BINDIR)/$(GPUBIN) -p -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) 

output: all $(OUTDIR) 
ifeq ($(strip $(INTERVAL)),)
	$(eval INTERVAL=$(DEFAULT_OUT_INTERVAL))
endif
	$(call timing, "$(BINDIR)/$(BIN)  -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE)", $(OUTDIR)/$(OUTPRE)$(ITER).timing)

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