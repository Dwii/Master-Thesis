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
LIBDIR=../../Libraries

# Binary default arguments
OUTPRE=lbm_
NX=100
NY=100
NZ=10

DEFAULT_IMG_INTERVAL=100
DEFAULT_OUT_INTERVAL=0

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.c=.o)))

# Flags
CFLAGS=-g -Wall -Wextra -pedantic -O3

# Non-standard libraries and headers
LIBNAM=pgm timing lbmcuda
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

all: libraries $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

libraries: 
	@$(foreach library, $(LIBDEP), $(MAKE) -C $(library) | grep -v 'Nothing to be done' || $$true;)

run: images

images: all $(OUTDIR)
	$(eval INTERVAL ?= $(DEFAULT_IMG_INTERVAL))
	echo "$(BINDIR)/$(BIN) -p -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -x $(NX) -y $(NY) -z $(NZ)"
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) -p -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -x $(NX) -y $(NY) -z $(NZ)" )

output: all $(OUTDIR) 
	$(eval INTERVAL ?= $(DEFAULT_OUT_INTERVAL))
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) -f -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -L -x $(NX) -y $(NY) -z $(NZ)", $(OUTDIR)/$(OUTPRE)$(ITER).timing)

outpal: all $(OUTDIR) 
	$(eval INTERVAL ?= $(DEFAULT_OUT_INTERVAL))
	$(call execute, $(LIBDEP), "$(BINDIR)/$(BIN) -F -i $(ITER) -I $(INTERVAL) -o $(OUTDIR) -O $(OUTPRE) -x $(NX) -y $(NY) -z $(NZ)")

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
