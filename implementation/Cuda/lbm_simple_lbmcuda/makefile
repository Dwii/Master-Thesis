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

.PHONY: all run images output clean rebuild
