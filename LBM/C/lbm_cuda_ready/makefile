# Compiler
CC=gcc

# Binary target
BIN=lbmFlowAroundCylinder

# Binary arguments
ITER=200000

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
OUTDIR=out
LIBDIR=../libs

# Test output file
TST=$(shell basename "$(abspath .)").out

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.c=.o)))

# Flags, libraries and includes
CFLAGS=-g -Wall -Wextra -pedantic -std=gnu99
LIBNAM=pgm array
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))


all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

run: all $(OUTDIR)
	$(BINDIR)/$(BIN) -p $(OUTDIR) -i $(ITER)

test: all $(OUTDIR) 
	$(BINDIR)/$(BIN) -f > $(OUTDIR)/$(TST) -i $(ITER)

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

.PHONY: all run test clean rebuild