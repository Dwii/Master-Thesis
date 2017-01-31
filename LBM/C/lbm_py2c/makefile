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
TST=test.out

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.c=.o)))

# Flags
CFLAGS=-g -Wall -Wextra -pedantic -std=gnu99

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