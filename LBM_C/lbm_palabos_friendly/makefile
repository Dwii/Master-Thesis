# Compiler
CC=gcc

# Binary target
BIN=lbmFlowAroundCylinder

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin
IMGDIR=images
LIBDIR=../libs

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(addprefix $(OBJDIR)/,$(SRCS:.c=.o))

# Flags, libraries and includes
CFLAGS=-g -Wall -Wextra -pedantic -std=gnu99
LIBNAM=pgm array
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))


all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

run: $(IMGDIR) all 
	$(BINDIR)/$(BIN) $(IMGDIR)

$(OBJDIR) $(BINDIR) $(IMGDIR): 
	mkdir -p $@

$(BINDIR)/$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ -Bdynamic $(LIBS)

-include $(OBJS:.o=.d)

$(OBJDIR)/%.o : %.c
	echo $(OBJS:.o=.d)
	$(CC) -MMD $(CFLAGS) -o $@ -c $< $(INCS)

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(IMGDIR)

rebuild: clean all

.PHONY: all run clean rebuild