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

# Files
SRC=$(wildcard $(SRCDIR)/*.c)
OBJ=$(addprefix $(OBJDIR)/,$(SRC:.c=.o))

# Flags, libraries and includes
CFLAGS=-g -Wall -Wextra -pedantic -std=gnu99
LIBNAM=pgm array
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INC=$(addprefix -I,$(LIBDEP))
LIB=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Rules

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

run: $(IMGDIR) all 
	$(BINDIR)/$(BIN) $(IMGDIR)

$(OBJDIR) $(BINDIR) $(IMGDIR): 
	mkdir -p $@

$(BINDIR)/$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $< -o $@ -Bdynamic $(LIB)

$(OBJDIR)/%.o : %.c
	$(CC) $(CFLAGS) -o $@ -c $< $(INC)

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(IMGDIR)

rebuild: clean all

.PHONY: all run clean rebuild