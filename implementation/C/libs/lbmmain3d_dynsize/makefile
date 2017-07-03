# Compiler
CC=gcc

# Binary target
BIN=liblbmmain3d_dynsize.so

# Directories
SRCDIR=.
OBJDIR=obj
OUTDIR=out
LIBDIR=..

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.c)
OBJS=$(addprefix $(OBJDIR)/,$(notdir $(SRCS:.c=.o)))

# Flags
CFLAGS=-g -Wall -Wextra -pedantic -O3

# Non-standard libraries
LIBNAM=pgm array timing
LIBDEP=$(addprefix $(LIBDIR)/,$(LIBNAM))
INCS=$(addprefix -I,$(LIBDEP))
LIBS=$(addprefix -L,$(LIBDEP)) $(addprefix -l,$(LIBNAM))

# Standard libraries
LIBS+=-lm

all: $(OBJDIR) $(BIN)

$(OBJDIR): 
	mkdir -p $@

-include $(OBJS:.o=.d)
	
$(OBJS): $(SRCS)
	$(CC) -MMD $(CFLAGS) -fPIC -o $@ -c $< $(INCS)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -fPIC -shared $^ -o $@ $(LIBS) $(INCS) -Wl,-undefined,dynamic_lookup

clean:
	rm -rf $(OBJDIR) $(BIN)

rebuild: clean all

.PHONY: all clean rebuild
