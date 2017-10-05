# Compiler
CC=gcc -fPIC
AR=ar

# Binary target
BIN=libtiming.so

# Directories
OBJDIR=obj

# Source and object files
SRCS=$(wildcard *.c)
OBJS=$(addprefix $(OBJDIR)/,$(SRCS:.c=.o))

# Flags
CFLAGS=-g -Wall -Wextra -pedantic -std=gnu99 -O3

# Standard libraries
LIBS+=$(if $(filter $(shell uname),Darwin),,-lrt)

all: $(OBJDIR) $(BIN)

$(OBJDIR) $(BINDIR) $(IMGDIR): 
	mkdir -p $@

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -fPIC -shared $^ -o $@ $(LIBS)

-include $(OBJS:.o=.d)

$(OBJDIR)/%.o : %.c
	$(CC) -MMD $(CFLAGS) -o $@ -c $<

clean:
	rm -rf $(OBJDIR) $(BIN)

rebuild: clean all

.PHONY: all clean rebuild