# Compiler
CC=gcc -fPIC
AR=ar

# Binary target
BIN=libarray.so

# Directories
OBJDIR=obj

# Source and object files
SRCS=$(wildcard *.c)
OBJS=$(addprefix $(OBJDIR)/,$(SRCS:.c=.o))

# Flags
CFLAGS=-g -Wall -Wextra -pedantic -std=gnu99 -O3

all: $(OBJDIR) $(BIN)

$(OBJDIR) $(BINDIR) $(IMGDIR): 
	mkdir -p $@

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -fPIC -shared $^ -o $@ $(LIBS) $(INCS)

-include $(OBJS:.o=.d)

$(OBJDIR)/%.o : %.c
	$(CC) -MMD $(CFLAGS) -o $@ -c $< $(INCS)

clean:
	rm -rf $(OBJDIR) $(BIN)

rebuild: clean all

.PHONY: all clean rebuild
