# Compiler
CC=gcc -fPIC
AR=ar

# Binary target
BIN=libarray.a

# Directories
OBJDIR=obj

# Source and object files
SRCS=$(wildcard *.c)
OBJS=$(addprefix $(OBJDIR)/,$(SRCS:.c=.o))

# Flags
CFLAGS=-g -Wall -Wextra -pedantic -std=gnu99

all: $(OBJDIR) $(BIN)

$(OBJDIR) $(BINDIR) $(IMGDIR): 
	mkdir -p $@

$(BIN): $(OBJS)
	$(AR) rcs $@ $^ 

-include $(OBJS:.o=.d)

$(OBJDIR)/%.o : %.c
	echo $(OBJS:.o=.d)
	$(CC) -MMD $(CFLAGS) -o $@ -c $< $(INCS)

clean:
	rm -rf $(OBJDIR) $(BIN)

rebuild: clean all

.PHONY: all clean rebuild