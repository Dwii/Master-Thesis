# Compiler
CC=gcc
NVCC=nvcc

# Binary target
BIN=lbmFlowAroundCylinder

# Directories
SRCDIR=.
OBJDIR=obj
BINDIR=bin

# Source and object files
SRCS=$(wildcard $(SRCDIR)/*.cu)
OBJS=$(addprefix $(OBJDIR)/,$(SRCS:.cu=.o))

# Flags
NVCC_FLAGS=-arch=sm_30

all: $(OBJDIR) $(BINDIR) $(BINDIR)/$(BIN)

run: $(IMGDIR) all 
	$(BINDIR)/$(BIN) $(IMGDIR)

$(OBJDIR) $(BINDIR) $(IMGDIR): 
	mkdir -p $@

$(BINDIR)/$(BIN): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

-include $(OBJS:.o=.d)

$(OBJDIR)/%.o : %.cu 
	$(NVCC) -M $(NVCC_FLAGS) $< > $(@:.o=.d)
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(IMGDIR)

rebuild: clean all

.PHONY: all run clean rebuild