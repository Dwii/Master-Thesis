# Compiler
NVCC=nvcc

# Binary target
BIN=liblbmcuda.so

# Directories
OBJDIR=obj

# Source and object files
SRCS=$(wildcard *.cu)
OBJS=$(addprefix $(OBJDIR)/,$(SRCS:.cu=.o))

# Flags
NVCC_FLAGS=--compiler-options '-fPIC' -Wno-deprecated-gpu-targets -O3 --ptxas-options=-v -Xptxas -dlcm=cg

all: $(OBJDIR) $(BIN)

$(OBJDIR) $(BINDIR) $(IMGDIR): 
	mkdir -p $@

$(BIN): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -shared $^ -o $@

-include $(OBJS:.o=.d)

$(OBJS) : $(SRCS)
	$(NVCC) -M $(NVCC_FLAGS) $< -odir $(OBJDIR) > $(@:.o=.d)
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $< 

clean:
	rm -rf $(OBJDIR) $(BIN)

rebuild: clean all

.PHONY: all clean rebuild