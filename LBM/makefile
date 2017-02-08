# usage: $(MAKE) [ test [ ITER=<total iterations> ] [ TST=<test filename> ] ]

RED=\033[0;31m
GRN=\033[0;32m
NC=\033[0m 

# Function:    compare
# Description: Test that two files are identical and print the check result
# Argument $1: Test title 
# Argument $2: First file
# Argument $3: Second title 
define compare
	-@diff -q $(2) $(3) 2>&1 1>/dev/null; if [ $$? -eq 0 ] ; then echo "$(1) $(GRN)OK$(NC)" ; else  echo "$(1) $(RED)KO$(NC)"; fi
endef


LANGS=Python C Cuda

# Default variables values
DEFAULT_ITER=1
DEFAULT_TST=test.out

# Flags
MAKEFLAGS=--no-print-directory

all: $(LANGS)

vars: 
# Set default ITER if not set (ensure that all test run are equivalent)
ifeq ($(strip $(ITER)),)
	$(eval MAKEFLAGS+=ITER=$(DEFAULT_ITER))
endif
# Set test output file name (ensure that test diff path are correct)
ifeq ($(strip $(TST)),)
	$(eval TST=$(DEFAULT_TST))
	$(eval MAKEFLAGS+=TST=$(TST))
endif

test: vars all

	@$(foreach lang, $(LANGS), $(MAKE) -C $(lang) test $(MAKEFLAGS);)

	@printf "\n----------- Tests with $(ITER) iterations ----------\n"
	@$(call compare, Test lbm_py2c:                      , Python/lbm_py/out/$(TST),                 C/lbm_py2c/out/$(TST))
	@$(call compare, Test lbm_palabos_friendly:          , Python/lbm_palabos_friendly/out/$(TST),   C/lbm_palabos_friendly/out/$(TST))
	@$(call compare, Test lbm_cuda_ready:                , Python/lbm_palabos_friendly/out/$(TST),   C/lbm_cuda_ready/out/$(TST))
	@$(call compare, Test lbm_c2cuda:                    , Python/lbm_palabos_friendly/out/$(TST),   Cuda/lbm_c2cuda/out/$(TST))
	@$(call compare, Test lbm_float_gpu_vs_cpu (CPU):    , Python/lbm_palabos_friendly/out/$(TST),   Cuda/lbm_float_gpu_vs_cpu/out/$(TST)_cpu)
	@$(call compare, Test lbm_float_gpu_vs_cpu (GPU):    , Python/lbm_palabos_friendly/out/$(TST),   Cuda/lbm_float_gpu_vs_cpu/out/$(TST)_gpu)
	@$(call compare, Compare lbm_float_gpu_vs_cpu (both):, Cuda/lbm_float_gpu_vs_cpu/out/$(TST)_gpu, Cuda/lbm_float_gpu_vs_cpu/out/$(TST)_cpu)

$(LANGS): 
	$(MAKE) -C $@ $(MAKEFLAGS) 

clean:
	@$(foreach lang, $(LANGS), $(MAKE) -C $(lang) clean;)

.PHONY: all vars test $(LANGS) clean
