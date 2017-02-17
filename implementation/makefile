# usage: $(MAKE) [ test [ ITER=<total iterations> ] [ OUT=<test filename> ] ]

RED=\033[0;31m
GRN=\033[0;32m
NC=\033[0m 

# Function:    compare
# Description: Test that two files are identical and print the check result 
# Argument $1: Test title 
# Argument $2: First file
# Argument $3: Second file 
# Argument $3: Timing file (optional)
define compare
	$(eval timing=$(if $(strip $(4)), (time: $(shell cat $(4))), ""))
	$(eval delta=(diff: $(shell python3 Tools/floats_delta.py $(2) $(3))))
	$(eval ok_msg="$(1)  $(GRN)OK$(NC)  $(timing) \n")
	$(eval ko_msg="$(1)  $(RED)KO$(NC)  $(delta)\n")
	-@diff -q $(2) $(3) 2>&1 1>/dev/null; if [ $$? -eq 0 ] ; then printf $(ok_msg) ; else printf $(ko_msg); fi
endef

# Code folders
LANGS=Python C Cuda

# Default variables values
DEFAULT_ITER=1
DEFAULT_OUT=lbm.out

# Flags
MAKEFLAGS=--no-print-directory

all: $(LANGS)

vars: 
# Set default ITER if not set (ensure that all test run are equivalent)
ifeq ($(strip $(ITER)),)
	$(eval MAKEFLAGS+=ITER=$(DEFAULT_ITER))
endif
# Set test output file name (ensure that test diff path are correct)
ifeq ($(strip $(OUT)),)
	$(eval OUT=$(DEFAULT_OUT))
	$(eval MAKEFLAGS+=OUT=$(OUT))
endif

output: vars
	@$(foreach lang, $(LANGS), $(MAKE) -C $(lang) output $(MAKEFLAGS);)

print_test: vars
	@$(call compare, Test lbm_py2c:                      , Python/lbm_py/out/$(OUT),                 C/lbm_py2c/out/$(OUT),                    C/lbm_py2c/out/$(OUT).timing,                  )
	@$(call compare, Test lbm_palabos_friendly:          , Python/lbm_palabos_friendly/out/$(OUT),   C/lbm_palabos_friendly/out/$(OUT),        C/lbm_palabos_friendly/out/$(OUT).timing,      )
	@$(call compare, Test lbm_cuda_ready:                , Python/lbm_palabos_friendly/out/$(OUT),   C/lbm_cuda_ready/out/$(OUT),              C/lbm_cuda_ready/out/$(OUT).timing,            )
	@$(call compare, Test lbm_c2cuda:                    , Python/lbm_palabos_friendly/out/$(OUT),   Cuda/lbm_c2cuda/out/$(OUT),               Cuda/lbm_c2cuda/out/$(OUT).timing,             )
	@$(call compare, Test lbm_float_gpu_vs_cpu (CPU):    , Python/lbm_palabos_friendly/out/$(OUT),   Cuda/lbm_float_gpu_vs_cpu/out/cpu_$(OUT), Cuda/lbm_float_gpu_vs_cpu/out/cpu_$(OUT).timing)
	@$(call compare, Test lbm_float_gpu_vs_cpu (GPU):    , Python/lbm_palabos_friendly/out/$(OUT),   Cuda/lbm_float_gpu_vs_cpu/out/gpu_$(OUT), Cuda/lbm_float_gpu_vs_cpu/out/gpu_$(OUT).timing)
	@$(call compare, Compare lbm_float_gpu_vs_cpu (both):, Cuda/lbm_float_gpu_vs_cpu/out/gpu_$(OUT), Cuda/lbm_float_gpu_vs_cpu/out/cpu_$(OUT))

test: vars output print_test

$(LANGS): 
	$(MAKE) -C $@ $(MAKEFLAGS) 

clean:
	@$(foreach lang, $(LANGS), $(MAKE) -C $(lang) clean;)

.PHONY: all vars output $(LANGS) clean
