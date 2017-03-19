# usage: $(MAKE) [ test [ ITER=<total iterations> ] [ OUT=<test filename> ] ]

RED=\033[0;31m
GRN=\033[0;32m
BLU=\e[34m
NC=\033[0m

# Function:    check
# Description: Check that two files are identical, print the result and execution timing
# Argument $1: Test title 
# Argument $2: First file
# Argument $3: Second file 
# Argument $4: Timing file (optional)
define check
	$(eval timing=$(if $(strip $(4)), (time: $(shell cat $(4))), ""))
	$(eval delta=(diff: $(shell python3 Tools/floats_delta.py $(2) $(3))))
	$(eval na_msg="[$(BLU)NA$(NC)] $(1) $(timing)\n")
	$(eval ok_msg="[$(GRN)OK$(NC)] $(1) $(timing)\n")
	$(eval ko_msg="[$(RED)KO$(NC)] $(1) $(delta)\n")
	$(eval msg=$(if $(and $(strip $(2)), $(strip $(3))), $(if $(shell diff $(2) $(3)), $(ko_msg), $(ok_msg)), $(na_msg) ))
	printf $(msg)
endef

# Code folders
LANGS=Python C Cuda

# Default variables values
DEFAULT_ITER=1
DEFAULT_OUTPRE=lbm_

# Flags
MAKEFLAGS=--no-print-directory

all: $(LANGS)

vars: 
# Set default ITER if not set (ensure that all test run are equivalent)
ifeq ($(strip $(ITER)),)
	$(eval ITER=$(DEFAULT_ITER))
	$(eval MAKEFLAGS+=ITER=$(DEFAULT_ITER))
endif
# Set test output file name prefix (ensure that test diff path are correct)
ifeq ($(strip $(OUTPRE)),)
	$(eval OUTPRE=$(DEFAULT_OUTPRE))
	$(eval MAKEFLAGS+=OUTPRE=$(OUTPRE))
endif
# Set output and timing file name
	$(eval OUT=$(OUTPRE)$(ITER).out)
	$(eval TIMING=$(OUTPRE)$(ITER).timing)

output: vars
	@$(foreach lang, $(LANGS), $(MAKE) -C $(lang) output $(MAKEFLAGS);)

print_test: vars
	@$(call check,lbm_py                            ,,,                                                                                    Python/lbm_py/out/$(TIMING)                )
	@$(call check,lbm_palabos_friendly (Python)     ,,,                                                                                    Python/lbm_palabos_friendly/out/$(TIMING)  )
	@$(call check,lbm_py2c                          , Python/lbm_py/out/$(OUT),                  C/lbm_py2c/out/$(OUT),                    C/lbm_py2c/out/$(TIMING)                   )
	@$(call check,lbm_palabos_friendly (C)          , Python/lbm_palabos_friendly/out/$(OUT),    C/lbm_palabos_friendly/out/$(OUT),        C/lbm_palabos_friendly/out/$(TIMING)       )
	@$(call check,lbm_cuda_ready                    , Python/lbm_palabos_friendly/out/$(OUT),    C/lbm_cuda_ready/out/$(OUT),              C/lbm_cuda_ready/out/$(TIMING)             )
	@$(call check,lbm_c2cuda                        , Python/lbm_palabos_friendly/out/$(OUT),    Cuda/lbm_c2cuda/out/$(OUT),               Cuda/lbm_c2cuda/out/$(TIMING)              )
	@$(call check,lbm_float_gpu_vs_cpu (CPU)        , Python/lbm_palabos_friendly/out/$(OUT),    Cuda/lbm_float_gpu_vs_cpu/out/cpu_$(OUT), Cuda/lbm_float_gpu_vs_cpu/out/cpu_$(TIMING))
	@$(call check,lbm_float_gpu_vs_cpu (GPU)        , Python/lbm_palabos_friendly/out/$(OUT),    Cuda/lbm_float_gpu_vs_cpu/out/gpu_$(OUT), Cuda/lbm_float_gpu_vs_cpu/out/gpu_$(TIMING))
	@$(call check,lbm_float_gpu_vs_cpu (GPU vs CPU) , Cuda/lbm_float_gpu_vs_cpu/out/gpu_$(OUT),  Cuda/lbm_float_gpu_vs_cpu/out/cpu_$(OUT) )
	@$(call check,lbm_simple_gpu_vs_cpu (GPU vs CPU), Cuda/lbm_simple_gpu_vs_cpu/out/gpu_$(OUT), Cuda/lbm_simple_gpu_vs_cpu/out/cpu_$(OUT))
	@$(call check,lbm_opt1                          , Python/lbm_palabos_friendly/out/$(OUT),    Cuda/lbm_opt1/out/$(OUT),                 Cuda/lbm_opt1/out/$(TIMING)                )

test: vars output print_test

$(LANGS): 
	$(MAKE) -C $@ $(MAKEFLAGS) 

clean:
	@$(foreach lang, $(LANGS), $(MAKE) -C $(lang) clean;)

.PHONY: all vars output $(LANGS) clean
