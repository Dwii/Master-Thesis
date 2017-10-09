# usage: $(MAKE) [ test [ ITER=<total iterations> ] [ OUT=<test filename> ] ]

RED=\033[0;31m
YEL=\033[0;33m
GRN=\033[0;32m
BLU=\033[0;34m
NC=\033[0m

NX=420
NY=180
quote='

# Function:    check
# Description: Check that two files are identical, print the result and execution timing
# Argument $1: Test title 
# Argument $2: First file
# Argument $3: Second file 
# Argument $4: Info file (optional)
define check
	$(eval err=)
	$(if $(shell test -n '$(strip $(4))' -a ! -e '$(strip $(4))' && echo ERR),$(eval err=(info file not found)))
	$(if $(shell test -n '$(strip $(3))' -a ! -e '$(strip $(3))' && echo ERR),$(eval err=($(strip $(3)) not found)))
	$(if $(shell test -n '$(strip $(2))' -a ! -e '$(strip $(2))' && echo ERR),$(eval err=($(strip $(2)) not found)))
	$(if $(err), 
	$(eval msg="[$(YEL)ER$(NC)] $(1) $(err)\n")
	,
	$(eval seconds=$(if $(strip $(4)), $(shell cat $(4) | sed -n $(quote)s/.*execution time: \(.*\).*/\1/p$(quote) ), ))
	$(eval time=$(if $(strip $(4)), $(shell printf "%02d:%02d:%02d" $$(($(seconds)/3600%24)) $$(($(seconds)/60%60)) $$(($(seconds)%60))), ""))
	$(eval lups=$(if $(strip $(4)), $(shell cat $(4) | sed -n $(quote)s/.*average lups: \(.*\).*/\1/p$(quote)), ))
	$(eval mlups=$(if $(strip $(lups)), $(shell export LC_NUMERIC="en_US.UTF-8"; printf "%.2f" $$(printf "$(lups)/10^6\n" | bc -l )), "unknown"))
	$(eval perf=$(if $(strip $(4)), (time: $(time), MLUPS: $(mlups)), ))
	$(eval delta=(diff: $(shell python3 Tools/floats_delta.py $(2) $(3))))
	$(eval na_msg="[$(BLU)NA$(NC)] $(1) $(perf)\n")
	$(eval ok_msg="[$(GRN)OK$(NC)] $(1) $(perf)\n")
	$(eval ko_msg="[$(RED)KO$(NC)] $(1) $(delta)\n")
	$(eval msg=$(if $(and $(strip $(2)), $(strip $(3))), $(if $(shell diff $(2) $(3)), $(ko_msg), $(ok_msg)), $(na_msg) ))
	)
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
# Set output and info file name
	$(eval OUT=$(OUTPRE)$(ITER).out)
	$(eval INFO=$(OUTPRE)$(ITER).info)

output: vars
	@$(foreach lang, $(LANGS), $(MAKE) -C $(lang) output $(MAKEFLAGS);)

print_test: vars
	@$(call check,lbm_py                            ,,,                                                                                    Python/lbm_py/out/$(INFO)                )
	@$(call check,lbm_palabos_friendly (Python)     ,,,                                                                                    Python/lbm_palabos_friendly/out/$(INFO)  )
	@$(call check,lbm_py2c                          , Python/lbm_py/out/$(OUT),                  C/lbm_py2c/out/$(OUT),                    C/lbm_py2c/out/$(INFO)                   )
	@$(call check,lbm_palabos_friendly (C)          , Python/lbm_palabos_friendly/out/$(OUT),    C/lbm_palabos_friendly/out/$(OUT),        C/lbm_palabos_friendly/out/$(INFO)       )
	@$(call check,lbm_cuda_ready                    , Python/lbm_palabos_friendly/out/$(OUT),    C/lbm_cuda_ready/out/$(OUT),              C/lbm_cuda_ready/out/$(INFO)             )
	@$(call check,lbm_c2cuda                        , Python/lbm_palabos_friendly/out/$(OUT),    Cuda/lbm_c2cuda/out/$(OUT),               Cuda/lbm_c2cuda/out/$(INFO)              )
	@$(call check,lbm_float_gpu_vs_cpu (CPU)        , Python/lbm_palabos_friendly/out/$(OUT),    Cuda/lbm_float_gpu_vs_cpu/out/cpu_$(OUT), Cuda/lbm_float_gpu_vs_cpu/out/cpu_$(INFO))
	@$(call check,lbm_float_gpu_vs_cpu (GPU)        , Python/lbm_palabos_friendly/out/$(OUT),    Cuda/lbm_float_gpu_vs_cpu/out/gpu_$(OUT), Cuda/lbm_float_gpu_vs_cpu/out/gpu_$(INFO))
	@$(call check,lbm_float_gpu_vs_cpu (GPU vs CPU) , Cuda/lbm_float_gpu_vs_cpu/out/gpu_$(OUT),  Cuda/lbm_float_gpu_vs_cpu/out/cpu_$(OUT) )
	@$(call check,lbm_simple_gpu_vs_cpu (GPU vs CPU), Cuda/lbm_simple_gpu_vs_cpu/out/gpu_$(OUT), Cuda/lbm_simple_gpu_vs_cpu/out/cpu_$(OUT))
	@$(call check,lbm_opt1                          , Python/lbm_palabos_friendly/out/$(OUT),    Cuda/lbm_opt1/out/$(OUT),                 Cuda/lbm_opt1/out/$(INFO)                )
	@$(call check,lbm_sailfish_hist (CPU)           ,,,                                                                                    Cuda/lbm_sailfish_hist/out/cpu_$(INFO)   )
	@$(call check,lbm_sailfish_hist (GPU)           ,,,                                                                                    Cuda/lbm_sailfish_hist/out/gpu_$(INFO)   )
	@$(call check,lbm_opt2                          , Python/lbm_palabos_friendly/out/$(OUT),    Cuda/lbm_opt2/out/$(OUT),                 Cuda/lbm_opt2/out/$(INFO)                )
	@$(call check,lbm_simple_3d (Python)            ,,,                                                                                    Python/lbm_simple_3d/out/$(INFO)         )
	@$(call check,lbm_simple_3d (CPU)               , Python/lbm_simple_3d/out/$(OUT),           C/lbm_simple_3d/out/$(OUT),               C/lbm_simple_3d/out/$(INFO)              )
	@$(call check,lbm_simple_3d (GPU)               , Python/lbm_simple_3d/out/$(OUT),           Cuda/lbm_simple_3d/out/$(OUT),            Cuda/lbm_simple_3d/out/$(INFO)           )
	@$(call check,lbm_simple_3d_dynsize (GPU)       , Python/lbm_simple_3d/out/$(OUT),           Cuda/lbm_simple_3d_dynsize/out/$(OUT),    Cuda/lbm_simple_3d_dynsize/out/$(INFO)   )
	@$(call check,lbm_simple_lbmcuda                , Python/lbm_simple_3d/out/$(OUT),           Cuda/lbm_simple_lbmcuda/out/$(OUT),       Cuda/lbm_simple_lbmcuda/out/$(INFO)      )

test: vars output print_test

$(LANGS): 
	$(MAKE) -C $@ $(MAKEFLAGS) 

clean:
	@$(foreach lang, $(LANGS), $(MAKE) -C $(lang) clean;)
	$(MAKE) -C Libraries clean

.PHONY: all vars output $(LANGS) clean
