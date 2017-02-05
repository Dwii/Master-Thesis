# usage: $(MAKE) [ test [ ITER=<total iterations> ] [ TST=<test filename> ] ]

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

	@printf "\n-------- Tests --------\n"
	diff -q Python/lbm_py/out/$(TST) C/lbm_py2c/out/$(TST)
	diff -q Python/lbm_palabos_friendly/out/$(TST) C/lbm_palabos_friendly/out/$(TST)
	diff -q Python/lbm_palabos_friendly/out/$(TST) C/lbm_cuda_ready/out/$(TST)
	diff -q Python/lbm_palabos_friendly/out/$(TST) Cuda/lbm_c2cuda/out/$(TST)

$(LANGS): 
	$(MAKE) -C $@ $(MAKEFLAGS) 

clean:
	@$(foreach lang, $(LANGS), $(MAKE) -C $(lang) clean;)

.PHONY: all vars test $(LANGS) clean
