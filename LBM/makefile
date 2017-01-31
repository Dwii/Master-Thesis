LANGS=Python C

all: $(LANGS)

test: all
	@$(foreach lang, $(LANGS), make -C $(lang) test;)

	@printf "\n-------- Tests --------\n"
	diff -q Python/lbm_py/out/lbm_py.out C/lbm_py2c/out/lbm_py2c.out
	diff -q Python/lbm_palabos_friendly/out/lbm_palabos_friendly.out C/lbm_palabos_friendly/out/lbm_palabos_friendly.out
	diff -q Python/lbm_palabos_friendly/out/lbm_palabos_friendly.out C/lbm_cuda_ready/out/lbm_cuda_ready.out
	diff -q Python/lbm_palabos_friendly/out/lbm_palabos_friendly.out Cuda/lbm_c2cuda/out/lbm_c2cuda.out

$(LANGS): 
	make -C $@

clean:
	@$(foreach lang, $(LANGS), make -C $(lang) clean;)

.PHONY: test clean $(LANGS)
