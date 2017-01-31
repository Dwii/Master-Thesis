LANGS=Python C Cuda


TST=test.out
all: $(LANGS)

test: all
	@$(foreach lang, $(LANGS), make -C $(lang) test;)

	@printf "\n-------- Tests --------\n"
	diff -q Python/lbm_py/out/$(TST) C/lbm_py2c/out/$(TST)
	diff -q Python/lbm_palabos_friendly/out/$(TST) C/lbm_palabos_friendly/out/$(TST)
	diff -q Python/lbm_palabos_friendly/out/$(TST) C/lbm_cuda_ready/out/$(TST)
	diff -q Python/lbm_palabos_friendly/out/$(TST) Cuda/lbm_c2cuda/out/$(TST)

$(LANGS): 
	make -C $@

clean:
	@$(foreach lang, $(LANGS), make -C $(lang) clean;)

.PHONY: test clean $(LANGS)
