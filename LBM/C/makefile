PROJECTS=lbm_py2c lbm_palabos_friendly lbm_cuda_ready

# Default variables values
DEFAULT_ITER=1

all: libs $(PROJECTS)

vars:
ifeq ($(strip $(ITER)),)
	$(eval MAKEFLAGS+=ITER=$(DEFAULT_ITER))
endif

test: vars all
	@$(foreach project, $(PROJECTS), make -C $(project) test $(MAKEFLAGS);)

libs: 
	make -C libs

$(PROJECTS): 
	make -C $@

clean:
	make -C libs clean
	@$(foreach project, $(PROJECTS), make -C $(project) clean;)

.PHONY: all vars test libs $(PROJECTS) clean
