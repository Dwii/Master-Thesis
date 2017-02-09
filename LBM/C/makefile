PROJECTS=lbm_py2c lbm_palabos_friendly lbm_cuda_ready

# Default variables values
DEFAULT_ITER=1

# Flags
MAKEFLAGS=--no-print-directory

all: libs $(PROJECTS)

vars:
ifeq ($(strip $(ITER)),)
	$(eval MAKEFLAGS+=ITER=$(DEFAULT_ITER))
endif

output: vars all
	@$(foreach project, $(PROJECTS), $(MAKE) -C $(project) output $(MAKEFLAGS);)

libs: 
	$(MAKE) -C libs

$(PROJECTS): 
	$(MAKE) -C $@ $(MAKEFLAGS)

clean:
	$(MAKE) -C libs clean
	@$(foreach project, $(PROJECTS), $(MAKE) -C $(project) clean;)

.PHONY: all vars output libs $(PROJECTS) clean
