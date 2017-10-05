PROJECTS=lbm_py2c lbm_palabos_friendly lbm_cuda_ready lbm_simple_3d

# Default variables values
DEFAULT_ITER=1

# Flags
MAKEFLAGS=--no-print-directory

all: $(PROJECTS)

vars:
ifeq ($(strip $(ITER)),)
	$(eval MAKEFLAGS+=ITER=$(DEFAULT_ITER))
endif

output: vars all
	@$(foreach project, $(PROJECTS), $(MAKE) -C $(project) output $(MAKEFLAGS);)

$(PROJECTS): 
	$(MAKE) -C $@ $(MAKEFLAGS)

clean:
	$(MAKE) -C clean
	@$(foreach project, $(PROJECTS), $(MAKE) -C $(project) clean;)

.PHONY: all vars output $(PROJECTS) clean
