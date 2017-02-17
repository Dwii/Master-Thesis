PROJECTS=lbm_float_gpu_vs_cpu lbm_c2cuda lbm_simple

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
	@$(foreach project, $(PROJECTS), $(MAKE) -C $(project) clean;)

.PHONY: all vars output $(PROJECTS) clean
