PROJECTS=lbm_float_gpu_vs_cpu lbm_c2cuda

# Default variables values
DEFAULT_ITER=1

# Flags
MAKEFLAGS=--no-print-directory

all: $(PROJECTS)

vars:
ifeq ($(strip $(ITER)),)
	$(eval MAKEFLAGS+=ITER=$(DEFAULT_ITER))
endif

test: vars all
	@$(foreach project, $(PROJECTS), $(MAKE) -C $(project) test $(MAKEFLAGS);)

$(PROJECTS): 
	$(MAKE) -C $@ $(MAKEFLAGS)

clean:
	@$(foreach project, $(PROJECTS), $(MAKE) -C $(project) clean;)

.PHONY: all vars test $(PROJECTS) clean
