PROJECTS=lbm_py lbm_palabos_friendly

# Default variables values
DEFAULT_ITER=1

# Flags
MAKEFLAGS=--no-print-directory

all: # nothing to compile

vars:
ifeq ($(strip $(ITER)),)
	$(eval MAKEFLAGS+=ITER=$(DEFAULT_ITER))
endif

test: vars
	$(foreach project, $(PROJECTS), $(MAKE) $(MAKEFLAGS) -C $(project) test;)

clean:
	@$(foreach project, $(PROJECTS), $(MAKE) -C $(project) clean;)

.PHONY: all vars test clean
