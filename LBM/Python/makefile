PROJECTS=lbm_py lbm_palabos_friendly

# Default variables values
DEFAULT_ITER=1

all: # nothing to compile

vars:
ifeq ($(strip $(ITER)),)
	$(eval MAKEFLAGS+=ITER=$(DEFAULT_ITER))
endif

test: vars
	$(foreach project, $(PROJECTS), make -C $(project) test $(MAKEFLAGS);)

clean:
	@$(foreach project, $(PROJECTS), make -C $(project) clean;)

.PHONY: all vars test clean
