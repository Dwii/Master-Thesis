PROJECTS=lbm_c2cuda

# Default variables values
DEFAULT_ITER=1

all: $(PROJECTS)

vars:
ifeq ($(strip $(ITER)),)
	$(eval MAKEFLAGS+=ITER=$(DEFAULT_ITER))
endif

test: vars all
	@$(foreach project, $(PROJECTS), make -C $(project) test $(MAKEFLAGS);)

$(PROJECTS): 
	make -C $@

clean:
	@$(foreach project, $(PROJECTS), make -C $(project) clean;)

.PHONY: all vars test $(PROJECTS) clean
