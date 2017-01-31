PROJECTS=lbm_c2cuda

all: $(PROJECTS)

run: all
ifeq ($(strip $(project)),)
	@echo
	@echo "usage: make run project=<project name>"
	@echo "project names: "
	@$(foreach project, $(PROJECTS), echo " - $(project)";)
else
	make -C $(project) run
endif

test: all
	@$(foreach project, $(PROJECTS), make -C $(project) test ITER=1;)

$(PROJECTS): 
	make -C $@

clean:
	@$(foreach project, $(PROJECTS), make -C $(project) clean;)

.PHONY: run test clean $(PROJECTS)
