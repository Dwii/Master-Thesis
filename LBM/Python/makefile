PROJECTS=lbm_py lbm_palabos_friendly

run: 
ifeq ($(strip $(project)),)
	@echo
	@echo "usage: make run project=<project name>"
	@echo "project names: "
	@$(foreach project, $(PROJECTS), echo " - $(project)";)
else
	make -C $(project) run
endif

test: 
	$(foreach project, $(PROJECTS), make -C $(project) test ITER=1;)

clean:
	@$(foreach project, $(PROJECTS), make -C $(project) clean;)

.PHONY: run test clean $(PROJECTS)
