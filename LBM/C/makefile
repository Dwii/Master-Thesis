PROJECTS=lbm_py2c lbm_palabos_friendly lbm_cuda_ready

all: libs $(PROJECTS)

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

libs: 
	make -C libs

$(PROJECTS): 
	make -C $@

clean:
	make -C libs clean
	@$(foreach project, $(PROJECTS), make -C $(project) clean;)

.PHONY: run test libs clean $(PROJECTS)
