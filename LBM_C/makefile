PROJECTS=lbm_py2c lbm_palabos_friendly

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

libs: 
	make -C libs

$(PROJECTS): 
	make -C $@

clean:
	make -C libs clean
	@$(foreach project, $(PROJECTS), make -C $(project) clean;)

.PHONY: run libs clean $(PROJECTS)