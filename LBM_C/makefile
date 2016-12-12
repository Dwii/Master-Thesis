all: libs lbm_py2c lbm_collisions_by_cells lbm_palabos_friendly

run: all
ifeq ($(strip $(project)),)
	@echo "missing project name: make run project=<project name>"
else
	make -C $(project) run
endif

libs: 
	make -C libs

lbm_py2c: 
	make -C lbm_py2c

lbm_collisions_by_cells: 
	make -C lbm_collisions_by_cells

lbm_palabos_friendly:
	make -C lbm_palabos_friendly

clean:
	make -C libs clean
	make -C lbm_py2c clean
	make -C lbm_collisions_by_cells clean
	make -C lbm_palabos_friendly clean

.PHONY: run libs clean lbm_py2c lbm_collisions_by_cells lbm_palabos_friendly