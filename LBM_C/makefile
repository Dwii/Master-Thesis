all: libs lbm_py2c

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

clean:
	make -C libs clean
	make -C lbm_py2c clean

.PHONY: run libs lbm_py2c clean