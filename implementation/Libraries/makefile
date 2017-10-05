all: array pgm timing lbmmain lbmmain3d lbmmain3d_dynsize lbmcuda

array: 
	$(MAKE) -C array

pgm: 
	$(MAKE) -C pgm

timing: 
	$(MAKE) -C timing

lbmmain: 
	$(MAKE) -C lbmmain

lbmmain3d: 
	$(MAKE) -C lbmmain3d

lbmmain3d_dynsize:
	$(MAKE) -C lbmmain3d_dynsize 

lbmcuda:
	$(MAKE) -C lbmcuda 

clean:
	$(MAKE) -C array clean
	$(MAKE) -C pgm clean
	$(MAKE) -C timing clean
	$(MAKE) -C lbmmain clean
	$(MAKE) -C lbmmain3d clean
	$(MAKE) -C lbmmain3d_dynsize clean
	$(MAKE) -C lbmcuda clean

.PHONY: array pgm timing lbmmain lbmmain3d lbmmain3d_dynsize lbmcuda clean
