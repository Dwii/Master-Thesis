# Compiler
PY=python3

# Script arguments
ITER=200000

# Directories
SRCDIR=.
OUTDIR=out

# Test output file
TST=$(shell basename "$(abspath .)").out


run: $(OUTDIR)
	$(PY) $(SRCDIR)/lbmFlowAroundCylinder.py $(ITER) $(OUTDIR)

test: $(OUTDIR) 
	$(PY) $(SRCDIR)/lbmFlowAroundCylinder.py $(ITER) > $(OUTDIR)/$(TST)

$(OUTDIR): 
	mkdir -p $@

clean:
	rm -rf $(OUTDIR)

.PHONY: run test clean