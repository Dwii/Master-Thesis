# Compiler
PY=python3

# Script arguments
ITER=200000

# Directories
SRCDIR=.
OUTDIR=out

# Test output file
OUT=lbm.out


run: $(OUTDIR)
	$(PY) $(SRCDIR)/lbmFlowAroundCylinder.py $(ITER) $(OUTDIR)

output: $(OUTDIR) 
	$(PY) $(SRCDIR)/lbmFlowAroundCylinder.py $(ITER) > $(OUTDIR)/$(OUT)

$(OUTDIR): 
	mkdir -p $@

clean:
	rm -rf $(OUTDIR)

.PHONY: run output clean