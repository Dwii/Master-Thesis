include ../../lbm_common.mk

# Compiler
PY=python3

# Directories
SRCDIR=.
OUTDIR=out

# Test output file
OUT=lbm.out


run: $(OUTDIR)
	$(PY) $(SRCDIR)/lbmFlowAroundCylinder.py $(ITER) $(OUTDIR)

output: $(OUTDIR)
	$(call timing, "$(PY) $(SRCDIR)/lbmFlowAroundCylinder.py $(ITER) > $(OUTDIR)/$(OUT)", $(OUTDIR)/$(OUT).timing)

$(OUTDIR):
	mkdir -p $@

clean:
	rm -rf $(OUTDIR)

.PHONY: run output clean