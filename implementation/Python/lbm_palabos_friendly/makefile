include ../../lbm_common.mk

# Compiler
PY=python3

# Directories
SRCDIR=.
OUTDIR=out

SCRIPT=lbmFlowAroundCylinder.py

# Script default arguments
OUTPRE=lbm_

DEFAULT_IMG_INTERVAL=100
DEFAULT_OUT_INTERVAL=0

run: images

images: $(OUTDIR)
	$(eval INTERVAL ?= $(DEFAULT_IMG_INTERVAL))
	$(PY) $(SRCDIR)/$(SCRIPT) IMG $(ITER) $(INTERVAL) $(OUTDIR) $(OUTPRE)

output: $(OUTDIR) 
	$(eval INTERVAL ?= $(DEFAULT_OUT_INTERVAL))
	$(call timing, "$(PY) $(SRCDIR)/$(SCRIPT) OUT $(ITER) $(INTERVAL) $(OUTDIR) $(OUTPRE)", $(OUTDIR)/$(OUTPRE)$(ITER).timing)

$(OUTDIR):
	mkdir -p $@

clean:
	rm -rf $(OUTDIR)

.PHONY: run images output clean