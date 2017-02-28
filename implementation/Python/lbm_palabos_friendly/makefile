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
ifeq ($(strip $(INTERVAL)),)
	$(eval INTERVAL=$(DEFAULT_IMG_INTERVAL))
endif
	$(PY) $(SRCDIR)/$(SCRIPT) IMG $(ITER) $(INTERVAL) $(OUTDIR) $(OUTPRE)

output: $(OUTDIR) 
ifeq ($(strip $(INTERVAL)),)
	$(eval INTERVAL=$(DEFAULT_OUT_INTERVAL))
endif
	$(call timing, "$(PY) $(SRCDIR)/$(SCRIPT) OUT $(ITER) $(INTERVAL) $(OUTDIR) $(OUTPRE)", $(OUTDIR)/$(OUTPRE)$(ITER).timing)

$(OUTDIR):
	mkdir -p $@

clean:
	rm -rf $(OUTDIR)

.PHONY: run images output clean