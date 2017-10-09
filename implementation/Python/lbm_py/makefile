include ../../lbm_common.mk

# Compiler
PY=python3

# Directories
SRCDIR=.
OUTDIR=out

SCRIPT=lbmFlowAroundCylinder.py

# Script default arguments
OUTPRE=lbm_

all: images

# Targets addition arguments
run:    TRGTARGS=NONE
images: TRGTARGS=IMG
output: TRGTARGS=OUT

# Targets defaut intervals
run output: INTERVAL?=0
images:     INTERVAL?=100

# Targets timing file path (left unset if not required)
output: INFO=$(OUTDIR)/$(OUTPRE)$(ITER).info

run images output: $(OUTDIR)
	$(call execute,, "$(PY) $(SRCDIR)/$(SCRIPT) $(TRGTARGS) $(ITER) $(INTERVAL) $(OUTDIR) $(OUTPRE)", $(INFO))

$(OUTDIR):
	mkdir -p $@

clean:
	rm -rf $(OUTDIR)

.PHONY: run images output clean