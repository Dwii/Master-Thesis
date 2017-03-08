# Function:    wait_output
# Description: Wait for a file to exist and have a certain size
# Argument $1: File path
# Argument $2: File size
define wait_output
	@ \
	while [[ ! -s $(1) || $$(cat $(1) | wc -c) -ne $(2) ]]; do \
		test -z $$end && printf "wait for $(1)"; end="\n"; printf "."; sleep 2 ; \
	done; \
	printf "$$end"
endef

DLT_SCRIPT=../../Tools/floats_delta.py
DLT_SCRIPT_SHELL=python3
