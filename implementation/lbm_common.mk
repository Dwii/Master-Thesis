noop=
space=$(noop) $(noop)
SHELL=/bin/bash

# Function:    execute
# Description: Export list of libraries path, execute command and store the 
#              execution time and LUPS of a LBM simulation if a info file 
#              path is provided.
# Argument $1: Dynamic libraries paths (separated by spaces)
# Argument $2: LBM simulation command (in a string) to execute
# Argument $3: (optional) info file path
define execute
	@info=$$(echo "$(3)" | xargs); \
	llp_var=$(if $(filter $(shell uname),Darwin),"DYLD_LIBRARY_PATH","LD_LIBRARY_PATH"); \
	[[ "$$info" == "" ]] && info=/dev/null; \
	eval "export $$llp_var=$${!llp_var}$(subst $(space):,:,$(addprefix :,$(1)))"; \
	start=$$(date +%s); \
	echo $$(eval $(2) | tee $(STD_FILE) > $$info ) ; \
	end=$$(date +%s); \
	T=$$(($$end-$$start)); \
	printf "execution time: $$T" >> $$info;
endef

# Rule .stdout: set STD_FILE as the standard output for execute
.stdout: ;@:
	$(eval STD_FILE=/dev/tty)

.PHONY: .stdout

# Binary arguments
ITER=3000