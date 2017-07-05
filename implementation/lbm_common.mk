noop=
space=$(noop) $(noop)

# Function:    execute
# Description: Export list of libraries path, execute command and store the 
#              execution time and LUPS of a LBM simulation if a timing file 
#              path is provided.
# Argument $1: Dynamic libraries paths (separated by spaces)
# Argument $2: LBM simulation command (in a string) to execute
# Argument $3: (optional) timing file path
define execute
	@timing=$$(echo "$(3)" | xargs); \
	[[ "$$timing" == "" ]] && timing=/dev/null; \
	export DYLD_LIBRARY_PATH=$$DYLD_LIBRARY_PATH$(subst $(space):,:,$(addprefix :,$(1))); \
	start=$$(date +%s); \
	echo $$(eval "$(2) | tee $(STD_FILE)" ) | sed -n 's/.*average lups: \(.*\).*/\1/p' > $$timing; \
	end=$$(date +%s); \
	T=$$(($$end-$$start)); \
	printf "$$T" >> $$timing;
endef

# Rule .stdout: set STD_FILE as the standard output for execute
.stdout: ;@:
	$(eval STD_FILE=/dev/tty)

.PHONY: .stdout

# Binary arguments
ITER=200000