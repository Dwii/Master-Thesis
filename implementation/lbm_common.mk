# Function:    timing
# Description: Store the execution time and LUPS of a LBM simulation in a file
# Argument $1: LBM simulation command (in a string) to execute
# Argument $2: File where the timing and lups are stored
define timing
	@start=$$(date +%s); \
	echo $$(eval "$(1)" | sed -n 's/average lups: \(.*\).*/\1/p') > $(2); \
	end=$$(date +%s); \
	T=$$(($$end-$$start)); \
	printf "$$T" >> $(2);
endef

# Binary arguments
ITER=200000