# Function:    timing
# Description: Measure the execution time of a command a put it in a file
# Argument $1: Command (in a string) to execute and time
# Argument $2: File where the timing should be stored
define timing
	@start=$$(date +%s); \
	eval $(1); \
	end=$$(date +%s); \
	T=$$(($$end-$$start)); \
	printf "%02d:%02d:%02d" "$$((T/3600%24))" "$$((T/60%60))" "$$((T%60))" > $(2);
endef

# Binary arguments
ITER=200000