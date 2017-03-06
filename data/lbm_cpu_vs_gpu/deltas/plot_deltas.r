#!/usr/bin/env Rscript

Sys.setenv(LANG = "en")

require(methods)

FILE <- "deltas.csv"

Y_MAX_MULT=1.09

setClass("slurmDate")

colClasses = c(
	'numeric',
    'numeric'
)

deltas <- read.table(file = FILE, colClasses=colClasses, header=FALSE, sep = ";")

min_delta = min(deltas[,2])
max_delta = max(deltas[,2])

min_iter = min(deltas[,1])
max_iter = max(deltas[,1])


print(min_delta)
print(max_delta)

sprintf("min_delta = %.60f", min_delta)
sprintf("max_delta = %.60f", max_delta)

ylim=c(min_delta, max_delta*Y_MAX_MULT)

plot( c(deltas[,2]), col=c("blue"), ylab="Différence moyenne entre CPU et GPU", xlab="Itération", bty="n", type = "l", ylim=ylim, xaxt="n")

v1 = c()
v2 = c()

for (i in 0:max_iter) {
    if (i %% 10000 == 0) {
        v1 = append(v1, i)
        v2 = append(v2, i)
    } else if (i %% 5000 == 0) {
        v1 = append(v1, i)
        v2 = append(v2, '')
    }
}

axis(side = 1, at = v1, labels = v2)

grid(nx = NULL, col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
