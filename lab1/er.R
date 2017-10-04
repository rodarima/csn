library(igraph, warn.conflicts = FALSE)
library(ggplot2)

N_MIN = 500
N_MAX = 5000
N_STEP = 500
DATA_FILE = "er.data"
REP = 10
UPDATE = T
N_RANGE = c(50, 100, 200, seq(N_MIN, N_MAX, N_STEP))
M = length(N_RANGE)
EPSILON = 0.5
SIZE = 4

if(UPDATE)
{
	avgsp = rep(0, M)

	for(i in seq(M)) {
		n = N_RANGE[i]
		print(sprintf("%d of %d. n = %d", i, M, n))
		p_hat = (1+EPSILON) * log(n)/n
		avgv = rep(0, REP)
		for(j in seq(REP)) {
			er = erdos.renyi.game(n, p_hat)
			edge_connectivity(er)
			avgv[j] = average.path.length(er)
		}
		avgsp[i] = mean(avgv)
	}
	save(avgsp, N_RANGE, file=DATA_FILE)

} else {
	load(DATA_FILE)
}

pdf(file="er.pdf", width=SIZE, height=SIZE)
print(qplot(N_RANGE, avgsp, type='o', xlab='Number of nodes', ylab='Average shortest path')
	+ geom_line() + theme_bw())
dev.off()
