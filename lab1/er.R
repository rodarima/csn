library(igraph, warn.conflicts = FALSE)

N_MIN = 500
N_MAX = 10000
N_STEP = 500
DATA_FILE = "er.data"
REP = 10
UPDATE = FALSE
N_RANGE = seq(N_MIN, N_MAX, N_STEP)
M = length(N_RANGE)

if(UPDATE)
{
	avgsp = rep(0, M)

	for(i in seq(M)) {
		n = N_RANGE[i]
		print(n)
		p_hat = log(n)/n
		avgv = rep(0, REP)
		for(j in seq(REP)) {
			er = erdos.renyi.game(n, p_hat)
			avgv[j] = average.path.length(er)
		}
		avgsp[i] = mean(avgv)
	}
	save(avgsp, N_RANGE, file=DATA_FILE)

} else {
	load(DATA_FILE)
}

pdf(file="er.pdf")
plot(N_RANGE, avgsp, type='o', xlab='Number of nodes', ylab='Average shortest path',
	main='Erdös-Rényi graph (p = ln(n)/n)')
dev.off()
