library(igraph, warn.conflicts = FALSE)
require(ggplot2)

REP = 2000
EPSILONS = seq(0.2, 0.6, 0.02)
R = length(EPSILONS)
n = 100
SIZE = 4

conn = matrix(, nrow=REP, ncol=R)
zeros = rep(0, R)
for (r in seq(R))
{
	print(r)
	epsilon = EPSILONS[r]
	p = (1 + epsilon) * log(n)/n
	for (i in seq(REP))
	{
		er = erdos.renyi.game(n, p)
		ec = edge_connectivity(er)
		conn[i, r] = ec
		if(ec == 0) { zeros[r] = zeros[r]+ 1}
	}
}

#print(conn)
#print(EPSILONS)
#print(colMeans(conn))

means = colMeans(conn)
pzeros = zeros/REP

pdf(file="er-prob.pdf", width=SIZE, height=SIZE)
print(qplot(EPSILONS, means, type='o', xlab='epsilon', ylab='Average edge connectivity',
#	main='Erdös-Rényi graph (p = (1+epsilon)*ln(n)/n) with n=200'
	) + theme_bw())
#grid()
dev.off()

pdf(file="er-zeros.pdf", width=SIZE, height=SIZE)
print(qplot(EPSILONS, pzeros, type='o', xlab='epsilon', ylab='Average separated graphs',
#	main='Erdös-Rényi graph (p = (1+epsilon)*ln(n)/n) with n=200'
	) + theme_bw())
#grid()
dev.off()
