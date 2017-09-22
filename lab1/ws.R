library(igraph)

# Constants

XN = 15
R = seq(1, XN)
P = 2**-(R-1)
DIM = 1
SIZE = 200
NEI = 4
REP = 100

# Compute mean C0 using samples in C0v. Same for L0.

L0v = rep(0, REP)
C0v = rep(0, REP)

for(n in seq(REP)) {
	g0 = watts.strogatz.game(DIM, SIZE, NEI, 0)
	C0v[n] = transitivity(g0)
	L0v[n] = average.path.length(g0)
}

L0 = mean(L0v)
C0 = mean(C0v)

# Now compute L and C as p changes.

L = rep(0, XN)
C = rep(0, XN)

for(r in R) {
	Lv = rep(0, REP)
	Cv = rep(0, REP)

	# Sample REP times

	for(n in seq(REP)) {
		g = watts.strogatz.game(DIM, SIZE, NEI, P[r])
		Cp = transitivity(g)
		Cv[n] = Cp/C0
		Lp = average.path.length(g)
		Lv[n] = Lp/L0
	}
	C[r] = mean(Cv)
	L[r] = mean(Lv)
}

# Plot as pdf

pdf(file="ws.pdf")

title = sprintf("Watts-Strogatz graph (dim=%d, size=%d, neigh=%d)",
	DIM, SIZE, NEI)

plot(P, C, log='x', 
	type='o', col='black',
	xlab='p', ylab='',
	main=title)

lines(P, L, 
	type='o', col='red')

legend("topright",
	legend = c("Cp/C0", "Lp/L0"),
	col=c('black', 'red'), pch=c(1,1), lty=c(1,1))

grid()
dev.off()
