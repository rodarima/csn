library(igraph, warn.conflicts = FALSE)
#library(ggplot2)
#library(reshape2)

# Constants

XN = 15
R = seq(1, XN)
P = 2**-(R-1)
DIM = 1
SIZE = 500
NEI = 4
REP = 500
PDF_SIZE = 6
PB = P[seq(XN/4)*4]

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
	print(r)
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

pdf(file="ws.pdf", width=PDF_SIZE, height=PDF_SIZE)


# ggplot seems to be more complicated for this task, or I don't have enough
# experience with it yet
#
#p1 = qplot(P, C, log='x', 
#	type='o', col='black',
#	xlab='p', ylab='',
#	main=title) + theme_bw()

#df = data.frame(P=P, C=C, L=L)
##xymelt <- melt(df, id.vars = "P")
##
##ggplot(xymelt, aes(x = 'P', y = value, color = variable)) +
##  theme_bw() +
##  geom_line() +
##  scale_colour_manual(values =c('black'='black','red'='red'), labels = c('c2','c1'))
#
#
#ggplot(df, aes(x=P)) + 
#	theme_bw() +
#	geom_point(aes(y=C), color='black') +
#	geom_line(aes(y=C), color='black') + 
#	geom_point(aes(y=L), color='red') +
#	geom_line(aes(y=L), color='red') + 
#	scale_x_log10(breaks=10**seq(0, -4),
#		labels = function(x) format(x, scientific = TRUE)) +
##	theme(legend.position = c(0.8, 0.8)) +
##	scale_colour_manual("",
##		breaks = c("TempMax", "TempMedia", "TempMin"),
##		values = c("red", "green", "blue")) +
#	scale_colour_manual(values =c('black'='black','red'='red'), labels = c('c2','c1')) +
#	labs(y="")

title = sprintf("Watts-Strogatz graph (dim=%d, size=%d, neigh=%d)",
	DIM, SIZE, NEI)

plot(P, C, log='x', 
	type='o', col='black',
	xlab='p', ylab='', main=title)
lines(P, L, 
	type='o', col='red')

legend("topright",
	legend = c("C(p)/C(0)", "L(p)/L(0)"),
	col=c('black', 'red'), pch=c(1,1), lty=c(1,1))

grid()
dev.off()
