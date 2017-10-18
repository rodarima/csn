require("VGAM")
require("bbmle")

DB = 'data/English_in-degree_sequence.txt'
LANGS = read.table('languages.txt', header=F, stringsAsFactors=F)$V1
NROWS = length(LANGS)

degree_sequence = read.table(DB, header = FALSE)
degree_spectrum = table(degree_sequence)
x <- degree_sequence$V1

# Truncated zeta (very slow)
rzeta <- function(gamma, kmax){
	ret = 0.0
	for(i in seq(kmax)){
		ret = ret + i**(-gamma)
	}
	return(ret)
}

# Compute the AIC corrected for small samples
get_AIC <- function(m2logL, K, N) {
	m2logL + 2*K*N/(N-K-1) # AIC with a correction for sample size
}

f1 <- function(x, i) {
	N = length(x)
	C = 0
	for (j in seq(N)) {
		C = C + sum(log(seq(2, x[j])))
	}
	mll_dpois <- function(lambda) {
		-(sum(x)*log(lambda)-N*(lambda+log(1-exp(-lambda)))-C)
	}
	ll <- mle2(
		mll_dpois,
		start = list(lambda = 2),
		method = "L-BFGS-B",
		lower = c(1e-5),
	)
	attr = attributes(summary(ll))
	m2ll = attr$m2logL
	best_lambda = attr$coef[1]
	t3$lambda[i] = best_lambda
	print(sprintf('f1: lambda = %f', best_lambda))
	t3 <<- t3
	return(get_AIC(m2ll, 1, N))
}
f2 <- function(x, i) {
	N = length(x)
	M = sum(x)
	mll_dgeo <- function(q) {
		-((M-N)*log(1-q)+N*log(q))

	}
	ll <- mle(
		mll_dgeo,
		start = list(q = 2),
		method = "L-BFGS-B",
		lower = c(1e-5),
		upper = c(1-1e-5),
	)
	attr = attributes(summary(ll))
	m2ll = attr$m2logL
	best_q = attr$coef[1]
	t3$q[i] = best_q
	t3 <<- t3
	#print(sprintf('f2: q = %f', best_q))
	return(get_AIC(m2ll, 1, N))
}
f3 <- function(x, i) {
	N = length(x)
	m2ll = -2*(-2*sum(log(x)) - N*log(pi**2/6))
	return(get_AIC(m2ll, 0, N))
}
f4 <- function(x, i) {
	N = length(x)
	mll_zeta <- function(gamma) {
		length(x) * log(zeta(gamma)) + gamma * sum(log(x))
	}
	ll <- mle2(
		mll_zeta,
		start = list(gamma = 2),
		method = "L-BFGS-B",
		lower = c(1.0000001),
		upper = c(10),
	)
	attr = attributes(summary(ll))
	m2ll = attr$m2logL
	best_gamma = attr$coef[1]
	t3$gamma1[i] = best_gamma
	t3 <<- t3
	#print(sprintf('f4: gamma = %f', best_gamma))
	return(get_AIC(m2ll, 1, N))
}
f5 <- function(x, i) {
	N = length(x)
	mll_rzeta <- function(gamma, kmax) {
		length(x) * log(rzeta(gamma, kmax)) + gamma * sum(log(x))
	}
	ll <- mle2(
		mll_rzeta,
		start = list(gamma = 2, kmax = 50),
		method = "L-BFGS-B",
		lower = c(1.0000001, 30),
		upper = c(10, 300),
	)
	attr = attributes(summary(ll))
	m2ll = attr$m2logL
	best_gamma = attr$coef[1]
	best_kmax = attr$coef[2]
	#print(sprintf('f5: gamma = %f kmax = %f', best_gamma, best_kmax))
	t3$gamma2[i] = best_gamma
	t3$kmax[i] = best_kmax
	t3 <<- t3
	return(get_AIC(m2ll, 2, N))
}

#f3_aic = f3(x)
#f4_aic = f4(x)
#f5_aic = f5(x)
#
#print(f3_aic)
#print(f4_aic)
#print(f5_aic)

fill_row <- function(x, i) {

	aics = c(f1(x, i), f2(x, i), f3(x, i), f4(x, i), f5(x, i))
	#print(aics)
	best_aic = min(aics)
	diff_aic = aics - best_aic
	for(j in length(aics)) {
		t4$a1[i] = diff_aic[1]
		t4$a2[i] = diff_aic[2]
		t4$a3[i] = diff_aic[3]
		t4$a4[i] = diff_aic[4]
		t4$a5[i] = diff_aic[5]
	}
	t4 <<- t4

	#print(diff_aic)
}

#summary(mle_zeta)




t3 = data.frame(
	lang=LANGS,
	lambda=numeric(NROWS),
	q=numeric(NROWS),
	gamma1=numeric(NROWS),
	gamma2=numeric(NROWS),
	kmax=numeric(NROWS)
)
t4 = data.frame(
	lang=LANGS,
	a1=numeric(NROWS),
	a2=numeric(NROWS),
	a3=numeric(NROWS),
	a4=numeric(NROWS),
	a5=numeric(NROWS)
)

# Fill a row
table_row <- function(lang, ds, i) {
	#d$lang[i] = lang # Doesn't work (?)
	fill_row(ds$V1, i)
}

# Build the table
for(i in seq(NROWS)){
	lang = LANGS[i]
	DB = sprintf('data/%s_in-degree_sequence.txt', lang)
	ds = read.table(DB, header = FALSE)
	table_row(lang, ds, i)
}

print(t3)
print(t4)


library(xtable)
xtable(t3)
xtable(t4)
