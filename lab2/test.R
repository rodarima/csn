require("VGAM")

DB = 'data/English_in-degree_sequence.txt'

degree_sequence = read.table(DB, header = FALSE)
degree_spectrum = table(degree_sequence)

x <- degree_sequence$V1
minus_log_likelihood_zeta <- function(gamma) {
	length(x) * log(zeta(gamma)) + gamma * sum(log(x))
}

mle_zeta <- mle(
	minus_log_likelihood_zeta,
	start = list(gamma = 2),
	method = "L-BFGS-B",
	lower = c(1.0000001)
)
summary(mle_zeta)
