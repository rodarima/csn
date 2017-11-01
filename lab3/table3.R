# Install the package BiRewire following the instructions at:
# https://bioconductor.org/packages/release/bioc/html/BiRewire.html
#
# It needs the packages: igraph, slam, tsne, Matrix
#
# I used the following commands:
# > source("https://bioconductor.org/biocLite.R")
# > biocLite("BiRewire")

REP = 25 # Number of repetitions
TABLE3 = 'table3.csv'
GRAPH_FILE = 'data/%s_syntactic_dependency_network.edges'
set.seed(3) # Reproducible results

#print('Loading BiRewire')
suppressMessages(library('BiRewire'))

analyze <- function(f) {

	#print(sprintf('Reading graph from %s', f))
	t = read.table(f, sep=' ', quote='', na.strings = '', colClasses = "character")
	g <- graph_from_data_frame(t)

	# Remove loops and multiedges
	#print('Cleaning graph')
	g <- simplify(g)

	ecount = length(E(g))
	Q = round(log(ecount))

	#tr1 = transitivity(g, type='global')
	trl1 = transitivity(g, type='localaverage', isolates='zero')
	#trv = rep(0, REP)
	trlv = rep(0, REP)

	for (r in seq(REP)) {

		#print(sprintf('Rewiring graph %d/%d', r, REP))
		g2 = birewire.rewire.undirected(g, Q * ecount, verbose=F)

		#trv[r] = transitivity(g2, type='global')
		trlv[r] = transitivity(g2, type='localaverage', isolates='zero')
		remove(g2)
	}

	#print(trv)
	#print(sprintf('Trans. global orig: %e, rewire mean: %e', tr1, mean(trv)))
	#print(sprintf('Trans. local orig: %e, rewire mean: %e', trl1, mean(trlv)))
	#prob_g = sum(trv >= tr1)/REP
	prob_l = sum(trlv >= trl1)/REP
	tr_mean = mean(trlv)
	#print(sprintf('global: prob(x_NH >= x) ~= %f', prob_g))
	#print(sprintf('local:  prob(x_NH >= x) ~= %f', prob_l))
	print(sprintf('%s %E %E %E', f, trl1, tr_mean, prob_l))
	remove(g)
}


graph_files <- "graphs.txt"
gf <- file(graph_files, open = "r")

LANGS <- readLines(gf, warn = FALSE)
NNOWS = length(lines)
print(lines)
print(NLANG)

t3 = data.frame(
	lang=LANGS,
	x=numeric(NROWS),
	xs=numeric(NROWS),
	prob=numeric(NROWS),
)

# Build the table
for(i in seq(NROWS)){
	lang = LANGS[i]
	DB = sprintf(GRAPH_FILE, lang)
	ds = read.table(DB, header = FALSE)
	table_row(lang, ds, i)
}

#while (length(line <- readLines(gf, n = 1, warn = FALSE)) > 0) {
#	analyze(line)
#}
