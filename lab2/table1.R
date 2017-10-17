LANGS = read.table('languages.txt', header=F, stringsAsFactors=F)$V1
NROWS = length(LANGS)

# Start a empty dataframe
d = data.frame(
	lang=LANGS,
	N=numeric(NROWS),
	maxdeg=numeric(NROWS),
	M_N=numeric(NROWS),
	N_M=numeric(NROWS)
)

# Fill a row
table1_row <- function(d, lang, ds, i) {
	#d$lang[i] = lang # Doesn't work (?)
	d$maxdeg[i] = max(ds)
	N = nrow(ds)
	M = sum(ds)
	d$N[i] = N
	d$M_N[i] = M/N
	d$N_M[i] = N/M
	return(d) # Why
}

# Build the table
for(i in seq(NROWS)){
	lang = LANGS[i]
	DB = sprintf('data/%s_in-degree_sequence.txt', lang)
	ds = read.table(DB, header = FALSE)
	d = table1_row(d, lang, ds, i)
}

#print(d)

# Print the table in TeX
library(xtable)
xtable(d)
