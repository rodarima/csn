all: report.pdf

report.pdf: ws.pdf er.pdf
	pdflatex report.tex

ws.pdf: ws.R
	Rscript ws.R

er.pdf: er.R
	Rscript er.R

clean:
	rm -f ws.pdf er.pdf report.pdf report.aux report.log
