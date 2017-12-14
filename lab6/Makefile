all: report.pdf

report.pdf: report.tex table/*
	#pdflatex -shell-escape report.tex
	latexmk -latexoption=-shell-escape -latexoption=-file-line-error -pdf report.tex
	@kill -HUP $$(pidof mupdf)

table/param_dd.tex: fit_dd.py fitcore.py
	python fit_dd.py

table/param_dt1.tex: fit_dt.py fitcore.py
	python fit_dt.py

#tableAIC.tex: fit_dd.py fitcore.py
#	python fit_dd.py

clean:
	rm -f report.{pdf,aux,log}


.PHONY: tables_dd tables_dt
