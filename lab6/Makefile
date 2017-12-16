all: report.pdf

report.pdf: report.tex tables_dd tables_dt
	#pdflatex -shell-escape report.tex
	latexmk -latexoption=-shell-escape -latexoption=-file-line-error -pdf report.tex
	@kill -HUP $$(pidof mupdf)

tables_dd: fit_dd.py fitcore.py data
	python fit_dd.py

tables_dt: fit_dt.py fitcore.py data
	python fit_dt.py

data: sim.py
	python sim.py

clean:
	rm -f report.{pdf,aux,log}


.PHONY: tables_dd tables_dt data
