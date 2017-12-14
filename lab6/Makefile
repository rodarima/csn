all: report.pdf

report.pdf: report.tex table/*
	#pdflatex -shell-escape report.tex
	latexmk -latexoption=-shell-escape -latexoption=-file-line-error -pdf report.tex
	@kill -HUP $$(pidof mupdf)

table/dd.tex: fit_dd.py
	python fit_dd.py

#tableAIC.tex: fit_dd.py fitcore.py
#	python fit_dd.py


clean:
	rm -f report.{pdf,aux,log}
