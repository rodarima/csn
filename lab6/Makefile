all: report.pdf

report.pdf: report.tex tableAIC.tex
	#pdflatex -shell-escape report.tex
	latexmk -latexoption=-shell-escape -latexoption=-file-line-error -pdf report.tex
	@kill -HUP $$(pidof mupdf)

tableAIC.tex: fit_dd.py fitcore.py
	python fit_dd.py


clean:
	rm -f report.{pdf,aux,log}
