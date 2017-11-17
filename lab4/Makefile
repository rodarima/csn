all: report.pdf

report.pdf: report.tex table1.tex table_param1.tex
	#pdflatex -shell-escape report.tex
	latexmk -latexoption=-shell-escape -latexoption=-file-line-error -pdf report.tex
	@kill -HUP $$(pidof mupdf)

table1.tex: table1.py languages.txt
	python table1.py

table_param1.tex: fit.py languages.txt
	python fit.py

clean:
	rm -f report.{pdf,aux,log}
