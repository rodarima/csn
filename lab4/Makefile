all: report.pdf

report.pdf: report.tex table1.tex
	#pdflatex -shell-escape report.tex
	latexmk -latexoption=-shell-escape -pdf report.tex

table1.tex: table1.py languages.txt
	python table1.py

clean:
	rm -f report.{pdf,aux,log}
