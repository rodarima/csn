all: report.pdf

report.pdf: report.tex table1.tex table2.tex table3.tex
	#pdflatex -shell-escape report.tex
	latexmk -latexoption=-shell-escape -pdf report.tex

table1.tex: table1.py languages.txt
	python table1.py

table2.tex: table2.py languages.txt
	python table2.py

table3.tex: table3.py table3.csv 
	python table3.py

table3.csv: table3.R languages.txt
	Rscript table3.R

clean:
	rm -f report.{pdf,aux,log}
