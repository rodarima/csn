for f in data/*; do
	echo "Preparing $f"
	sed -e "1d" $f > ${f%*.txt}.edges
done
