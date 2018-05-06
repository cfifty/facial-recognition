import csv

# Gets number of lines that differ between two entered CSV files
def compareCSVs(file1, file2):
	with open('file1.csv', 'r') as f1, open('file2.csv', 'r') as f2:
    file1 = f1.readlines()
    file2 = f2.readlines()

    differences = 0

	for line in f2:
		if line not in f1:
			differences = differences + 1

	return differences