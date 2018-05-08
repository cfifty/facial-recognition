import csv

# Gets number of lines that differ between two entered CSV files
def compareCSVs(file1, file2):
	with open(file1, 'r') as f1, open(file2, 'r') as f2:
		file1 = f1.readlines()
		file2 = f2.readlines()

		differences = 0

		for line in file1:
			if line not in file2:
				differences = differences + 1

		return differences

def main():
	print(compareCSVs('submission.csv', 'submissionknn.csv')) #../eric_code/submission.csv

if __name__ == '__main__':
	main()