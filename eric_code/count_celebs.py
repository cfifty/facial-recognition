import csv
from collections import Counter
import sys

def countCelebrities(filename):
    counter = Counter()
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counter[row['celebrity_name']] += 1
    return counter

def replaceUnknown(filename):
    counter = Counter()
    rows = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            counter[row['celebrity_name']] += 1
    least_common = counter.most_common()[-1][0]
    with open(filename, "w") as f:
        fieldnames = ['image_label', 'celebrity_name']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
             if row["celebrity_name"] == "Unknown":
                 row["celebrity_name"] = least_common
             writer.writerow(row)
       

replaceUnknown("outputs/aggregate.csv")
print("done")

#    return counter
