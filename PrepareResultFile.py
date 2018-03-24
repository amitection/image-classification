

path_to_file = "../testing-results/mk10-mymodel-7layers.txt"

import csv

with open(path_to_file, 'rb') as inf, open(path_to_file+'2.txt', 'wb') as outf:
    csvreader = csv.DictReader(inf)
    fieldnames = ['imid'] + csvreader.fieldnames  # add column name to beginning
    csvwriter = csv.DictWriter(outf, fieldnames)
    csvwriter.writeheader()
    for id, row in enumerate(csvreader, 0):
        csvwriter.writerow(dict(row, imid=''+str(id)))