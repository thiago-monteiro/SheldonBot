import csv

with open('bbt.csv', newline='', encoding='UTF-8') as f:
    reader = csv.reader(f)
    data = list(reader)

new_csv = []

for line in data:
    if line[2] == "Sheldon":
        for i in range(4):
            new_csv.append(line)
        
    else:
        new_csv.append(line)

with open('bbt_weighted.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerows(new_csv)
