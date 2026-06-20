import csv

with open('bbt.csv', newline='', encoding='UTF-8') as f:
    reader = csv.reader(f)
    data = list(reader)

new_csv = []

for i in range(0, len(data) - 2, 2):
    new_csv.append({
        "instruction": data[i][1],
        "input": "",
        "output": data[i + 1][1]
    })

import json
with open('data.json', 'w') as f:
    json.dump(new_csv, f)