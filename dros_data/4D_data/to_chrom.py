import csv
import json

data = {}
data["2L"] = []
data["2R"] = []
data["3L"] = []
data["3R"] = []
data["X"] = []


with open("dmel-4Dsites.txt") as f:
    reader = csv.reader(f, delimiter = "\t")
    for line in reader:
        try:
            data[line[0]].append(int(line[1]))
        except:
            pass

with open("4d_sites.txt", "w") as f:
    json.dump(data, f)
