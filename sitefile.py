import json
import sys

datafile = "holder.txt"
savefile = "/path/to/working/directory/sites.json"

n = 0 # Haploid sample size, or 2*diploid sample size
data = {}

# Reads the temporary file "holder.txt" and reformats it to be readable by 2-SFS
with open(datafile) as f:
    for line in f:
        x = line.split()
        data[int(x[1])] = [int(x[5]), n-int(x[4]), [x[3]]]

with open(savefile, "w") as f:
    json.dump(data, f)


