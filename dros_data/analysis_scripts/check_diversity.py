import numpy as np
import matplotlib.pyplot as plt
import json

save_path = "/n/home12/efenton/for_windows/newer_2sfs/Chr2L/" 

with open("Chr2L_sites.json") as sf:
    site_data = json.load(sf)

print(len(site_data.keys()))

'''
plt.figure()
for (key, item) in site_data.items():
    plt.plot(int(key), int(item), ",", color="k")

plt.savefig(save_path + "diversity.png")
plt.close()
'''
