import matplotlib.pyplot as plt
import json

site_file = "../dros_data/{}/{}_4D_all_sites.json"
save_path = "figures/"
save_name = save_paty + "diversity.pdf"

chroms = ["Chr2L", "Chr2R", "Chr3L", "Chr3R"]
chroms = ["Chr2L"]

for chrom in chroms:
    with open(site_file.format(chrom, chrom)) as sf:
        data = json.load(sf)
        plt.figure()
        for key in data.keys():
            plt.plot(int(key), int(data[key][0]), ",", color = "k")

plt.savefig(save_name)
