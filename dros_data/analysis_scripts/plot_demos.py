import numpy as np
import matplotlib.pyplot as plt
import json

f_name = "{}/{}_demo.folded=True.txt"
save_path = "/n/home12/efenton/for_windows/newer_2sfs/"
chroms = ["Chr2L", "Chr2R", "Chr3L", "Chr3R"]

plt.figure()
for chro in chroms:
    with open(f_name.format(chro, chro), "r") as demofile:
        data = json.load(demofile)

    sizes_plot = []
    times_plot = [0]
    for t in data["times"]:
        times_plot.append(t)
        times_plot.append(t)
    times_plot.append(1.4 * max(times_plot))
    times_plot = np.array(times_plot) / max(times_plot)
    for x in data["sizes"]:
        sizes_plot.append(x)
        sizes_plot.append(x)
    sizes_plot = np.array(sizes_plot) / max(sizes_plot)

    plt.plot(times_plot, sizes_plot, label=chro)

plt.legend()
plt.xlabel("Time [coal. units]")
plt.ylabel("Pop. size [a.u.]")
# plt.xlim([0, 1.2])
plt.savefig(save_path + "dros_demos.png")
plt.close()

