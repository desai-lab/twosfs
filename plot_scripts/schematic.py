import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
from twosfs.statistics import sample_twosfs, sample_spectra
import json
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib import patches, lines

config = configuration_from_json("../simulation_parameters.json")
save_path = "figures/"

beta_params = '"alpha":{}'

demo_file = '../simulations/fitted_demographies/model={}.params={{{}}}.folded=True.txt'
rec_file = '../simulations/recombination_search/model={}.params={{{}}}.folded=True.pair_density=10000.sequence_length=25.power_rep={}.hdf5'
sim_file = '../simulations/initial_spectra/model={}.params={{{}}}.rep=all.hdf5'
ks_file = '../simulations/ks_distances/model={}.params={{{}}}.folded=True.pair_density=10000.sequence_length=25.power_rep={}.hdf5'

def demo_to_plot(sizes, times):
    t = [0]
    for time in times:
        t.append(time)
        t.append(time)
    t.append(t[-1] * 10)
    t = np.array(t) / t[-1]
    y = []
    for size in sizes:
        y.append(size)
        y.append(size)
    return y, t

def lower_subplot(ax, lower, upper):
    x0 = ax.get_position().x0
    x1 = ax.get_position().x1
    y0 = ax.get_position().y0
    y1 = ax.get_position().y1
    dx = x1 - x0
    dy = y1 - y0 - upper
    ax.set_position([x0, y0 - lower, dx, dy])

def center_subplot(ax, squish):
    x0 = ax.get_position().x0
    x1 = ax.get_position().x1
    y0 = ax.get_position().y0
    y1 = ax.get_position().y1
    dx = (x1 - x0) * squish
    d = (x1 - x0) - dx
    dy = (y1 - y0)
    ax.set_position([x0 + d/2, y0, dx, dy])

######## Loading ########

alpha = 1.5
params = ("beta", beta_params.format(alpha))
with open(demo_file.format(*params)) as df:
    beta_data = json.load(df)

params = ("beta", beta_params.format(alpha), 0)
with h5py.File(rec_file.format(*params)) as hf:
    ks_d = dict(hf.get("spectra_high").attrs)["ks_distance"]
    spec_growth = spectra_from_hdf5(hf.get("spectra_high"))
    two_growth = spec_growth.normalized_twosfs(folded=True, k_max=20)[:,1:,1:]
spec_beta = load_spectra(sim_file.format(*params[:-1]))
two_beta = spec_beta.normalized_twosfs(folded=True, k_max=20)[:,1:,1:]

with h5py.File(ks_file.format(*params)) as hf:
    ks = np.array(hf.get("ks_null"))

pd = np.full(25, 100000)
spec_resamp = sample_spectra(spec_growth, 1, pd, False)
two_resamp1 = spec_resamp.normalized_twosfs(folded=True, k_max=20)[:,1:,1:]
spec_resamp = sample_spectra(spec_growth, 1, pd, False)
two_resamp2 = spec_resamp.normalized_twosfs(folded=True, k_max=20)[:,1:,1:]
spec_resamp = sample_spectra(spec_growth, 1, pd, False)
two_resamp3 = spec_resamp.normalized_twosfs(folded=True, k_max=20)[:,1:,1:]
spec_resamp = sample_spectra(spec_growth, 1, pd, False)
two_resamp4 = spec_resamp.normalized_twosfs(folded=True, k_max=20)[:,1:,1:]

######## Plotting ########

fig = plt.figure(figsize=(7.2, 3.9))
outer = gridspec.GridSpec(2, 4, wspace=0.55, hspace=0.45, left=0.07, right=0.98, bottom=0.09, top=0.95)
axs = [plt.Subplot(fig, outer[i]) for i in [0, 1, 2, 3, 7, 6, 5, 4]]

inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.4)
saxs = [plt.Subplot(fig, inner[i]) for i in range(2)]
center_subplot(saxs[0], 0.5)
center_subplot(saxs[1], 0.5)
axs[1].axis("off")

inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], hspace=0.4)
axs[2] = plt.Subplot(fig, inner[0])

inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[3], wspace=0.1, hspace=0.4)
raxs = [plt.Subplot(fig, inner[i]) for i in range(4)]
for side in ["top", "bottom", "left", "right"]:
    axs[3].spines[side].set_visible(False)
axs[3].set_xticks([])
axs[3].set_yticks([])

d = 0.03
lower_subplot(raxs[0], d, d)
lower_subplot(raxs[1], d, d)
lower_subplot(raxs[2], 0, d)
lower_subplot(raxs[3], 0, d)

lower_subplot(saxs[1], d, d)

inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[5], wspace=0.1, hspace=0.4)
paxs = [plt.Subplot(fig, inner[i]) for i in range(4)]
axs[6].axis("off")

lower_subplot(paxs[0], d, d)
lower_subplot(paxs[1], d, d)
lower_subplot(paxs[2], 0, d)
lower_subplot(paxs[3], 0, d)

[fig.add_subplot(ax) for ax in axs]
[fig.add_subplot(ax) for ax in saxs]
[fig.add_subplot(ax) for ax in paxs]
[fig.add_subplot(ax) for ax in raxs]

# Collect data
axs[0].axis("off")
axs[0].set_xlim(0,1)
axs[0].set_ylim(0,1)
poly = [4, 7, 10]
npoly = [i for i in range(13) if i not in poly]
for i, c in zip(npoly, "ACTGTATGAC"):
    for j in range(3, 10):
        axs[0].text(i/13, j/10, c, fontsize=7, transform = axs[0].transAxes, ha="center", va="center")

c0 = "lightcoral"
for j in range(3, 6):
    axs[0].text(4/13, j/10, "G", fontsize=7, transform = axs[0].transAxes, ha="center", va="center",
                color=c0)
for j in range(6, 10):
    axs[0].text(4/13, j/10, "C", fontsize=7, transform = axs[0].transAxes, ha="center", va="center")


axs[0].text(7/13, 6/10, "T", fontsize=7, transform = axs[0].transAxes, ha="center", va="center", color=c0)
for j in range(3, 6):
    axs[0].text(7/13, j/10, "A", fontsize=7, transform = axs[0].transAxes, ha="center", va="center")
for j in range(7, 10):
    axs[0].text(7/13, j/10, "A", fontsize=7, transform = axs[0].transAxes, ha="center", va="center")

axs[0].text(10/13, 3/10, "A", fontsize=7, transform = axs[0].transAxes, ha="center", va="center", color=c0)
axs[0].text(10/13, 7/10, "A", fontsize=7, transform = axs[0].transAxes, ha="center", va="center", color=c0)
for j in [4,5,6,8,9]:
    axs[0].text(10/13, j/10, "T", fontsize=7, transform = axs[0].transAxes, ha="center", va="center")

c1 = "#545E75"
axs[0].arrow(5/13, 0.25, 1.8/13, 0, color=c1, length_includes_head=True, head_width=0.006, head_length=0.01)
axs[0].arrow(6/13, 0.25, -1.8/13, 0, color=c1, length_includes_head=True, head_width=0.006, head_length=0.01)
axs[0].text(5.5/13, 0.19, r"$\varphi_{1,3}(d=3)=1$", fontsize=6, transform=axs[0].transAxes, ha="center", va="center",
            color=c1)

c2 = "#3F826D"
axs[0].arrow(6/13, 0.13, 3.8/13, 0, color=c2, length_includes_head=True, head_width=0.006, head_length=0.01)
axs[0].arrow(8/13, 0.13, -3.8/13, 0, color=c2, length_includes_head=True, head_width=0.006, head_length=0.01)
axs[0].text(7/13, 0.07, r"$\varphi_{2,3}(d=6)=1$", fontsize=6, transform=axs[0].transAxes, ha="center", va="center",
            color=c2)

c3 = "#4BC6B9"
axs[0].arrow(8/13, 0.01, 1.8/13, 0, color=c3, length_includes_head=True, head_width=0.006, head_length=0.01)
axs[0].arrow(9/13, 0.01, -1.8/13, 0, color=c3, length_includes_head=True, head_width=0.006, head_length=0.01)
axs[0].text(8.5/13, -0.05, r"$\varphi_{1,2}(d=3)=1$", fontsize=6, transform=axs[0].transAxes, ha="center", va="center",
            color=c3)

axs[0].set_title("Collect data", fontsize=7)

# Construct SFS
saxs[0].loglog(np.arange(19) + 1, np.array(beta_data["sfs_obs"])[:-1], ".", color="k", ms=5)
saxs[0].set_title("Construct the SFS", fontsize=7)
saxs[0].set_xlabel("Site freq.", fontsize=6)
saxs[0].set_ylabel("Frac. of sites", fontsize=6)
saxs[1].pcolormesh( np.log2(two_beta[0,:-1,:-1]), cmap="Purples")
saxs[1].set_title("Construct the 2-SFS", fontsize=7)
saxs[1].set_xlabel("Freq. site 1", fontsize=6)
saxs[1].set_ylabel("Freq. site 2", fontsize=6)

# Fit a Kingman demography
y, t = demo_to_plot(beta_data["sizes"], beta_data["times"])
axs[2].loglog(t, y, color="k")
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[2].set_xlabel("Time in past", fontsize=6)
axs[2].set_ylabel("Pop. size", fontsize=6)
axs[2].set_title("Fit a Kingman demography", fontsize=7)

# Simulate several rec. rates
raxs[0].pcolormesh( np.log2(two_growth[2,:-1,:-1]), cmap="Purples")
raxs[1].pcolormesh( np.log2(two_growth[5,:-1,:-1]), cmap="Purples")
raxs[2].pcolormesh( np.log2(two_growth[9,:-1,:-1]), cmap="Purples")
raxs[3].pcolormesh( np.log2(two_growth[16,:-1,:-1]), cmap="Purples")
raxs[0].set_title(r"$\varphi_{i,j}(r = 0.15)$", fontsize=6)
raxs[1].set_title(r"$\varphi_{i,j}(r = 0.16)$", fontsize=6)
raxs[2].set_title(r"$\varphi_{i,j}(r = 0.17)$", fontsize=6)
raxs[3].set_title(r"$\varphi_{i,j}(r = 0.18)$", fontsize=6)
axs[3].set_title("Simulate several rec. rates", fontsize=7)
axs[3].set_xlabel("Frequency site 1", fontsize=7)
axs[3].set_ylabel("Frequency site 2", fontsize=7)

# Measure KS distance
X = np.arange(1, 21)
cdf_beta = np.cumsum(np.cumsum(two_beta[0], axis=0), axis=1)[0]
cdf_growth = np.cumsum(np.cumsum(two_resamp1[20], axis=0), axis=1)[0]
cdf_beta = np.insert(cdf_beta, 0, 0)
cdf_growth = np.insert(cdf_growth, 0, 0)
x = np.zeros(20)
y = np.zeros(20)
np.random.seed(320549)
for i in range(19):
    x[i+1] = x[i] + np.random.random()+i/10
    y[i+1] = y[i] + np.random.random()
x /= np.max(x)
y /= np.max(y)
axs[4].step(X, x, label="Null CDF", color=c1)
axs[4].step(X, y, label="Data CDF", color=c2)
axs[4].arrow(6.5, x[6]+0.10, 0, y[6] - x[6] - 0.12, color=c0, length_includes_head=True,
             head_width=0.1, head_length=0.006)
axs[4].arrow(6.5, y[6]-0.10, 0, x[6] - y[6] + 0.12, color=c0, length_includes_head=True,
             head_width=0.1, head_length=0.006)
axs[4].text(0.37, 0.16, "KS distance", fontsize=6, transform = axs[4].transAxes, color=c0)
axs[4].legend(fontsize=6)
axs[4].set_xticks([0, 10, 20])
axs[4].set_title("Measure KS distances", fontsize=7)

# Choose rec. rate that minimizes KS distance
axs[5].plot([0.14, 0.15, 0.16, 0.17, 0.18], [1215, 864, 597, 796, 1052], ".", color="k")
axs[5].set_xlabel("Recombination rate", fontsize=7)
axs[5].set_ylabel("KS distance", fontsize=7)
axs[5].text(0.35, 0.2, r"$\hat{r}$", fontsize=6, transform = axs[5].transAxes, color=c2)
axs[5].arrow(0.37, 0.18, 0.10, -0.09, color=c2, transform=axs[5].transAxes)
axs[5].set_title("Choose rec. rate that\nminimizes KS distance", fontsize=7)

# Resample the 2-SFS
paxs[0].pcolormesh( np.log2(two_resamp1[0] / two_growth[0]), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
paxs[1].pcolormesh( np.log2(two_resamp2[0] / two_growth[0]), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
paxs[2].pcolormesh( np.log2(two_resamp3[0] / two_growth[0]), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
paxs[3].pcolormesh( np.log2(two_resamp4[0] / two_growth[0]), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
paxs[0].set_title(r"$\varphi_{ij}^{resamp;1}$", fontsize=6)
paxs[1].set_title(r"$\varphi_{ij}^{resamp;2}$", fontsize=6)
paxs[2].set_title(r"$\varphi_{ij}^{resamp;3}$", fontsize=6)
paxs[3].set_title(r"$\varphi_{ij}^{resamp;4}$", fontsize=6)
axs[6].set_title("Resample the 2-SFS", fontsize=7)

# Generate null KS dist. and p-value
p = round(sum((ks > ks_d) / len(ks)), 3)
axs[7].hist(ks, 100, color=c1, label="Null KS distribution")
axs[7].plot([ks_d], [200], "*", color=c2, label="Data KS value")
axs[7].legend(fontsize=6)
axs[7].set_xlabel("KS distance", fontsize=7)
axs[7].set_ylabel("Counts", fontsize=7)
axs[7].set_ylim(0, 470)
axs[7].text(0.8, 0.55, f"p={p}", fontsize=6, color=c2, transform=axs[7].transAxes, ha="center", va="center")
axs[7].set_title("Generate null KS\n" + r"dist. and $p$-value", fontsize=7)

# Format axes
[ax.tick_params(labelsize=6) for ax in axs]
[ax.set_xticks([]) for ax in raxs]
[ax.set_yticks([]) for ax in raxs]
[ax.set_xticks([]) for ax in paxs]
[ax.set_yticks([]) for ax in paxs]
[ax.set_xticklabels([]) for ax in axs]
[ax.set_yticklabels([]) for ax in axs]
[ax.set_xticklabels([]) for ax in saxs]
[ax.set_yticklabels([]) for ax in saxs]
saxs[1].set_xticks([])
saxs[1].set_yticks([])

# Draw the arrows
def draw_arrow(ax1, ax2, direc, buff1=0.01, buff2=0.01):
    if direc == "lr":
        x0 = ax1.get_position().x1 + buff1
        x1 = ax2.get_position().x0 - buff2
        y0 = (ax1.get_position().y1 - ax1.get_position().y0) / 2 + ax1.get_position().y0
        y1 = (ax2.get_position().y1 - ax2.get_position().y0) / 2 + ax2.get_position().y0
    elif direc == "rl":
        x0 = ax1.get_position().x0 - buff1
        x1 = ax2.get_position().x1 + buff2
        y0 = (ax1.get_position().y1 - ax1.get_position().y0) / 2 + ax1.get_position().y0
        y1 = (ax2.get_position().y1 - ax2.get_position().y0) / 2 + ax2.get_position().y0
    elif direc == "ud":
        y0 = ax1.get_position().y0 - buff1
        y1 = ax2.get_position().y1 + buff2
        x0 = (ax1.get_position().x1 - ax1.get_position().x0) / 2 + ax1.get_position().x0
        x1 = (ax2.get_position().x1 - ax2.get_position().x0) / 2 + ax2.get_position().x0
    arrow = patches.FancyArrowPatch([x0, y0], [x1, y1], transform=fig.transFigure, 
                    color="tab:red", arrowstyle="Simple", shrinkA=0, shrinkB=0,
                    linewidth=0.5, mutation_scale=12)
    fig.patches.append(arrow)

draw_arrow(axs[0], saxs[0], "lr", 0.01, 0.037)
draw_arrow(axs[0], saxs[1], "lr", 0.01, 0.037)
draw_arrow(saxs[0], axs[2], "lr", 0.01, 0.035)
draw_arrow(axs[2], axs[3], "lr", 0.01, 0.035)
draw_arrow(axs[3], axs[4], "ud", 0.06, 0.05)
draw_arrow(axs[4], axs[5], "rl")
draw_arrow(axs[5], axs[6], "rl", 0.04)
draw_arrow(axs[6], axs[7], "rl")

# Bendy arrow
x0 = saxs[1].get_position().x1 + 0.01
x1 = raxs[2].get_position().x0 - 0.03
y0 = (saxs[1].get_position().y1 - saxs[1].get_position().y0) / 2 + saxs[1].get_position().y0
a1 = lines.Line2D([x0, x1], [y0, y0], transform=fig.transFigure, color="tab:red", linewidth=3)
fig.patches.append(a1)

y1 = 2 * ( axs[3].get_position().y0 - axs[4].get_position().y1 ) / 3 + axs[4].get_position().y1 - 0.01
a2 = lines.Line2D([x1, x1], [y0, y1], transform=fig.transFigure, color="tab:red", linewidth=3)
fig.patches.append(a2)

x2 = axs[4].get_position().x0 + 0.05
a3 = lines.Line2D([x1, x2], [y1, y1], transform=fig.transFigure, color="tab:red", linewidth=3)
fig.patches.append(a3)

y2 = axs[4].get_position().y1 + 0.05
a4 = patches.FancyArrowPatch([x2, y1], [x2, y2], transform=fig.transFigure, 
             color="tab:red", arrowstyle="Simple", shrinkA=0, shrinkB=0,
             linewidth=0.5, mutation_scale=12)
fig.patches.append(a4)

for ax, l in zip(axs, "abcdefgh"):
    ax.text(-0.18, 1.00, l+".", fontweight = "bold", transform = ax.transAxes, fontsize=7)

fig.savefig(save_path + "schematic.pdf")

