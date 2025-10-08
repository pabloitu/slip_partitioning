import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = plt.get_cmap("magma")
norm = mpl.colors.Normalize(vmin=0, vmax=1)

cb = mpl.colorbar.ColorbarBase(
    ax,
    cmap=cmap,
    norm=norm,
    orientation='horizontal'
)

# Remove ticks and labels
cb.set_ticks([])
cb.set_ticklabels([])
cb.outline.set_visible(False)

plt.savefig("magma_bar.png", dpi=300, bbox_inches='tight', transparent=True)
