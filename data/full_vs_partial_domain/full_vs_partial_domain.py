# Plot the relation between the size of the sub-domain and the size of his envelope 
import matplotlib.pyplot as plt

N=50

rng = range(2,N)

def sub_size(n):
#	print(n, ':', (n+2)**3 - (n-2)**3)
	return (n)**3 - (n-2)**3

full_domain = [(n**3, n**3) for n in rng ]

partial_domain = [(n**3, sub_size(n)) for n in rng ]

fig, ax = plt.subplots()

pf=plt.plot(*zip(*full_domain), label='Sous-domaine complet')
pp=plt.plot(*zip(*partial_domain), label='Enveloppe seulement')

plt.legend(title="Transfert", loc="lower center", bbox_to_anchor=(0.5,-0.3), ncol=2) #, ncol=2

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
axins = zoomed_inset_axes(ax, 100, loc=2) # zoom-factor: 2.5, location: upper-left

axins.yaxis.tick_right()

pf=axins.plot(*zip(*full_domain))
pp=axins.plot(*zip(*partial_domain))

axins.set_xlim(4, 499) # apply the x-limits
axins.set_ylim(0, 499) # apply the y-limits

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

#plt.xticks([n**3 for n in range(N)], [n**3 for n in range(N)])
ax.set_xlabel('Taille du sous-domaine')
ax.set_ylabel('Taille du transfert')

plt.show()
fig.savefig("full_vs_partial.pdf", bbox_inches='tight')