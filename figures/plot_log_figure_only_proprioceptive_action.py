import numpy
import matplotlib.pyplot as plt

fig_convergence = plt.figure(1,figsize=(12,6))

x = numpy.loadtxt('log_supplement_only_proprioceptive_action.txt')

plt.subplot(122)
plt.plot(x[:,0])
plt.xlim([0,500])
plt.ylim([-10,200])
plt.xlabel('Steps')
plt.ylabel('Free Energy')

ax = plt.subplot(121)
plt.plot(x[:,0])
plt.ylim([-10,200])
ax.axvspan(0, 500, alpha=0.3, color='red')
plt.xlim([0,30000])
plt.xlabel('Steps')
plt.ylabel('Free Energy')

fig_convergence.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.95,
                    wspace=0.2, hspace=0.15)
                    
fig_convergence.savefig('fig_only_proprioceptive_action_convergence.pdf')
                    
plt.show()
