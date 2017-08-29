import numpy
import matplotlib.pyplot as plt

x = numpy.loadtxt('log_deepAI_paper.txt')

plt.plot(x[:,0])
plt.xlabel('Steps')
plt.ylabel('Free Energy')
plt.show()

