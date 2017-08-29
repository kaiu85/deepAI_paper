import matplotlib.pyplot as plt
import numpy

x = numpy.arange(-1.0, 1.0, 0.001)

y = numpy.zeros( (len(x),) )
for i in range(len(x)):
    pos = x[i]
    if pos < 0.0:
        y[i] = -2*pos - 1
    else:
        y[i] = -pow(1+5*numpy.square(pos),-0.5)-numpy.square(pos)*pow(1 + 5*numpy.square(pos),-1.5)-pow(pos,4)/16.0

fig_potential = plt.figure(1,figsize=(6,6))
plt.plot(x,-0.05*0.001*numpy.cumsum(y))
plt.ylabel('Potential $G(x)$ / a.u.')
plt.xlabel('Position $x$ / a.u.')

fig_potential.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95)

fig_potential.savefig('fig_potential.pdf')

fig_force = plt.figure(2,figsize=(6,6))
plt.plot(x,0.05*y)
plt.ylabel('Force $F_g(x)$ / a.u.')
plt.xlabel('Position $x$ / a.u.')

fig_force.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95)

fig_force.savefig('fig_force.pdf')

plt.show()
        

#ifelse(T.lt(pos,0.0),-2*pos - 1,-T.pow(1+5*T.numpy.square(pos),-0.5)-T.numpy.square(pos)*T.pow(1 + 5*T.numpy.square(pos),-1.5)-T.pow(pos,4)/16.0) - 0.25*vt
