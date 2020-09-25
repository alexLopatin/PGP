import matplotlib.pyplot as plt 
import numpy as np
import sys

step = int(sys.argv[1])

size = int(sys.argv[2])

f=open("tests/GPU.txt")
x = range(step, step + size, step)

y = np.array([[ float(t) for t in h.split(' ')] for h in f])

for i in range(0, y.shape[1]):
	plt.plot(x, y[:,i], label = 'GPU #' + str(i + 1)) 

f=open("tests/CPU.txt")

y = np.array([float(h) for h in f])

plt.plot(x, y, label = 'CPU') 

plt.xlabel('N') 
plt.ylabel('Время') 
plt.grid(True)
plt.legend()
plt.show() 