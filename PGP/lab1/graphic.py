import matplotlib.pyplot as plt 
import numpy as np

f=open("tests/GPU.txt")
x = range(10000, 110000, 10000)

y = np.array([[ float(t) for t in x.split(' ')] for x in f])

for i in range(0, y.shape[1]):
	plt.plot(x, y[:,i], label = 'GPU #' + str(i + 1)) 

f=open("tests/CPU.txt")
x = range(10000, 110000, 10000)

y = np.array([float(x) for x in f])

plt.plot(x, y, label = 'CPU') 

plt.xlabel('N') 
plt.ylabel('Время') 
plt.grid(True)
plt.legend()
plt.show() 