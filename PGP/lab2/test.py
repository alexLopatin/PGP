import sys
import random

size = int(sys.argv[1])

f = open('tests/test{0}.txt'.format(size), 'w')

f.write("{0}\n".format(size))

for i in range(0, size):
	f.write("{0} ".format(random.randint(-100000, 100000)))

f.write("\n")

for i in range(0, size):
	f.write("{0} ".format(random.randint(-100000, 100000)))