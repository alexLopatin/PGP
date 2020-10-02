import sys
import random

radius = int(sys.argv[1])

f = open('tests/test{0}.txt'.format(radius), 'w')

f.write("tests/in.data\ntests/out{0}.data\n{0}".format(radius))
