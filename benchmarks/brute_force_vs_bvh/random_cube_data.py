import sys
from random import *

#D=3
#N=2**24

def write_random_data_file(N,D):
    "Writes file containing N random entries containing D coordinates"
    fnm = 'random_points_%d-N_%d-D.dat' %(N,D)
    f=open(fnm, 'w')
    f.write('%d %d ' %(N,D))
    for ii in range(N):
        for jj in range(D):
            f.write('%10.8f ' %(random()))
    f.write('\n')
    f.close()
    return

syntax = "syntax: '$> "+sys.argv[0]+" <number-of-points> <(optional) number of spatial dimensions (default=1)>"

if len(sys.argv) == 2:
   if not sys.argv[1].isdigit(): print(syntax)
   else:
      N=int(sys.argv[1])
      D=1
      write_random_data_file(N,D)
elif len(sys.argv) == 3:
    if not sys.argv[1].isdigit() or not sys.argv[1].isdigit(): print(syntax)
    else:
      N=int(sys.argv[1])
      D=int(sys.argv[2])
      write_random_data_file(N,D)
else: print(syntax)

