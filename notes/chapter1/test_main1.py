import numpy as np 

la = [[4,7,4,8,],[3,0,2,3,],[1,9,1,6,],]
lb = [[4,4,],[4,5,],[2,9,],[9,9,],]

a = np.array(la)
b = np.array(lb)

print(a@b)