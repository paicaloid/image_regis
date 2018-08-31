import numpy as np
import surface3d

a = np.array([[1,2,4],[4,3,5]])

print (np.amax(a))
print (a)
print (a[1,0])
x,y = np.unravel_index(np.argmax(a, axis=None), a.shape)

print (x,y)

# surface3d.surface3d(10, 20)