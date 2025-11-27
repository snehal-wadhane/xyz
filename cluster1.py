import numpy as np
import math
points=np.array([ 
    [2, 10],   # P1
    [2, 5], # P2
    [8, 4],  # P3
    [5, 8], # P4
    [7, 5],   # P5
    [6, 4],  # P6
    [1, 2],  # P7
    [4, 9] ])

c1=points[0] 
c2=points[3] 
c3=points[6]
m1=[]
m2=[]
m3=[]

def dist(a,b):
   return math.sqrt((a[0]-b[0])**2+ (a[1]-b[1])**2)
def upd(cluster):
    x=sum(points[p-1][0] for p in cluster)/len(cluster)
    y=sum(points[p-1][1] for p in cluster)/len(cluster)
    return (float(x),float(y))
for i,p in enumerate(points):
    d1=dist(c1,p)
    d2=dist(c2,p)
    d3=dist(c3,p)

    if d1 < d2 and d1 < d3:
        m1.append(i)
    elif d2 < d1 and d2 < d3:
        m2.append(i)
    else:
        m3.append(i)

if 5 in m1:
    print('p6=m1')
elif 5 in m2:
    print('p6=m2')
else:
    print('p6=m3')

print(f'nm1: {upd(m1)}')
print(f'nm2: {upd(m2)}')
print(f'nm3: {upd(m3)}')
