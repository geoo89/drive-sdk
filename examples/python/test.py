import numpy as np




X = np.array([
[1,0,0,0, 0],
[1,1,0,0,10],
[1,0,1,0,20],
[1,0,0,1,30],
[1,0,0,0,40],
[1,0,0,0,50],
[1,0,0,0,60]])

X = np.array([
[1, 3.1,   2,   1,  0],
[1,   7, 5.1,   3,  1],
[1,  11,   8, 5.1,  2],
[1,  15,  11,   7,3.1],
[1,19.1,  14,   9,  4],
[1,  23,17.1,  11,  5],
[1,  27,  20,14.1,  6]])


print(np.linalg.matrix_rank(X))
y = np.array([10,20,30,40,50,60,70])
print(X)
print(y)
a, stuff1, stuff2, stuff3 = np.linalg.lstsq(X, y)
print(a)
u0 = np.array([a[1], a[2], a[3], a[4], -1]) # normal vector
u1 = np.array([-a[2], a[1], 0, 0, 0])
u2 = np.array([0, -a[3], a[2], 0, 0])
u3 = np.array([0, 0, -a[4], a[3], 0])
u4 = np.array([0, 0, 0,     1, a[4]])

A = np.transpose(np.array([u0, u1, u2, u3, u4]))
q, r = np.linalg.qr(A)
print(np.transpose(q))
d = np.transpose(q)[4]
print("reality check:")
print(d)
print(a[1] * d[0] +  a[2] * d[1] + a[3] * d[2] + a[4] * d[3] - d[4], -a[0])

