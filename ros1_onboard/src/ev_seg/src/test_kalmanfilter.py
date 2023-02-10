# import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

x = np.array(range(10))
y = np.array(range(10))
area = np.array(range(10))


x = x + np.random.random(10)
y = y + np.random.random(10)
area = area + 30
area = area + np.random.random(10)


state = []
kalman = KalmanFilter(6,3)
kalman.x = np.array([1,1,30,0,0,0])
kalman.F = np.array([[1,0,0,1,0,0], [0,1,0,0,1,0], [0,0,1,0,0,1], [0, 0,0, 1,0, 0], [0, 0, 0,0,1, 0], [0, 0,0 ,0,0, 1]])
kalman.Q = 0.1 * np.eye(6) #Q_discrete_white_noise(dim=6, dt=0.1, var=0.13)
kalman.Q[2,2] *= 5
kalman.Q[5,5] *= 5
kalman.H = np.array([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.]])
kalman.P *= 1.
kalman.R *= 1.
for i in range(10):
    est = np.array([x[i],y[i],area[i]])
    kalman.predict()
    kalman.update(est)
    state.append(kalman.x[0:3])


state = np.array(state)

plt.figure('1')
plt.plot(state[:,0],state[:,1],'*')
plt.plot(x,y,'.')

plt.figure('2')
plt.plot(range(10),state[:,2],'*')
plt.plot(range(10),area,'*')
plt.show()
