import numpy as np
import transforms3d as tf3d


class IntrinsicParameters():
    f_x = 3.1256205068923504e+02
    f_y = 3.1368128098467218e+02
    o_x = 2.0063402055289504e+02
    o_y = 1.5307121553844439e+02
    x_lim = 400
    y_lim = 300


class TransformationCameraToDrone():
    tx = 0.
    ty = 0.
    tz = 0.
    roll = -np.pi/2
    pitch = 0.
    yaw = np.pi/2

    def getbTc(self):
        bTc = np.zeros((4,4)) # homogeneous matrix from camera frame to drone frame
        bTc[0:3,0:3] = tf3d.euler.euler2mat(self.roll, self.pitch, self.yaw)
        bTc[0,3] = self.tx
        bTc[1,3] = self.tx
        bTc[2,3] = self.tx
        bTc[3,3] = 1.
        return bTc
    
    def getcTb(self):
        cTb = np.zeros((4,4)) # homogeneous matrix from drone frame to camera frame
        bTc = self.getbTc()
        cTb[0:3,0:3] = np.transpose(bTc[0:3,0:3])
        cTb[0:3,3] = -bTc[0:3,3]
        cTb[3,3] = 1.
        return cTb
