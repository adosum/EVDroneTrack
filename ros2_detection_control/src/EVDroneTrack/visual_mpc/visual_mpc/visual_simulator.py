import rclpy
from rclpy.node import Node
import numpy as np
import transforms3d as tf3d
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from visual_mpc.camera_settings import IntrinsicParameters, TransformationCameraToDrone
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import qos_profile_sensor_data
qos_profile_sensor_data.depth = 1


class VisualSimulator(Node):
    def __init__(self):
        super().__init__('visual_simulator')
        comm_namespace_description = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                  description='Command Center Namespace')
        target_namespace_description = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                  description='Tracking Target Namespace')
        drone_namespace_description = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                  description='Drone Namespace')
        simulated_area_ratio_description = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                  description='Simulated area ratio relative to depth')
        self.declare_parameter('command_namespace', 'CommandCenter', comm_namespace_description)
        self.declare_parameter('target_namespace', 'Drone1', target_namespace_description)
        self.declare_parameter('drone_namespace', 'Drone2', drone_namespace_description)
        self.declare_parameter('area_ratio', 50., simulated_area_ratio_description)
        comm_namespace = self.get_parameter('command_namespace').value
        target_namespace = self.get_parameter('target_namespace').value
        drone_namespace = self.get_parameter('drone_namespace').value
        self.area_ratio = self.get_parameter('area_ratio').value

        # odometry
        self.target_odom = np.array([0.,0.,0.,1.,0.,0.,0.])
        self.drone_odom = np.array([0.,0.,0.,1.,0.,0.,0.])

        # camera intrinsic matrix
        intrinsic_param = IntrinsicParameters()
        self.intrinsic_matrix = np.zeros((3,3))
        self.intrinsic_matrix[0,0] = intrinsic_param.f_x
        self.intrinsic_matrix[1,1] = intrinsic_param.f_y
        self.intrinsic_matrix[0,2] = intrinsic_param.o_x
        self.intrinsic_matrix[1,2] = intrinsic_param.o_y
        self.intrinsic_matrix[2,2] = 1.

        self.x_lim = intrinsic_param.x_lim
        self.y_lim = intrinsic_param.y_lim

        # camera fixed transformation relative to the drone frame
        camera_transform = TransformationCameraToDrone()
        self.cTb = camera_transform.getcTb()

        # set up subscribers to odometry
        self.create_subscription(
            Odometry,
            '/'+target_namespace+'/EKF/odom',
            self.target_odom_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry,
            '/'+drone_namespace+'/EKF/odom',
            self.drone_odom_callback,
            qos_profile_sensor_data
        )
        # create publisher for simulated visual feastures
        self.visual_state_publisher = self.create_publisher(
            Vector3,
            '/VisualState',
            qos_profile_sensor_data
        )
        # create service to start/stop publishing the simulated visual state
        self.create_service(
            Empty,
            '/'+comm_namespace+'/StartExperiment',
            self.create_visual_publisher
        )
        self.create_service(
            Empty,
            '/'+comm_namespace+'/StopExperiment',
            self.destroy_visual_publisher
        )
        self.experiment_started = False
        self.publiser_timer = None

    def publish_visual_state(self):
        wPbt = self.target_odom[0:3] - self.drone_odom[0:3] # vector DT (drone body to target) expressed in world frame
        bPt = tf3d.quaternions.quat2mat(self.drone_odom[3:7]).dot(wPbt) # target position expressed in drone's frame
        cPt = self.cTb[0:3,0:3].dot(bPt) + self.cTb[0:3,3]  # target position expressed in camera's frame
        pt = self.intrinsic_matrix.dot(cPt)  # augmented image point
        if abs(pt[2]-0.) < 1e-7:
            self.get_logger().warn('Image depth is incorrect')
            return
        px = pt[0]/pt[2]
        py = pt[1]/pt[2]
        if px < 0. or px > self.x_lim or py < 0. or py > self.y_lim:
            self.get_logger().warn('Detected point is not in the field of view')
            return
        msg = Vector3()
        msg.x = px
        msg.y = py
        msg.z = self.area_ratio/pt[2]
        self.visual_state_publisher.publish(msg)

    def target_odom_callback(self, msg):
        self.target_odom = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ])

    def drone_odom_callback(self, msg):
        self.drone_odom = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ])

    def create_visual_publisher(self, request, response):
        if not self.experiment_started:
            self.publisher_timer = self.create_timer(0.04, self.publish_visual_state)
            self.experiment_started = True
        return response
    
    def destroy_visual_publisher(self, request, response):
        if self.experiment_started:
            self.destroy_timer(self.publiser_timer)
            self.experiment_started = False
        return response

def main(args=None):
    rclpy.init(args=args)
    visual_simulator = VisualSimulator()
    try:
        rclpy.spin(visual_simulator)
    except KeyboardInterrupt:
        print("Shutting down visual simulator node")
    visual_simulator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()