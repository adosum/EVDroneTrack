import rclpy
from rclpy.node import Node
import numpy as np
import math
from ls2n_interfaces.msg import *
from ls2n_interfaces.srv import *
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from std_srvs.srv import Empty
from visual_mpc.acados_settings import visual_tracking_ocp_settings
from visual_mpc.camera_settings import IntrinsicParameters, TransformationCameraToDrone
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import qos_profile_sensor_data
qos_profile_sensor_data.depth = 1


class VisualMPCParameters:
    WeightQ = np.array([    # state weights
        1,      # visual weights
        1,
        100,
        10,     # orientation weights
        10,
        10,
        10,
        5,      # linear velocity weights
        5,
        5
    ])
    WeightR = np.array([    # control weights
        1,
        5,
        5,
        5
    ])
    Tf = 0.5                # prediction horizon
    N = 50                  # discretization number
    omega_xy_max = 3.       # maximum angular rate (x,y axis)
    omega_z_max = 1.5       # maximum angular rate (z axis)
    v_max = 10.             # maximum linear velocity


class DroneVisualMPC(Node):
    def __init__(self):
        super().__init__('visual_mpc_controller')
        # declare ros parameters
        desired_depth_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                  description='Desired depth of the tracking object')
        desired_area_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                  description='Desired area of the tracking object')
        self.declare_parameter('visual_state.desired_depth', 1., desired_depth_descriptor)
        self.declare_parameter('visual_state.desired_area', 50., desired_area_descriptor)
        self.desired_depth = self.get_parameter('visual_state.desired_depth').value
        self.desired_area = self.get_parameter('visual_state.desired_area').value

        # camera parameters 
        intrinsic_param = IntrinsicParameters()
        x_lim = intrinsic_param.x_lim
        y_lim = intrinsic_param.y_lim

        camera_transform = TransformationCameraToDrone()
        cRb = camera_transform.getcTb()[0:3,0:3]

        # create client for DroneParameters
        self.drone_param_client = self.create_client(
            DroneParameters,
            'GetParameters'
        )
        
        # request drone parameters
        self.mass = 1. # drone mass
        self.max_thrust = 47. # drone maximum thrust
        self.request_params() # request from drone bridge

        # set up ocp solver
        mpc_params = VisualMPCParameters()
        x0 = np.array([x_lim/2, y_lim/2, self.desired_depth, 1., 0., 0., 0., 0., 0., 0.]) # initial state seen as desired state as well
        self.acados_solver = visual_tracking_ocp_settings(self.mass, cRb, self.desired_depth, self.desired_area, intrinsic_param, mpc_params.Tf, mpc_params.N,
            mpc_params.WeightQ, mpc_params.WeightR, x0, self.max_thrust, mpc_params.omega_xy_max, mpc_params.omega_z_max, mpc_params.v_max)
        #self.params = mpc_params

        # time constants for control
        self.Tf = mpc_params.Tf
        self.N = mpc_params.N
        self.delta_t = self.Tf/self.N  # control interval

        # state reference for each control loop
        self.x_ref = x0
        # control reference
        u_ref = np.array([9.81*self.mass,0.,0.,0.])
        # state + control reference for acados solver
        self.y_ref = np.concatenate((self.x_ref, u_ref))

        # visual state
        self.visual_state = x0[0:3]
        # coefficient for the computation of an: an = Z_star*sqrt(a_star/a) -> coeff = Z_star*sqrt(a_star)
        self.coeff_an = self.desired_depth*math.sqrt(self.desired_area)
        # timestamp
        self.visual_timestamp = -1.

        # drone state
        self.drone_attitude = np.array([1., 0., 0., 0.])
        self.drone_velocity = np.array([0., 0., 0.])

        # drone status
        self.drone_status = DroneStatus.IDLE

        # drone thrust rates command
        self.drone_command = np.array([0., 0., 0., 0.])

        # create publishers and subscribers for the states and control
        self.create_subscription(
            Odometry,
            'EKF/odom',
            self.odometry_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(
            DroneStatus,
            'Status',
            self.status_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Vector3,
            '/VisualState',
            self.visual_callback,
            qos_profile_sensor_data
        )
        self.command_publisher = self.create_publisher(
            RatesThrustSetPoint, 
            'RatesThrustSetPoint',
            qos_profile_sensor_data
        )
        # create service and client for starting/stopping the experiments
        self.create_service(
            Empty,
            'StartExperiment',
            self.start_experiment
        )
        self.start_control_client = self.create_client(
            StartControl,
            'StartControl'
        )
        self.create_service(
            Empty,
            "StopExperiment",
            self.stop_experiment
        )
        self.land_disarm_client = self.create_client(
            Empty,
            "LandDisarm"
        )

        self.controller_timer = None
        self.experiment_started = False

    def request_params(self):
        req = DroneParameters.Request()
        while not self.drone_param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('DroneParameters service not available, waiting again...')
        future = self.drone_param_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        self.mass = result.mass
        self.max_thrust = result.max_thrust

    def start_experiment(self, request, response):
        if not self.experiment_started:
            self.get_logger().info('Starting experiment')
            start_control_request = StartControl.Request()
            start_control_request.control_mode = StartControl.Request.RATES_THRUST
            self.start_control_client.call_async(start_control_request)
            self.start_time = self.get_clock().now()
            self.controller_timer = self.create_timer(self.delta_t, self.spin_controller)
            self.experiment_started = True
        else:
            self.get_logger().info("Experiment already started")
        return response

    def stop_experiment(self, request, response):
        if self.experiment_started:
            self.get_logger().info("Stopping experiment")
            self.land_disarm_client.call_async(Empty.Request())
            # Reset everything
            self.experiment_started = False
            self.destroy_timer(self.controller_timer)
        return response

    def spin_controller(self):
        if self.drone_status == DroneStatus.FLYING:
            # check visual state timestamp
            if self.visual_timestamp < 0.:
                self.get_logger().warn('Visual state not available, not flying')
                return
            t_now = self.get_clock().now().nanoseconds/1e9
            if abs(t_now - self.visual_timestamp) > 0.5:
                self.get_logger().warn('Visual state out of time, hovering')
                self.visual_state = self.x_ref[0:3]
            # update current state for each control loop
            curr_state = np.array([
                self.visual_state[0],
                self.visual_state[1],
                self.visual_state[2],
                self.drone_attitude[0],
                self.drone_attitude[1],
                self.drone_attitude[2],
                self.drone_attitude[3],
                self.drone_velocity[0],
                self.drone_velocity[1],
                self.drone_velocity[2]
            ])
            self.acados_solver.set(0, "lbx", curr_state)
            self.acados_solver.set(0, "ubx", curr_state)    
            # update reference
            for i in range(self.N):
                self.acados_solver.set(i, 'yref', self.y_ref)
                if i == (self.N-1):
                    self.acados_solver.set(self.N, 'yref', self.x_ref)
            # solve ocp by acados solver
            acados_status = self.acados_solver.solve()
            if acados_status != 0:
                self.get_logger().warn('Acados solver failed, returned status {}'.format(acados_status))      
            # send the control solution from the solver
            self.drone_command = self.acados_solver.get(0, 'u')
            self.publish_command()

        elif self.drone_status == DroneStatus.EMERGENCY_STOP:
            self.drone_command = np.array([0., 0., 0., 0.])
            self.controller_timer.destroy()
            self.experiment_started = False
        else:
            self.get_logger().warn('Drone status not flying')
            return

    def publish_command(self):
        control_msg = RatesThrustSetPoint()
        control_msg.thrust = self.drone_command[0]
        control_msg.rates[0] = self.drone_command[1]
        control_msg.rates[1] = self.drone_command[2]
        control_msg.rates[2] = self.drone_command[3]
        self.command_publisher.publish(control_msg)

    def status_callback(self, msg):
        self.drone_status = msg.status

    def odometry_callback(self, msg):
        self.drone_attitude = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ])
        self.drone_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ])
    
    def visual_callback(self, msg):
        if abs(msg.z - 0.) < 1e-7:
            self.get_logger().warn('Detected area is zero, visual feature must be lost')
            return
        an = self.coeff_an / math.sqrt(msg.z)
        self.visual_state[0] = msg.x/an
        self.visual_state[1] = msg.y/an
        self.visual_state[2] = an
        self.visual_timestamp = self.get_clock().now().nanoseconds/1e9


def main(args=None):
    rclpy.init(args=args)
    visual_mpc_controller = DroneVisualMPC()
    try:
        rclpy.spin(visual_mpc_controller)
    except (KeyboardInterrupt, RuntimeError):
        print('Shutting down drone visual mpc controller')
    finally:
        visual_mpc_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()