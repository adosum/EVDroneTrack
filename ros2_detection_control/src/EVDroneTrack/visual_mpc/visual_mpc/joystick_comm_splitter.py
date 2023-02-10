import rclpy
from rclpy.node import Node
from ls2n_interfaces.msg import *
from ls2n_interfaces.srv import *
from std_srvs.srv import Empty
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import qos_profile_sensor_data
qos_profile_sensor_data.depth = 1

class JoystickCommSplitter(Node):
    def __init__(self):
        super().__init__('joystick_comm_splitter')
        # declare ros parameters
        comm_namespace_description = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                  description='Command Center Namespace')
        leader_namespace_description = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                  description='Leader Drone Namespace')
        follower_namespace_description = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                  description='Follower Drone Namespace')                                         
        self.declare_parameter('command_namespace', 'CommandCenter', comm_namespace_description)
        self.declare_parameter('leader_namespace', 'Drone1', leader_namespace_description)
        self.declare_parameter('follower_namespace', 'Drone2', follower_namespace_description)
        comm_namespace = self.get_parameter('command_namespace').value
        leader_namespace = self.get_parameter('leader_namespace').value
        follower_namespace = self.get_parameter('follower_namespace').value

        # set up subscriber and publishers for keepAlive
        self.create_subscription(
            KeepAlive,
            '/'+comm_namespace+'/KeepAlive',
            self.split_keep_alive_command,
            qos_profile_sensor_data
        )
        self.leader_keep_alive_publisher = self.create_publisher(
            KeepAlive,
            '/'+leader_namespace+'/KeepAlive',
            qos_profile_sensor_data
        )
        self.follower_keep_alive_publisher = self.create_publisher(
            KeepAlive,
            '/'+follower_namespace+'/KeepAlive',
            qos_profile_sensor_data
        )

        # set up service and clients for spinning motors
        self.create_service(
            Empty,
            '/'+comm_namespace+'/SpinMotors',
            self.split_spin_motors_command
        )
        self.leader_spin_motors_client = self.create_client(
            Empty,
            '/'+leader_namespace+'/SpinMotors'
        )
        self.follower_spin_motors_client = self.create_client(
            Empty,
            '/'+follower_namespace+'/SpinMotors'
        )

        # set up service and clients for starting experiments
        self.create_service(
            Empty,
            '/'+comm_namespace+'/StartExperiment',
            self.split_start_experiment_command
        )
        self.leader_start_exp_client = self.create_client(
            Empty,
            '/'+leader_namespace+'/StartExperiment'
        )
        self.follower_start_exp_client = self.create_client(
            Empty,
            '/'+follower_namespace+'/StartExperiment'
        )

        # set up service and clients for stopping experiments
        self.create_service(
            Empty,
            '/'+comm_namespace+'/StopExperiment',
            self.split_stop_experiment_command
        )
        self.leader_stop_exp_client = self.create_client(
            Empty,
            '/'+leader_namespace+'/StopExperiment'
        )
        self.follower_stop_exp_client = self.create_client(
            Empty,
            '/'+follower_namespace+'/StopExperiment'
        )

    def split_keep_alive_command(self, received_msg):
        # split keep alive commands
        msg = KeepAlive()
        msg.keep_alive = True
        msg.feedback = False
        msg.stamp = self.get_clock().now().to_msg()
        self.leader_keep_alive_publisher.publish(msg)
        self.follower_keep_alive_publisher.publish(msg)

    def split_spin_motors_command(self, request, response):
        self.leader_spin_motors_client.call_async(Empty.Request())
        self.follower_spin_motors_client.call_async(Empty.Request())
        return response

    def split_start_experiment_command(self, request, response):
        self.leader_start_exp_client.call_async(Empty.Request())
        self.follower_start_exp_client.call_async(Empty.Request())
        return response

    def split_stop_experiment_command(self, request, response):
        self.leader_stop_exp_client.call_async(Empty.Request())
        self.follower_stop_exp_client.call_async(Empty.Request())
        return response

def main(args=None):
    rclpy.init(args=args)
    joystick_comm_splitter = JoystickCommSplitter()
    try:
        rclpy.spin(joystick_comm_splitter)
    except KeyboardInterrupt:
        print("Shutting down joystick command splitter node")
    joystick_comm_splitter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()