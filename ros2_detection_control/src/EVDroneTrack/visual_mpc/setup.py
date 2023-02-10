from setuptools import setup
import os
from glob import glob

package_name = 'visual_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='shiyu.liu@ls2n.fr',
    description='Visual MPC for EVDroneTrack project',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visual_mpc_node = visual_mpc.visual_mpc:main',
            'visual_simulator = visual_mpc.visual_simulator:main',
            'joystick_comm_splitter = visual_mpc.joystick_comm_splitter:main'
        ],
    },
)
