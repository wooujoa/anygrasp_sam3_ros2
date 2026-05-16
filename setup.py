from setuptools import setup
import os
from glob import glob

package_name = 'anygrasp_sam3_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jwg',
    maintainer_email='jwg@example.com',
    description='ROS 2 bridge node that runs AnyGrasp on SAM3-generated target point clouds.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'anygrasp = anygrasp_sam3_ros2.anygrasp:main',
            'anygrasp_custom = anygrasp_sam3_ros2.anygrasp_custom:main',
            'ANYGRASP = anygrasp_sam3_ros2.ANYGRASP:main',
        ],
    },
)
