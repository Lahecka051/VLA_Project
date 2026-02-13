from setuptools import find_packages, setup

package_name = 'jetson_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/fusion.launch.py']),
        ('share/' + package_name + '/launch', ['launch/fusion.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='laheckaf',
    maintainer_email='laheckaf@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
	    'camera_yolo_node = jetson_detection.camera_yolo_node:main',
	    'viewer_node = jetson_detection.viewer_node:main',
            'imu_node = jetson_detection.imu_node:main',
            'fusion_node = jetson_detection.fusion_node:main',
        ],
    },
)
