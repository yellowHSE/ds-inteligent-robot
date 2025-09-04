from setuptools import find_packages, setup

package_name = 'rokey_pjt'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_checker = rokey_pjt.depth_checker:main',
            'depth_checker_mouse_click = rokey_pjt.depth_checker_mouse_click:main',
            'yolo_depth_checker = rokey_pjt.yolo_depth_checker:main',
            'yolo_depth_checker_compressed = rokey_pjt.yolo_depth_checker_compressed:main',
            'tf_trans = rokey_pjt.tf_point_transform:main',
            'yolo_depth_tf = rokey_pjt.yolo_depth_tf:main',
            'object_xyz_marker = rokey_pjt.object_xyz_marker:main',
            'yolo_depth_marker = rokey_pjt.yolo_depth_marker:main',
            'yolo_depth_marker_compressed = rokey_pjt.yolo_depth_marker_compressed:main',
            'nav_to_pose = rokey_pjt.nav_to_pose:main',
            'navigation_test = rokey_pjt.navigation_test:main',
        ],
    },
)
