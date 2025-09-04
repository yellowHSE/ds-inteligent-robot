# from setuptools import find_packages, setup

# package_name = 'turtle_parking'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=find_packages(exclude=['test']),
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='rokey',
#     maintainer_email='rokey@todo.todo',
#     description='TODO: Package description',
#     license='TODO: License declaration',
#     tests_require=['pytest'],
#     entry_points={
#         'console_scripts': [
#             'database_node = turtle_parking.database_node:main',
#             'fake_carinfo_pub = turtle_parking.fake_carinfo_pub:main',
#             'parking = turtle_parking.parking:main',
#             'nav_to_pose = turtle_parking.nav_to_pose:main',
#             'license_plate = turtle_parking.license_plate:main',
#             'fake_car_class = turtle_parking.fake_car_class:main',
#         ],
#     },
# )

from setuptools import find_packages, setup
import os

package_name = 'turtle_parking'

def rel_walk(root):
    """root 이하의 모든 파일 경로를 현재 디렉터리 기준 '상대경로'로 반환"""
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            out.append(os.path.join(dirpath, fn))
    return out

data_files = [
    # ament 인덱스 / 메타 (상대경로)
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),

    # launch 파일
    ('share/' + package_name + '/launch', ['launch/web_server.launch.py']),

    # 웹 루트 + index.html
    ('share/' + package_name + '/web', ['turtle_parking/web/index.html']),

    ('share/' + package_name + '/web', ['turtle_parking/web/customer.html']),
]

# web/assets/**/* 전체 복사 (상대경로 유지)
assets_root = 'turtle_parking/web/assets'
if os.path.isdir(assets_root):
    for f in rel_walk(assets_root):
        # f: e.g. turtle_parking/web/assets/js/app.js
        rel_parent = os.path.relpath(os.path.dirname(f), 'turtle_parking/web')  # e.g. assets/js
        dst_dir = 'share/{}/web/{}'.format(package_name, rel_parent)
        data_files.append((dst_dir, [f]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='Parking RTDB + web dashboard',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'database_node = turtle_parking.database_node:main',
            'license_plate = turtle_parking.license_plate:main',

            'web_server = turtle_parking.web_server:main',

            'robot_decision = turtle_parking.robot_decision:main',

            'exit_request_watcher = turtle_parking.exit_request_watcher:main',
            'obb_car_size = turtle_parking.obb_car_size:main',
            'navigation_lisence_test = turtle_parking.navigation_lisence_test:main',
            'car_exit = turtle_parking.car_exit:main',

            'battery_dock_monitor = turtle_parking.battery_dock_monitor:main',
            'lift_detection = turtle_parking.lift_detection:main',
            'lift_navigation = turtle_parking.lift_navigation:main',
            'person_detect = turtle_parking.person_detect:main',
        ],
    },
)