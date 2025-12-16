import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'autorace_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'signs'), glob(os.path.join("signs", "*.[png]*"))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ar1sto',
    maintainer_email='ar1sto@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "follow_path = autorace_detect.follow_path:main"
        ],
    },
)
