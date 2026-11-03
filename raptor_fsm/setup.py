from setuptools import find_packages, setup

package_name = 'raptor_fsm'

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
    maintainer='Aycan Coskun',
    maintainer_email='aycancoskuun@gmail.com',
    description='Six-state sensor fusion state machine that commands ArduPilot through MAVROS.',
    license='AGPL-3.0-only',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
