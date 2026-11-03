from glob import glob

from setuptools import find_packages, setup

package_name = 'raptor_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # The ONNX model is not in git; see models/README.md.
        ('share/' + package_name + '/models', glob('models/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aycan Coskun',
    maintainer_email='aycancoskuun@gmail.com',
    description='YOLOv5n bird detection node with N-of-M confirmation.',
    license='AGPL-3.0-only',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bird_detector_node = raptor_perception.bird_detector_node:main',
            'bird_range_sweep = raptor_perception.bird_range_sweep:main',
        ],
    },
)
