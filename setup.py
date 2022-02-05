from setuptools import setup

package_list = open('./requirements.txt').readlines()

setup(
    name='climbing_thing',
    version='0.1',
    description='Climbing computer vision project',
    url='https://github.com/DoubleTimeOnly/Climbing-Thing',
    packages=package_list
)
