from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = ['pyyaml', 'scikit-learn', 'pandas', 'numpy', 'tensorflow-gpu', 'opencv-python']
setup(
    name='cloogo',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Classifier test'
)