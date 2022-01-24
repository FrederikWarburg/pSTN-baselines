from setuptools import setup, find_packages

setup(
    name='p_stn',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        '': [
            'install_requirements.txt', '*.json'
        ]
    },
    install_requires=[],
    dependency_links=[]
)