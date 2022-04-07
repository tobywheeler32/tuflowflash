from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='tuflowflash',
    url='https://github.com/lokhorstivar/tuflowflash',
    author='John Ladan',
    author_email='jladan@uwaterloo.ca',
    # Needed to actually package something
    packages=['tuflowflash'],
    # Needed for dependencies
    install_requires=["gdal","argparse","requests","configparser","typing","cftime","netCDF4","pandas",],
    # *strongly* suggested for sharing
    version='0.2',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code',
    entry_points={"console_scripts": ["run-tuflow-flash = tuflowflash.start_sim:main"]}
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)