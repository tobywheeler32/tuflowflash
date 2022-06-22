from setuptools import setup

setup(
    name='tuflowflash',
    url='https://github.com/lokhorstivar/tuflowflash',
    author='Ivar Lokhorst',
    author_email='ivar.lokhorst@rhdhv.com',
    packages=['tuflowflash'],
    install_requires=["gdal","argparse","requests","configparser","typing","cftime","netCDF4","pandas"],#,"pyproj==3.3"],
    version='0.1',
    license='MIT',
    description='Code to operationally run a tuflow-flash system',
    entry_points={"console_scripts": ["run-tuflow-flash = tuflowflash.start_sim:main"]}
    # long_description=open('README.txt').read(),
)
