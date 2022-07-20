from setuptools import setup

version = "0.1.dev0"

long_description = "\n\n".join(
    [
        open("README.rst").read(),
        open("changes.rst").read(),
    ]
)

install_requires=["gdal","argparse","requests","configparser","typing","cftime","netCDF4","pandas"]

tests_require = [
    "mock",
    "pytest",
    "pytest-black",
    "pytest-flakes",
    "pytest-cov",
    "pytest-mypy"
]

setup(
    name='tuflowflash',
    version = version,
    description='Code to operationally run a tuflow-flash system',
    long_description=long_description,
    classifiers=["Programming Language :: Python"],
    author='Ivar Lokhorst',
    author_email='ivar.lokhorst@rhdhv.com',
    url='https://github.com/lokhorstivar/tuflowflash',
    license="MIT",
    packages=['tuflowflash'],
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={"test": tests_require},
    entry_points={"console_scripts": ["run-tuflow-flash = tuflowflash.start_sim:main"]}
)
