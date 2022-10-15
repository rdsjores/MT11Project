from setuptools import setup

setup(
    name="flightCollection",
    version="0.1",
    author="Robin de Saint Jores, Vianney Ville",
    author_email="robindesaintjores@gmail.com.com",
    description="tool to exploit flight data",
    long_description=open("readme.txt").read(),
    license="MIT",
    packages=[
        "flightCollection",
    ],  # folder name
    install_requires=[
        "pandas>=1.4.2",
        "numpy>=1.4.2",
    ],
)
