"""package"""
from setuptools import find_packages, setup

with open("README.md") as f:
    desc = f.read()

extras = {
    "dev": [
        "mercantile",
        "rasterio",
        "pytest<5.4",
        "pytest-asyncio<0.11.0",
        "pytest-cov",
        "shapely",
        "botocore==1.15.32",
        "boto3==1.12.32",
        "aioboto3",
    ]
}

setup(
    name="aiocogeo-tiler",
    description="Asynchronous cogeotiff tiler",
    long_description=desc,
    long_description_content_type="text/markdown",
    version="0.1.0",
    author=u"Jeff Albrecht",
    author_email="geospatialjeff@gmail.com",
    url="https://github.com/geospatial-jeff/aiocogeo-tiler",
    license="mit",
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="cogeo COG",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "aiocogeo==0.2.*",
        "morecantile",
        "rasterio>=1.1.7",
        "rio-tiler==2.0.0rc2",
    ],
    test_suite="tests",
    setup_requires=["pytest-runner"],
    extras_require=extras,
    tests_require=extras["dev"],
)
