from setuptools import setup

def README():
    rtrn = b""
    with open('README.md') as f:
        rtrn = f.read()
    return rtrn

setup(
    name='recops',
    version='0.1.6',
    description='Recops',
    long_description=README(),
    author='@grey-land',
    license='Do No Harm License',
    packages=["recops"],
    scripts=[
        "bin/recops",
        "bin/recops-download-models.sh",
    ],
    package_dir={"recops":"recops"},
    package_data={
        "recops": [
            "static/*",
            "templates/*",
        ],
    },
    zip_safe=False,
    include_package_data=True,
)
