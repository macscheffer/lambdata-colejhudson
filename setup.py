import setuptools

setuptools.setup(
    name="lambdata-colejhudson",
    version="0.0.1",
    author="Cole Hudson",
    author_email="cole@colejhudson.com",
    description="example python package",
    url="https://github.com/colejhudson/lambdata-colejhudson",
    install_requires=[
        "numpy",
        "pandas"
    ],
    packages=setuptools.find_packages(),
)