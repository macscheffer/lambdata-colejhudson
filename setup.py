import setuptools

setuptools.setup(
    name="lambdata-colejhudson",
    version="0.0.3",
    author="Cole Hudson",
    author_email="cole@colejhudson.com",
    description="example python package",
    url="https://github.com/colejhudson/lambdata-colejhudson",
    install_requires=[
        "numpy",
        "pandas",
        "sklearn"
    ],
    packages=setuptools.find_packages(),
)
