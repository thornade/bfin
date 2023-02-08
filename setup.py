import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bfin',
    version='0.0.1',
    author='thornade',
    author_email='thornade@nowhere',
    description='Handler to query ISIN information from different web ressources',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/thornade/bfin',
    project_urls = {
        "Bug Tracker": "https://github.com/thornade/bfin/issues"
    },
    license='',
    packages=['bfin'],
    install_requires=['requests','pandas','lxml','pyecharts','numpy'],
    include_package_data=True,
    package_data={'': ['CSV/*.csv']},    
)