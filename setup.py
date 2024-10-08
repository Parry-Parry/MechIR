from setuptools import setup, find_packages

package_name = 'mechir'

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name=package_name,  # The name of your package
    version='0.0.1',  # Your package version
    packages=find_packages(where='src'),  # Look for packages in the 'src' directory
    package_dir={'': 'src'},  # Map the package name to the 'src' directory
    author='Anon A. Mous',
    author_email='anon@mous.com',
    description='A default python template',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    requires=required,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)