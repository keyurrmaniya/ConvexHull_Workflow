from setuptools import setup

setup(
    name='convexhull_workflow',
    version='0.1.0',
    py_modules=['workflow'],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'pymatgen',
        'ase',
        'mp-api',
        'pyyaml'
    ],
    entry_points={
        'console_scripts': [
            'convex_hull=workflow:main',
        ],
    },
)
