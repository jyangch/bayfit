from pathlib import Path

from setuptools import find_packages, setup

_info_ = {}
root = Path(__file__).resolve().parent
exec((root / 'bayfit' / '__info__.py').read_text(encoding='utf-8'), _info_)


readme_path = root / 'README.md'
long_description = readme_path.read_text(encoding='utf-8')


setup(
    name='bayfit',
    version=_info_['__version__'],
    description='Bayesian inference-based curve fitting for generic x/y data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jun Yang',
    author_email='jyang@smail.nju.edu.cn',
    url='https://github.com/jyangch/bayfit',
    license='GPLv3',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.20.4',
        'numba>=0.56',
        'pandas>=1.4.2',
        'scipy>=1.8.0',
        'tabulate>=0.9.0',
        'plotly>=5.22.0',
        'matplotlib>=3.2.1',
        'corner>=2.2.1',
        'getdist>=1.7.2',
        'emcee>=3.1.1',
        'lmfit>=1.0.3',
        'iminuit>=2.0.0',
    ],
    extras_require={
        'multinest': ['pymultinest>=2.12'],
    },
    packages=find_packages(exclude=['examples*', 'docs*', 'test*']),
    include_package_data=True,
    project_urls={
        'Source Code': 'https://github.com/jyangch/bayfit',
    },
)
