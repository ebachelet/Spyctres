from setuptools import setup, find_packages

setup(
    name="Spyctres",
    version="0.4.0",
    description="It fits your spectra!",
    keywords='Spectroscopy',
    author="Etienne Bachelet, Yiannis Tsapras",
    license='GPL-3.0',
    url="https://github.com/ebachelet/Spyctres",
    download_url = '',
    install_requires=[
        'numpy>=1.26,<2',
        'scipy>=1.11,<1.17',
        'matplotlib',
        'astropy>=6,<8',
        'speclite',
        'pysynphot',
        'synphot>=1.1',
        'stsynphot>=1.5',
        # pysynphot imports pkg_resources at runtime.
        'setuptools<81',
    ],
    python_requires='>=3.12,<4',
    classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Developers',
		'Topic :: Software Development :: Build Tools',
                'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.12',
],
    zip_safe=False,
    packages=find_packages('.'),
    include_package_data=True,
    package_data={
        'Spyctres.data': ['*.dat', '*.fits'],
    },
    entry_points={
        'console_scripts': [
            'spyctres=Spyctres.cli:main',
        ],
    },
)
