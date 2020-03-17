from setuptools import setup, find_packages, Extension

setup(
    name="Spyctres",
    version="0.1.0",
    description="It fits your spectra!",
    keywords='Spectroscopy',
    author="Etienne Bachelet",
    author_email="etibachelet@gmail.com",
    license='GPL-3.0',
    url="https://github.com/ebachelet/Spyctres",
    download_url = '',
    install_requires=['scipy','numpy','matplotlib','astropy','speclite','pysynphot'],
    python_requires='>=3.6,<4',
    test_suite="nose.collector",
    classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Developers',
		'Topic :: Software Development :: Build Tools',
                'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Programming Language :: Python :: 3',	   
],
    zip_safe=False,
    packages=find_packages('.'),
    include_package_data=True,
    package_data={'':['H_2MASS_responses.dat','J_2MASS_responses.dat','K_2MASS_responses.dat','LBL_A10_s0_w050_R0300000_T.fits']},     
)
