from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['PYSYN_CDBS'] = '/home/ebachelet/cdbs/'


import scipy.optimize as so
from scipy.stats import rv_continuous
from matplotlib.patches import Ellipse
import speclite.filters
import speclite

import astropy.units as u
import astropy.constants as constantes

from Spyctres import Spyctres


spectra = {}
uas_to_rad = 180*3600*10**6/np.pi
solar_radii_to_parsec = 2.254*10**-8


Spyctres.define_2MASS_filters()
Spyctres.define_GAIA_filters()
twomass_filters = speclite.filters.load_filters('MASS-J', 'MASS-H','MASS-K')
sdss_filters = speclite.filters.load_filters('sdss2010-*')
bessel_filters = speclite.filters.load_filters('bessell-*')
gaia_filters = speclite.filters.load_filters('GAIA-G')




#Adding new spectra

spectra = {}
SEDS=[]
ABS = []
magnifications = []

# SALT, time obs 06/06/2020, magnification ~ 1.8
magnification = 1.8
spectrum_SALT1 = np.loadtxt('/home/ebachelet/Work/Microlensing/OMEGA/Gaia20bof/Spectra/SALT_06_06_2020.dat')
spectrum_SALT1 = np.c_[spectrum_SALT1,[10**-16]*len(spectrum_SALT1)]



wave_obs = [7472,4670,6405]#,12355,16458,21603]#Angstrom
#wave_obs = [5536.14809278,8086.39904467,6773.70451294]
ab_corrections = Spyctres.derive_AB_correction([sdss_filters[1],sdss_filters[3],gaia_filters[0]])
#ab_corrections = [0.37,-0.08,0.1]#,0.91,1.39,1.85] #http://www.astro.osu.edu/~martini/usefuldata.html
obs_mags = [15,16,15.15]#,11.849-2.5*np.log10(magnification),10.862-2.5*np.log10(magnification),10.513-2.5*np.log10(magnification)] # J,H,K are faked from Vizier
obs_emags = [0.1,0.1,0.1]#,0.1,0.1]
filters = ['SDSS_i','SDSS_g','GAIA_G']#,'MASS_J','MASS_H','MASS_K']


sed = np.c_[wave_obs,obs_mags,obs_emags,filters]
SEDS.append(sed)
ABS.append(ab_corrections)

fake_spectrum = Spyctres.star_spectrum( 5.28401032e+03, -7.60922125e-01,
        3.48377479e+00,catalog='k93models')
wave = np.array(fake_spectrum.wave) #Angstrom 
step = 50 #Angstrom
global_mask = (wave>spectrum_SALT1[0,0]+step) & (wave<spectrum_SALT1[-1,0]-step)
wave = wave[global_mask]


bin_spec,cov = Spyctres.bin_spectrum(spectrum_SALT1,np.array(wave))
SNR = bin_spec[:,1]/bin_spec[:,2]
#offset1 = 10**(offset1(bin_spec[:,0])/2.5)
data_fit = np.c_[bin_spec[:,0],bin_spec[:,1],bin_spec[:,1]/SNR]
data_fit1 = np.c_[bin_spec[:,0],bin_spec[:,1],bin_spec[:,1]/SNR]


SED_flux,SED_mag = Spyctres.derive_SED_from_obs_mag(wave_obs,obs_mags,obs_emags,ab_corrections,filters)
offset1,eoffset1,quant1,equant1 = Spyctres.SED_offset(SED_mag,bin_spec,bessel_filters,sdss_filters,twomass_filters,gaia_filters)
spectra['SALT_06_06_2020'] = data_fit
magnifications.append(magnification)


#Find telluric lines

telluric_lines = Spyctres.load_telluric_lines()
telluric_lines,telluric_mask = Spyctres.telluric_lines(telluric_lines,fake_spectrum.wave,threshold=0.95)

# Plot
plt.yscale('log')
plt.errorbar(data_fit[:,0],data_fit[:,1],data_fit[:,2],fmt='.',label='SALT')
plt.fill_between(telluric_lines[:,0],0,2*data_fit[:,1].max(),where=telluric_mask,color='grey',alpha=0.25)
plt.show()
import pdb; pdb.set_trace()



