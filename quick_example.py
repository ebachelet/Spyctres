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

import Spyctres

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

   
    if ax is None:
        ax = plt.gca()
    
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

import astropy.units as u
import astropy.constants as constantes
def sample_Bailer_Jones(x_sample,y_sample,rlen =1.99552094, omega=0.0738, domega=0.0357):
	sample = []
	for ind,x in enumerate(x_sample):
		Y_BJ = Bailer_Jones_pdf(x,rlen =1.99552094, omega=0.0738, domega=0.0357)

		if y_sample[ind]<Y_BJ:
			sample.append(x)

	return sample

def Bailer_Jones_pdf(x,rlen =1.99552094, omega=0.0738, domega=0.0357):

	omega_zp=-0.029 #fixed zero point
	w_gaia =x*x*np.exp(-x/rlen)/domega * np.exp(- (1./(2.*domega*domega))*np.power(omega-omega_zp-1./x,2.0))
	
	return w_gaia




def find_priors2(params):

	theta_s, Av,Rv = params

	if (theta_s<0.1):

		return -np.inf

	if (theta_s>2):

		return -np.inf

	if (Av<2.0) | (Av>5):

		return -np.inf

	return 0




def objective_function_3(params,data,magnification,mask,isochrones,obs_mags,extinction):
	
	#priors
	priors = find_priors3(params,isochrones,obs_mags,extinction)
	if np.isinf(priors):
		return -np.inf

	
	radius, Av,Rv,Teff,abundance,logg,distance = params
	star_spectrum = Spyctres.star_spectrum(Teff,abundance,logg,catalog='k93models')
	normalisation = (radius/distance*solar_radii_to_parsec)**2
	wave = star_spectrum.wave * u.Angstrom
	spectrum = star_spectrum.flux*normalisation* (u.erg / u.cm**2 / u.s / u.Angstrom)

	intern_mask = (np.array(wave)>=data[0,0]) & (np.array(wave)<=data[-1,0])
	wave = wave[intern_mask]
	spectrum = spectrum[intern_mask]

	abso = 10**(Spyctres.absorption_law(Av,Rv,data[:,0]/10000)/2.5)


	res = ((data[:,1]-np.array((spectrum*magnification).data)/abso)/data[:,2])**2
	
	chichi = np.sum(res[~mask])
	chichi  = -0.5*chichi
	

	chichi += priors

	
	return chichi
	
def observed_mag_priors(params,isochrones,obs_mags,extinction):

	theta_s, Av,Rv,Teff,abundance,logg,distance = params
	index = np.argmin((isochrones[:,1]-abundance)**2+(isochrones[:,7]-np.log10(Teff))**2+(isochrones[:,8]-logg)**2)
	absolute_mags = isochrones[index,[27,30,31,32]]
	mu = 5*np.log10(distance)-5

	res = np.abs(absolute_mags+mu+extinction-obs_mags[:,0])/obs_mags[:,1]
	if np.any(res)>2:
		return np.inf
	else:
		return 0

def find_priors(params,isochrones,obs_mags,extinction):

	theta_s, Av,Rv,Teff,abundance,logg,rescaling = params

	if (theta_s<0.1):

		return np.inf

	if (theta_s>2.0):

		return np.inf

	if (Av<0.5) | (Av>5):

		return np.inf
	if np.abs(abundance)>0.99:
		return np.inf
	if np.abs(Teff-4000)>1000:
		return np.inf

	if np.abs(logg-2)>1.5:
		return np.inf
	if Teff<3500:
		return np.inf
	#pp = observed_mag_priors(params,isochrones,obs_mags,extinction)
	
	pp = 0
	
	return pp



def find_priors3(params,isochrones,obs_mags,extinction):

	radius, Av,Rv,Teff,abundance,logg,distance = params

	if (radius<1):

		return np.inf

	if (radius>100):

		return np.inf

	if (Av<2.0) | (Av>5):

		return np.inf
	if np.abs(abundance)>0.9:
		return np.inf
	if np.abs(Teff-4500)>1000:
		return np.inf

	if np.abs(logg-2)>1.5:
		return np.inf
	if Teff<3500:
		return np.inf

	pp = observed_mag_priors(params,isochrones,obs_mags,extinction)
	

	
	return pp


def objective_function(params,data,magnification,mask,isochrones,obs_mags,extinction):


	#priors
	priors = find_priors(params,isochrones,obs_mags,extinction)
	if np.isinf(priors):
		return np.inf

	
	theta_s, Av,Rv,Teff,abundance,logg,rescaling = params
	star_spectrum = Spyctres.star_spectrum(Teff,abundance,logg,catalog='k93models')
	normalisation = (10**theta_s/uas_to_rad)**2
	wave = star_spectrum.wave * u.Angstrom
	spectrum = star_spectrum.flux*normalisation* (u.erg / u.cm**2 / u.s / u.Angstrom)

	intern_mask = (np.array(wave)>=data[0,0]) & (np.array(wave)<=data[-1,0])
	wave = wave[intern_mask]
	spectrum = spectrum[intern_mask]

	abso = 10**(Spyctres.absorption_law(Av,Rv,data[:,0]/10000)/2.5)


	eflux = 10**rescaling*data[:,2]
	
	res = ((data[:,1]-np.array((spectrum*magnification).data)/abso)/(eflux))**2+np.log(eflux**2)
	
	chichi = np.sum(res[~mask])

	

	chichi += priors

	
	return chichi

def objective_function_2(params,data,magnification,mask,isochrones,obs_mags,extinction):

	#priors
	priors = find_priors(params,isochrones,obs_mags,extinction)
	if np.isinf(priors):
		return -np.inf

	
	theta_s, Av,Rv,Teff,abundance,logg,rescaling = params
	star_spectrum = Spyctres.star_spectrum(Teff,abundance,logg,catalog='k93models')
	normalisation = (10**theta_s/uas_to_rad)**2

	wave = star_spectrum.wave * u.Angstrom
	spectrum = star_spectrum.flux*normalisation* (u.erg / u.cm**2 / u.s / u.Angstrom)

	intern_mask = (np.array(wave)>=data[0,0]) & (np.array(wave)<=data[-1,0])
	wave = wave[intern_mask]
	spectrum = spectrum[intern_mask]

	abso = 10**(Spyctres.absorption_law(Av,Rv,data[:,0]/10000)/2.5)

	eflux = 10**rescaling*data[:,2]
	
	res = ((data[:,1]-np.array((spectrum*magnification).data)/abso)/(eflux))**2+np.log(eflux**2)
	
	
	chichi = np.sum(res[~mask])
	chichi  = -0.5*chichi
	

	chichi += priors

	
	return chichi
	
def define_2MASS_filters():

	#J
	data = np.loadtxt('J_2MASS_responses.txt')
	MASS_J = speclite.filters.FilterResponse(wavelength = data[:,0]*u.um , response = data[:,1], meta=dict(group_name='MASS', band_name='J'))
	#H
	data = np.loadtxt('H_2MASS_responses.txt')
	MASS_J = speclite.filters.FilterResponse(wavelength = data[:,0]*u.um , response = data[:,1], meta=dict(group_name='MASS', band_name='H'))
	#K
	data = np.loadtxt('K_2MASS_responses.txt')
	MASS_J = speclite.filters.FilterResponse(wavelength = data[:,0]*u.um , response = data[:,1], meta=dict(group_name='MASS', band_name='K'))


def SED_offset_XShooter(sed,spectrum):

	sorted_spec = spectrum[spectrum[:,0].argsort(),]
	unique,unique_index = np.unique(sorted_spec[:,0],return_index=True)

	unique_spec = sorted_spec[unique_index]


	interp_spec = interpolate.interp1d(unique_spec[:,0],unique_spec[:,1])
	wave_temp = np.arange(np.min(np.round(spectrum[:,0])+1),np.max(np.round(spectrum[:,0])-1))
	spec_temp = interp_spec(wave_temp)

	spec_mags_ab = []

	for ind,fil in enumerate(sed[:,-1]):
		
		if 'Bessel' in fil:

			templates = bessel_filters

		if 'SDSS' in fil:

			templates = sdss_filters

		if 'MASS' in fil:

			templates = twomass_filters

		list_of_bands = [templates[i].meta['band_name'] for i in range(len(templates))]

		index = np.where(fil.split('_')[-1] == np.array(list_of_bands))[0][0]
		spec_mag_ab = templates[index].get_ab_magnitude(spec_temp* (u.erg / u.cm**2 / u.s / u.Angstrom),wave_temp*u.nm)
		spec_mags_ab.append(spec_mag_ab)
	offset_XShooter = np.sum((spec_mags_ab-sed[:,1].astype(float))*1/sed[:,2].astype(float)**2)/np.sum(1.0/sed[:,2].astype(float)**2)
	
	return 10**(offset_XShooter/2.5)

spectra = {}
uas_to_rad = 180*3600*10**6/np.pi
solar_radii_to_parsec = 2.254*10**-8


define_2MASS_filters()
twomass_filters = speclite.filters.load_filters('MASS-J', 'MASS-H','MASS-K')
sdss_filters = speclite.filters.load_filters('sdss2010-*')
bessel_filters = speclite.filters.load_filters('bessell-*')





#Adding new spectra

spectra = {}

# XShooter, time obs 29/07/2019, magnification ~ 8.14

magnification = 8.14 #from photometry
wave_obs = [5456,6156,7472,12355,16458,21603]#Angstrom

ab_corrections = [0.02,0.16,0.37,0.91,1.39,1.85] #http://www.astro.osu.edu/~martini/usefuldata.html
obs_mags = [13.5,12.5,11.5,11.849-2.5*np.log10(magnification),10.862-2.5*np.log10(magnification),10.513-2.5*np.log10(magnification)] # J,H,K are faked from Vizier
obs_emags = [0.1,0.1,0.1,0.026,0.024,0.023]
filters = ['Bessel_V','SDSS_r','SDSS_i','MASS_J','MASS_H','MASS_K']
extinction = [2.2,2.2-1.1,0.63,0.4,0.26]

SED_flux,SED_mag = Spyctres.derive_SED_from_obs_mag(wave_obs,obs_mags,obs_emags,ab_corrections,filters)

NIR = Spyctres.extract_spectrum('./Spectres/TOO_Gaia19bld_runA_SCI_SLIT_FLUX_MERGE1D_NIR.fits')
VIS = Spyctres.extract_spectrum('./Spectres/TOO_Gaia19bld_runA_SCI_SLIT_FLUX_MERGE1D_VIS.fits')
UVB = Spyctres.extract_spectrum('./Spectres/TOO_Gaia19bld_runA_SCI_SLIT_FLUX_MERGE1D_UVB.fits')

spectrum_XShooter = np.r_[UVB,VIS,NIR]

spectrum_XShooter = spectrum_XShooter[spectrum_XShooter[:,0].argsort(),]
spectrum_XShooter[:,0] *= 10 #nm to Angstrom
offset = Spyctres.SED_offset(SED_mag,spectrum_XShooter,bessel_filters,sdss_filters,twomass_filters)



fake_spectrum = Spyctres.star_spectrum(4140,0.25,1.6,catalog='k93models')
wave = np.array(fake_spectrum.wave) #Angstrom 
step = 50 #Angstrom
global_mask = (wave>spectrum_XShooter[0,0]+step) & (wave<spectrum_XShooter[-1,0]-step)
wave = wave[global_mask]

bin_spec = Spyctres.bin_spectrum(spectrum_XShooter,np.array(wave))
SNR = bin_spec[:,1]/bin_spec[:,2]
data_fit = np.c_[bin_spec[:,0],bin_spec[:,1]*offset,bin_spec[:,1]*offset/SNR]


spectra['XShooter_29_07_2019'] = data_fit

#Find telluric lines


telluric_mask = bin_spec[:,0]>0
telluric_lines = fits.open('/home/ebachelet/Downloads/pwv_R300k_airmass1.0/LBL_A10_s0_w050_R0300000_T.fits')#https://www.aanda.org/articles/aa/pdf/2014/08/aa23790-14.pdf
telluric_lines = np.c_[telluric_lines[1].data['lam']*10000,telluric_lines[1].data['trans']]
telluric_lines,telluric_mask = Spyctres.telluric_lines(telluric_lines,wave,threshold=0.98)

import pdb; pdb.set_trace()



