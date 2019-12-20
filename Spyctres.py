import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate
import pysynphot as PS
import astropy.units as u
import astropy.constants as constantes


 
def star_spectrum(Teff,abundance,logg,catalog='k93models'):

	spectrum = PS.Icat(catalog,Teff,abundance,logg)

	return spectrum



def bin_absorption(data,lambda_ref):

	steps = np.diff(lambda_ref)/2
	steps = np.r_[steps,steps[-1]]
	abso = []

	for ind,lamb in enumerate(lambda_ref):
		try:
			#index = np.argmin(np.abs(data[:,0]-lamb))
			index_moins = np.argmin(np.abs(data[:,0]-lamb+steps[ind]))	
			index_plus = np.argmin(np.abs(data[:,0]-lamb-steps[ind]))	
			#abso.append((data[index,1]))
			abso.append(np.median(data[index_moins:index_plus,1]))		
			# http://www.analyticalgroup.com/download/WEIGHTED_MEAN.pdf
			
		except:
			pass			
	return np.c_[lambda_ref,abso]



def absorption_law(Av,Rv,lamb):
	#articles.adsabs.harvard.edu/pdf/1989ApJ...345..245C
	A=[]
	B=[]
	for lambI in lamb:
		lambi = 1./lambI
		if (lambi<1.1) & (lambi>0.3):
	
			a = 0.574*lambi**1.61
			b = -0.527*lambi**1.61
			
		if (lambi>1.1) & (lambi<3.3):
			y = lambi-1.82
			a = 1+0.17699*y-0.50477*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
			b =  1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
		
		
		if (lambi<0.3) | (lambi>3.3):
		
				a = 0
				b = 0
		A.append(a)
		B.append(b)

	a_lamb = (np.array(A)+np.array(B)/Rv)*Av
	
	return a_lamb
def extract_spectrum(name):


	spec =  fits.open(name)
	head = spec[0].header
	


	num_pixels = head['NAXIS1']
	ref_pixel = head['CRPIX1']
	value_ref_pixel = head['CRVAL1']

	try:
		#LCO spectra
		coef_pixel_wave = head['CD1_1']
		flux = spec[0].data[0,0]
		eflux = spec[0].data[3,0]

	except:
		#XSHOOTER spectra
		coef_pixel_wave = head['CDELT1']	
		flux = spec[0].data
		eflux = spec[1].data


	wavelenght = value_ref_pixel+coef_pixel_wave*np.arange(0,num_pixels)

	mask = (flux>0) & (eflux>0)
	spectrum = np.c_[wavelenght,flux,eflux][mask]	
	mask = np.abs(spectrum[:,2]/spectrum[:,1])<0.1

	spectrum = np.c_[spectrum[:,0],spectrum[:,1],spectrum[:,2],mask]


	return spectrum


def bin_spectrum(data,lambda_ref):
	#https://arxiv.org/pdf/1705.05165.pdf
	#http://www.analyticalgroup.com/download/WEIGHTED_MEAN.pdf
	# match https://www.astrobetter.com/blog/2013/08/12/python-tip-re-sampling-spectra-with-pysynphot/ but gives errors
	steps = np.diff(lambda_ref)/2
	steps = np.r_[steps,steps[-1]]
	flux = []

	cij = []
	for ind,lamb in enumerate(lambda_ref):


		index = np.argmin(np.abs(data[:,0]-lamb))
		index_moins = np.argmin(np.abs(data[:,0]-lamb+steps[ind]))	
		index_plus = np.argmin(np.abs(data[:,0]-lamb-steps[ind]))



		winside = data[index_moins:index_plus+1,0]
		einside = data[index_moins:index_plus+1,2]
		finside = data[index_moins:index_plus+1,1]
		bins = np.array([(data[i+1,0]-data[i-1,0])/2 for i in range(index_moins,index_plus+1)])

		efficiency = np.zeros(len(winside))
		efficiency[1:-1] = 1
			
		efficiency[0] = np.abs(0.5-(data[index_moins,0]-lamb+steps[ind]))
		efficiency[-1] = np.abs(0.5-(data[index_plus,0]-lamb-steps[ind]))

		flux.append(np.sum(efficiency*bins*finside)/np.sum(bins*efficiency))
		cij_line = np.zeros(len(data))
		cij_line[index_moins:index_plus+1] = efficiency*bins/np.sum(bins*efficiency)
		cij.append(cij_line)

	covariance = np.array(cij)

	eflux = np.dot(covariance,data[:,2]**2)**0.5

		
	return np.c_[lambda_ref,flux,eflux]

def mag_to_fluxdens(mag,emag,ab_corr,wave):

	mag += ab_corr
	mag_to_jansky = 10**(-0.4*(mag+48.6))/10**-23
	
	jansky_to_fluxdens = mag_to_jansky*3.0*10**-5/wave**2
	ejansky_to_fluxdens = emag*jansky_to_fluxdens*2.5/np.log(10)

	return jansky_to_fluxdens,ejansky_to_fluxdens


def telluric_lines(telluric_lines,wave,threshold=0.98):

	try:
			
		telluric_lines = np.loadtxt('Telluric_lines_Xshooter.txt')
	except:
		
		telluric_lines = bin_absorption(telluric_lines,np.array(wave))
		np.savetxt('Telluric_lines_Xshooter.txt',telluric_lines)
	
	telluric_mask = wave<0
        

	for ind,lam in enumerate(wave):

		index = np.argmin(np.abs(telluric_lines[:,0]-lam))
		if telluric_lines[index,1]<threshold:

			telluric_mask[ind] = True

	

	return telluric_lines,telluric_mask
	
def SED_offset(sed,spectrum,bessel_filters,sdss_filters,twomass_filters):

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
		spec_mag_ab = templates[index].get_ab_magnitude(spec_temp* (u.erg / u.cm**2 / u.s / u.Angstrom),wave_temp*u.Angstrom)
		spec_mags_ab.append(spec_mag_ab)
	offset = np.sum((spec_mags_ab-sed[:,1].astype(float))*1/sed[:,2].astype(float)**2)/np.sum(1.0/sed[:,2].astype(float)**2)

	return 10**(offset/2.5)

def derive_SED_from_obs_mag(wave,mags,emags,ab_corrections,filters):

	flux = []
	eflux = []

	for ind,wav in enumerate(wave):

		flu,eflu = mag_to_fluxdens(mags[ind],emags[ind],ab_corrections[ind],wav)
		flux.append(flu)
		eflux.append(eflu)

	SED_flux =np.c_[wave,flux,eflux,filters] #erg/s/cm**2/angstrom

	SED_mag =np.c_[wave,mags+np.array(ab_corrections),emags,filters]#AB magnitudes

	return SED_flux,SED_mag

