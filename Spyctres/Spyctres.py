import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate
import pysynphot as PS
import astropy.units as u
import astropy.constants as constantes
import astropy.modeling.physical_models as apm
import os
import speclite.filters
import speclite
import scipy.interpolate as si


import stsynphot
from synphot import units, SourceSpectrum
from synphot.models import BlackBodyNorm1D

UAS_TO_RAD = 180*3600*10**6/np.pi

class BlackBody(object):

    def __init__(self,temperature=5778):
    
        self.temperature = temperature
        #self.bb = apm.BlackBody(temperature=self.temperature*u.K)

    def __call__(self,Lambda):
    
        lamb = Lambda.value

        c1 = 1.1910429723971884e-16*10**50 #Angstrom to meter^-5
        c2 = constantes.h.value*constantes.c.value/(lamb*constantes.k_B.value*self.temperature)*10**10
        bb = c1/(lamb)**5/(np.exp(c2)-1)
        
        return bb*15.815130864774007*Lambda.value#photlam, 10**-7*np.pi/(1.98644746*10**-8/Lambda.value)
        
        
def get_element_lines(wavelength_range = [2000,10000], require_elements=['H','HE','HG','CA','FE','MG','NA','O'],intensity_threshold=50):

    elements_lines_path  = os.path.dirname(__file__)+'/data/Reader_Corliss_Lines.fits' #http://cdsarc.u-strasbg.fr/viz-bin/cat/VI/16
    lines = fits.open(elements_lines_path)
    waves = lines[1].data['wavel']
    
    # Air-Vacuum correction https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    
    mask = waves.astype(float)>2000
    
    s = 10**4/waves[mask]
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    waves[mask] *= n

    intensities = lines[1].data['INT']
    elements = lines[1].data['Element']
    ions  = lines[1].data['Spectrum']
        
    good_lines = np.c_[waves,elements,ions,intensities]

    general_mask = (intensities>intensity_threshold)&(waves>wavelength_range[0])&(waves<wavelength_range[1])

    if len(require_elements) == 0:
    
        pass
    
    else:
        
        elements_mask = [True if i in require_elements else False for i in elements]
        
        general_mask = (general_mask) & (elements_mask)
           
    return good_lines[general_mask] 
        
def plot_element_lines(figure_axe,lines):


    for line in lines:

        figure_axe.axvline(float(line[0]),linestyle=':',color='grey',alpha=0.5)
        figure_axe.text(float(line[0]),0.8,line[1]+line[2],rotation='vertical',fontdict=dict(color='grey',fontsize=10),bbox=dict(alpha=0.0,facecolor='w',edgecolor='w'))
        




def star_model_new(parameters,wave,catalog='k93models'):


    theta_s, Av,Teff,abundance,logg = parameters
    
    try:
        model_spectrum = star_spectrum_new(Teff,abundance,logg,catalog=catalog)
    except:
        return np.inf

    normalisation = (10**theta_s/UAS_TO_RAD)**2    

    abso = 10**(Wang_absorption_law(Av,np.array(wave)/10000)/2.5)
    flux = np.array(model_spectrum(wave*u.AA)*1.98644746*10**-8/wave)
    spectrum = flux*normalisation/abso

    model = np.c_[np.array(wave),np.array(spectrum),[1]*len(spectrum)]
    
    return model
            

def star_model(parameters,catalog='k93models'):


    theta_s, Av,Teff,abundance,logg = parameters
    
    try:
            starspectrum = star_spectrum(Teff,abundance,logg,catalog=catalog)
    except:
            return None
            
    normalisation = (10**theta_s/UAS_TO_RAD)**2    
    
    wave = starspectrum.wave * u.Angstrom
    abso = 10**(Wang_absorption_law(Av,np.array(wave)/10000)/2.5)
    spectrum = starspectrum.flux*normalisation* (u.erg / u.cm**2 / u.s / u.Angstrom)/abso

    model = np.c_[np.array(wave),np.array(spectrum),[1]*len(spectrum)]
    
    return model

def sed_chichi(spectrum,sed):

    chichi = 0
    #breakpoint()
    for obs in sed:
    
        filt = obs[0]
        mag_ab = obs[1]
        emag_ab = obs[2]
        
        predicted_mag_ab = filt.get_ab_magnitude(spectrum[:,1],spectrum[:,0])
        #breakpoint()
        chichi += (mag_ab-predicted_mag_ab)**2/emag_ab**2
        
    return chichi


def sed_chichi_new(spectrum,sed,magnification=1):

    chichi = 0
    #breakpoint()
    for obs in sed:
    
        filt = obs[0]
        mag_ab = obs[1]
        emag_ab = obs[2]
        
        flux = np.array(spectrum(filt._wavelength*u.AA)*1.99*10**-8/filt._wavelength)
        predicted_mag_ab = filt.get_ab_magnitude(flux,filt._wavelength)
        #breakpoint()
        chichi += (mag_ab-predicted_mag_ab)**2/emag_ab**2
        
    return chichi

def fit_spectra_one_star(spectras=[],seds=[],magnifications=[],catalog='k93models'):

    pass
    
def fit_spectra_one_star_chichi(params,spectras=[],seds=[],magnifications=[],telluric_lines_mask=None,catalog='k93models'):

    star_parameters = params[:5]
    
    model_spectrum = star_model(star_parameters,catalog=catalog)

    if model_spectrum is None:
        
        return np.inf
   
    try:
    
        rescale_flux_parameters = [params[5+i] for i in range(len(spectras))]
        
    except:
    
        rescale_flux_parameters = None 

    try:
    
        rescale_errors_parameters = [params[5+len(spectras)+i] for i in range(len(spectras))]
        
    except:
    
        rescale_errors_parameters = None    
    

    chichi = 0
    
    for ind,data in enumerate(spectras):

        tobin = np.copy(model_spectrum)
        tobin[:,1] *= magnifications[ind] 
        
        if rescale_flux_parameters is not None:
        
            rescale_flux = 10**rescale_flux_parameters[ind]
            tobin[:,1] /= rescale_flux
        
        mask = (tobin[:,0]>=data[0,0]) & (tobin[:,0]<=data[-1,0])
        
        if  tobin[mask].shape == data.shape:
        
            bin_spec = tobin[mask]
        
        else:
            
            bin_spec,cov = bin_spectrum(tobin,data[:,0])
        
        if telluric_lines_mask is not None:
        
            mask = telluric_lines_mask(data[:,0]).astype(bool)
            
        else:
        
            mask = [True]*len(data)
        
        errors = data[mask,2]**2
        
        if rescale_errors_parameters is not None:
        
            rescale_errors = 10**rescale_errors_parameters[ind]
            errors *= rescale_errors**2
       
        res = ((data[mask,1]-bin_spec[mask,1]))**2/errors+np.log(errors)+np.log(2*np.pi)

        chichi += np.sum(res)
     
    chichi_seds = 0    
    for ind,sed in enumerate(seds):
        
        sed = seds[ind]
        tobin = np.copy(model_spectrum)
        tobin[:,1] *= magnifications[ind]    
       
        chichi_seds += sed_chichi(tobin,sed)
            
    chichi += chichi_seds
            
    return chichi      
            

def fit_spectra_one_star_chichi_new(params,spectras=[],seds=[],magnifications=[],telluric_lines_mask=None,catalog='k93models'):

    theta_s, Av,Teff,abundance,logg = params[:5]
    
    #model_spectrum = star_model(star_parameters,catalog=catalog)
    try:
        model_spectrum = stsynphot.grid_to_spec(catalog, Teff,abundance,logg) 
    except:
        return np.inf

    normalisation = (10**theta_s/UAS_TO_RAD)**2    
  
    try:
    
        rescale_flux_parameters = [params[5+i] for i in range(len(spectras))]
        
    except:
    
        rescale_flux_parameters = None 

    try:
    
        rescale_errors_parameters = [params[5+len(spectras)+i] for i in range(len(spectras))]
        
    except:
    
        rescale_errors_parameters = None    
    

    chichi = 0
    
    for ind,data in enumerate(spectras):

        flux = np.array(model_spectrum(data[:,0]*u.AA)*1.99*10**-8/data[:,0])
    
        wave = data[:,0] 
        abso = 10**(Wang_absorption_law(Av,np.array(wave)/10000)/2.5)
        flux *= normalisation/abso
        

   
        if rescale_flux_parameters is not None:
        
            rescale_flux = 10**rescale_flux_parameters[ind]
            flux /= rescale_flux
       
        if telluric_lines_mask is not None:
        
            mask = telluric_lines_mask(data[:,0]).astype(bool)
            
        else:
        
            mask = [True]*len(data)
        
        errors = data[mask,2]**2
        
        if rescale_errors_parameters is not None:
        
            rescale_errors = 10**rescale_errors_parameters[ind]
            errors *= rescale_errors**2
       
        res = ((data[mask,1]-flux))**2/errors+np.log(errors)+np.log(2*np.pi)

        chichi += np.sum(res)
     
    chichi_seds = 0    
    for ind,sed in enumerate(seds):
        
        sed = seds[ind]    
        
        chi = 0
        for obs in sed:
        
            filt = obs[0]
            mag_ab = obs[1]
            emag_ab = obs[2]
            
            wave = filt._wavelength*u.AA
            flux = np.array(model_spectrum(wave)*1.99*10**-8/wave)
             
            abso = 10**(Wang_absorption_law(Av,np.array(wave)/10000)/2.5)
            flux *= normalisation/abso
            predicted_mag_ab = filt.get_ab_magnitude(flux,wave)
            #breakpoint()
            chi += (mag_ab-predicted_mag_ab)**2/emag_ab**2

        chichi_seds += chi

    chichi += chichi_seds
            
    return chichi      




def source_blend_from_flux(specs,magnifications):

    v1 = 0
    v2 = 0  
    A = 0
    B = 0
    C = 0

    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    t6 = 0
    t7 = 0
       

    for i in range(len(specs)):
        
         A += magnifications[i]**2/specs[i][:,2]**2
         B += magnifications[i]/specs[i][:,2]**2
         C += 1/specs[i][:,2]**2
         v1 += specs[i][:,1]*magnifications[i]/specs[i][:,2]**2
         v2 += specs[i][:,1]*1/specs[i][:,2]**2


         t1 += magnifications[i]*specs[i][:,1]/specs[i][:,2]**2
         t2 += 1/specs[i][:,2]**2
         t3 += magnifications[i]/specs[i][:,2]**2
         t4 += specs[i][:,1]/specs[i][:,2]**2
         t5 += magnifications[i]**2/specs[i][:,2]**2
         t6 += 0
         t7 += magnifications[i]**2/specs[i][:,2]**2
    
    aa = np.zeros((len(A),len(A)))
    bb = np.zeros((len(A),len(A)))
    cc = np.zeros((len(A),len(A)))

    np.fill_diagonal(aa,A)
    np.fill_diagonal(bb,B)
    np.fill_diagonal(cc,C) 
    mat = np.block([[aa,bb],[bb,cc]])
    

    fs = (t1*t2-t3*t4)/(t5*t2-t3**2)
    fb = (t7*t4-t3*t1)/(t5*t2-t3**2)
    sol,res, rank, s = np.linalg.lstsq(mat,np.r_[v1,v2,],rcond=-1)

    cov = np.linalg.pinv(np.dot(mat.T,mat))
    if len(res) !=0:
        chi2 = np.sum((res-np.dot(mat,sol))**2)/(len(res)-len(specs[0]))
        #import pdb; pdb.set_trace()
        cov *= chi2
        efs = cov.diagonal()[:len(A)]**0.5 
        efb = cov.diagonal()[len(A):]**0.5
        
    else:
        print('Not enough spectra to estimate covariance! Not reliable covariance')
        chi2 = 1
        efs = 1/(magnifications[0]-magnifications[1])*(specs[0][:,2]**2+specs[1][:,2]**2)**0.5
        efb = 1/(magnifications[0]-magnifications[1])*(specs[0][:,2]**2*magnifications[1]**2+specs[1][:,2]**2*magnifications[0]**2)**0.5    
    fs = sol[:len(A)] 
    fb = sol[len(A):]
    
    return fs,fb,efs,efb
	
def source_blend_from_flux2(specs,magnifications):

    mat_1_spectra = np.zeros((len(specs[0]),len(specs[0])))
    mat_mag = np.copy(mat_1_spectra)
    np.fill_diagonal(mat_mag,1)
    mat_blend = np.copy(mat_1_spectra)    
    np.fill_diagonal(mat_blend,1)
    
    for ind,mag in enumerate(magnifications):
    
        weights = specs[ind][:,2]
        matmag = mat_mag*mag/weights

        
        try:
            mat_tot = np.r_[mat_tot,np.c_[matmag,mat_blend/weights]]   
            obs = np.r_[obs,specs[ind][:,1]/weights] 
        except:
            mat_tot = np.c_[matmag,mat_blend/weights]
            obs = specs[ind][:,1]/weights 


    
    sol,res, rank, s = np.linalg.lstsq(mat_tot,obs,rcond=-1)
    cov = np.linalg.pinv(np.dot(mat_tot.T,mat_tot))
    sigmas = cov.diagonal()**0.5
    
    fs = sol[:len(specs[0])]
    fb = sol[len(specs[0]):]
    efs = sigmas[:len(specs[0])]
    efb = sigmas[len(specs[0]):]

    return fs,fb,efs,efb
    
    
def star_spectrum(Teff,abundance,logg,catalog='k93models'):

    spectrum = PS.Icat(catalog,Teff,abundance,logg)

    return spectrum

def star_spectrum_new(Teff,abundance,logg,catalog='k93models'):

    if catalog == 'BlackBody':
    
        spectrum = BlackBody(temperature=Teff)
        
    else:

        spectrum = stsynphot.grid_to_spec(catalog, Teff,abundance,logg) 

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


def Cardelli_absorption_law(Av,Rv,lamb):

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

def Wang_absorption_law(Av,lamb):
    #https://iopscience.iop.org/article/10.3847/1538-4357/ab1c61/pdf
    alambda = np.zeros(len(lamb))
    mask = lamb<1
    Y = 1/lamb-1.82
    
    alambda[mask] = Av*(1+0.7499*Y[mask]-0.1086*Y[mask]**2-0.08909*Y[mask]**3+0.02905*Y[mask]**4+0.01069*Y[mask]**5+0.001707*Y[mask]**6-0.001002*Y[mask]**7)
    
    alambda[~mask] = Av*0.3722*lamb[~mask]**-2.070
    mask = alambda<0
    alambda[mask] = 0

    return alambda    
   
    
def absorption_law_2(Av,Rv,lamb):
    
    a_lamb = Av*((0.55/lamb)**Rv)
    
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
#        eflux = (spec[0].data[3,0]**2+spec[0].data[2,0])**0.5
        eflux = spec[0].data[3,0]
        #eflux = np.array([1.0]*len(eflux))
    except:
        #XSHOOTER spectra
        coef_pixel_wave = head['CDELT1']    
        flux = spec[0].data
        eflux = spec[1].data


    wavelenght = value_ref_pixel+coef_pixel_wave*(np.arange(0,num_pixels)-ref_pixel)

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
    
    mask = (lambda_ref>=data[0,0]) & (lambda_ref<data[-1,0])
    
    flux = []
    errors = []
    cij = []

    for ind,lamb in enumerate(lambda_ref[mask]):

        try:
            index = np.argmin(np.abs(data[:,0]-lamb))

            if np.abs(data[index,0]-lamb)>10**-10: 
               
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
                
            else:
                   
                flux.append(data[index,1])
                cij_line = np.zeros(len(data))
                cij_line[index] = 1
                cij.append(cij_line)
                
        except:
                breakpoint()
                     
    covariance = np.array(cij)
    
    #eflux = np.dot(covariance,data[:,2]**2)**0.5
    final_covariance = np.dot(covariance*data[:,2],(covariance*data[:,2]).T)
    eflux = final_covariance.diagonal()**0.5

    return np.c_[lambda_ref[mask],flux,eflux],final_covariance


def define_2MASS_filters():


    TWOMASS_filters_path = os.path.dirname(__file__)+'/data/'
    #J
    data = np.loadtxt( TWOMASS_filters_path+'J_2MASS_responses.dat')
    MASS_J = speclite.filters.FilterResponse(wavelength = data[:,0]*u.um , response = data[:,1], meta=dict(group_name='MASS', band_name='J'))
    #H
    data = np.loadtxt( TWOMASS_filters_path+'H_2MASS_responses.dat')
    MASS_J = speclite.filters.FilterResponse(wavelength = data[:,0]*u.um , response = data[:,1], meta=dict(group_name='MASS', band_name='H'))
    #K
    data = np.loadtxt( TWOMASS_filters_path+'K_2MASS_responses.dat')
    MASS_J = speclite.filters.FilterResponse(wavelength = data[:,0]*u.um , response = data[:,1], meta=dict(group_name='MASS', band_name='K'))


def define_GAIA_filters():


    GAIA_filter_path = os.path.dirname(__file__)+'/data/'
    #G
    data = np.loadtxt( GAIA_filter_path+'G_GAIA_responses.dat')
    GAIA_G = speclite.filters.FilterResponse(wavelength = data[:,0]*u.nm , response = data[:,1], meta=dict(group_name='GAIA', band_name='G'))
  
def define_SDSS_prime_filters():

    SDSS_prime_filters_path = os.path.dirname(__file__)+'/data/'
    #u'
    data = np.loadtxt( SDSS_prime_filters_path+'SLOAN_SDSS.uprime_filter.dat')
    SDSS_prime_u = speclite.filters.FilterResponse(wavelength = data[:,0]*u.AA , response = data[:,1], meta=dict(group_name='SDSS_prime', band_name='u'))
    #g'
    data = np.loadtxt( SDSS_prime_filters_path+'SLOAN_SDSS.gprime_filter.dat')
    SDSS_prime_g = speclite.filters.FilterResponse(wavelength = data[:,0]*u.AA , response = data[:,1], meta=dict(group_name='SDSS_prime', band_name='g'))
    #r'
    data = np.loadtxt( SDSS_prime_filters_path+'SLOAN_SDSS.rprime_filter.dat')
    SDSS_prime_r = speclite.filters.FilterResponse(wavelength = data[:,0]*u.AA , response = data[:,1], meta=dict(group_name='SDSS_prime', band_name='r'))
    #i'
    data = np.loadtxt( SDSS_prime_filters_path+'SLOAN_SDSS.iprime_filter.dat')
    SDSS_prime_i = speclite.filters.FilterResponse(wavelength = data[:,0]*u.AA , response = data[:,1], meta=dict(group_name='SDSS_prime', band_name='i'))
    #z'
    data = np.loadtxt( SDSS_prime_filters_path+'SLOAN_SDSS.zprime_filter.dat')
    SDSS_prime_z = speclite.filters.FilterResponse(wavelength = data[:,0]*u.AA , response = data[:,1], meta=dict(group_name='SDSS_prime', band_name='z'))

def mag_to_fluxdens(seds,mag_system='Vega'):

    if mag_system=='Vega':
    
        ab_corr = derive_AB_correction(seds[:,0])
        
    else:
    
        ab_corr = 0
        
    mag = seds[:,1].astype(float)
    emag = seds[:,2].astype(float)
    pivots = np.array([i.effective_wavelength.value for i in seds[:,0]])
    
    mag += ab_corr
    mag_to_jansky = 10**(-0.4*(mag+48.6))/10**-23
    
    jansky_to_fluxdens = mag_to_jansky*1/(3.34*10**4)*1/pivots**2

    ejansky_to_fluxdens = emag*jansky_to_fluxdens*2.5/np.log(10)

    fluxdens = np.c_[pivots.tolist(),jansky_to_fluxdens.tolist(),ejansky_to_fluxdens.tolist()]


    return fluxdens


def load_telluric_lines(threshold = 0.95):
    
    
    telluric_lines_path  = os.path.dirname(__file__)+'/data/LBL_A10_s0_w050_R0300000_T.fits' #https://www.aanda.org/articles/aa/pdf/2014/08/aa23790-14.pdf
    telluric_lines = fits.open(telluric_lines_path)
    telluric_lines = np.c_[telluric_lines[1].data['lam']*10000,telluric_lines[1].data['trans']]
    telluric_mask = telluric_lines[:,1]<threshold
    
    telluric_lines_interp = si.interp1d(telluric_lines[:,0],telluric_lines[:,1],fill_value='extrapolate')
    telluric_mask_interp = si.interp1d(telluric_lines[:,0],telluric_mask.astype(int),fill_value='extrapolate')
    
    
    

    return telluric_lines_interp, telluric_mask_interp
    
    
#def telluric_lines(telluric_lines,wave,threshold=0.98):
#    # 
#    
##    import scipy.interpolate as si
##    interp = si.interp1d(telluric_lines[:,0],telluric_lines[:,1],kind='cubic',bounds_error=False,fill_value = 0)
##    telluric_lines = np.c_[wave,interp(wave)]
##    telluric_mask = telluric_lines[:,1]<threshold
#     
#    telluric_lines = bin_absorption(telluric_lines,np.array(wave))

#    
#    telluric_mask = wave<0
#        

#    for ind,lam in enumerate(wave):

#        index = np.argmin(np.abs(telluric_lines[:,0]-lam))
#        if telluric_lines[index,1]<threshold:

#            telluric_mask[ind] = True

#    
#    breakpoint()
#    return telluric_lines,np.c_[wave,telluric_mask]

    
def SED_offset(sed,spectrum,bessel_filters=None,sdss_filters=None,twomass_filters=None,gaia_filters=None):
    #breakpoint()
    sorted_spec = spectrum[spectrum[:,0].argsort(),]
    unique,unique_index = np.unique(sorted_spec[:,0],return_index=True)

    unique_spec = sorted_spec[unique_index]


    interp_spec = interpolate.interp1d(unique_spec[:,0],unique_spec[:,1],bounds_error = False,fill_value =0)
    #wave_temp = np.arange(np.min(np.round(spectrum[:,0])+1),np.max(np.round(spectrum[:,0])-1))
    #spec_temp = interp_spec(wave_temp) 
    waves_center = []
    spec_mags_ab = []
    wave_range = []
    for ind,fil in enumerate(sed[:,-1]):
        
        if 'Bessel' in fil:

            templates = bessel_filters

        if 'SDSS' in fil:

            templates = sdss_filters

        if 'MASS' in fil:

            templates = twomass_filters

        if 'GAIA' in fil:

            templates = gaia_filters

        list_of_bands = [templates[i].meta['band_name'] for i in range(len(templates))]

        index = np.where(fil.split('_')[-1] == np.array(list_of_bands))[0][0]
        #import pdb; pdb.set_trace()   
        try:
             spec_mag_ab = templates[index].get_ab_magnitude(unique_spec[:,1]* (u.erg / u.cm**2 / u.s / u.Angstrom),unique_spec[:,0]*u.Angstrom)
        except:
             spec_mag_ab = templates[index].get_ab_magnitude(interp_spec( templates[index].wavelength)* (u.erg / u.cm**2 / u.s / u.Angstrom),templates[index].wavelength*u.Angstrom)
        waves_center.append(templates[index].effective_wavelength.value)
        spec_mags_ab.append(spec_mag_ab)
        wave_range.append(templates[index].wavelength[-1]-  templates[index].wavelength[0])
        #import pdb; pdb.set_trace()   
    res =   spec_mags_ab-sed[:,1].astype(float)

    quant = 10**(res/2.5)
    equant = sed[:,2].astype(float)*quant/2.5*np.log(10)
    mean = np.sum(quant/equant**2)/np.sum(1/equant**2)
    emean = 1/np.sum(1/equant**2)**0.5
    #offset = np.sum((spec_mags_ab-sed[:,1].astype(float))*1/sed[:,2].astype(float)**2)/np.sum(1.0/sed[:,2].astype(float)**2)
    #error = (1/np.sum(1/sed[:,2].astype(float)**2))**0.5*10**(offset/2.5)*np.log(10)/2.5
    #import pdb; pdb.set_trace()   
    return mean,emean,quant,equant

    #phot_calib = interpolate.interp1d(waves_center,np.array(spec_mags_ab)-sed[:,1].astype(float),fill_value = 'extrapolate',kind=0)
    return phot_calib
    
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
    
    
def derive_AB_correction(filters):

    correction = []
    

    for fil in filters:

        correction.append(fil.get_ab_magnitude(PS.Vega.flux,PS.Vega.wave))
        
    return np.array(correction)


