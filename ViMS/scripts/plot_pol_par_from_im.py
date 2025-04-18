import os, sys, argparse, logging
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions
#import pyregion
from lib_fits import AllImages
#from AllImages import get_beam_area
import os
import glob
import argparse


parser = argparse.ArgumentParser(description='Plot polarisation properties from Q and U images')
parser.add_argument('--basename_pol', type=str, help='Basename of the polarisation images e.g obs02_3c286_mask-final')
parser.add_argument('--basename', type=str, help='Base name of the images')
parser.add_argument('--path', type=str, help='Path to the folder')
parser.add_argument('--region', type=str, help='Name of the region file')
parser.add_argument('--freq_list', type=str, help='Name of the frequency list file')
parser.add_argument('--rms_list', type=str, help='Name of the rms list file')
parser.add_argument('--name_out', type=str, help='Name of the output image')
parser.add_argument('--mean_rm', type=float, help='Mean RM value')
args = parser.parse_args()


path = args.path 
basename = args.basename
region = path +'/STOKES_CUBES/'+ args.region
freq_file = path +'/STOKES_CUBES/' + args.freq_list
rms_file = path +'/STOKES_CUBES/' + args.rms_list
name_polf = path +'/STOKES_CUBES/' + args.basename_pol + '_polf.fits'
name_pol = path +'/STOKES_CUBES/' + args.basename_pol + '_P.fits'
name_out = path +'/STOKES_CUBES/' + args.name_out
mean_RM = args.mean_rm #mean RM of the ionosphere



#-----------------End of inputs-------------------------------------------

#define model functions of 3C286
def model_RM(x):
    '''model of RM value for 3C286 from Hugo & Perley 2024'''
    return 0.12 #rad per m²

def model_pola(nu):
    '''model of frequency dep. polarisation angle of 3C286 Hugo & Perley (2024)'''
    c = 2.99792458e8 #speed of light in m/s
    result = np.zeros_like(nu)
    
    for i,f in enumerate(nu):
        wavelength = c/(f*1e9) #in m
        if f >= 1.7 and f <= 12: #in GHz
            result[i] = 32.64 - 85.37*wavelength**2
        elif f < 1.7: #in GHz
            result[i] = 29.53 + wavelength**2*(4005.88*np.log10(f)**3 - 39.38)
        else:
            logging.error('Only for frequencies below 12 GHz.')
            result[i] = np.nan
    return result

def model_polf(nu):
    c = 2.99792458e8 #speed of light in m/s
    result = np.zeros_like(nu)
    
    for i,f in enumerate(nu):
        wavelength = c/(f*1e9) #in m

        if f <= 12 and f >= 1.1: #in GHz
            result[i] = 0.080 - 0.053*wavelength**2 - 0.015*np.log10(wavelength**2)
        elif f < 1.1: #in GHz
            result[i] = 0.029 - 0.172*wavelength**2 - 0.067*np.log10(wavelength**2)
        else:
            logging.error('Only for frequencies below 12 GHz.')
            result[i] = np.nan
    return result

#import ds9 region file
region_list = Regions.read(region, format='ds9')
sky_region = region_list[0]


#define function to extract flux from region
def flux_measurement(image):
    header = image.img_hdr
    data = image.img_data
    wcs = image.get_wcs()

    beam_area_pix = image.get_beam_area(unit='pixel')

    pix_region = sky_region.to_pixel(wcs=wcs)
    mask = pix_region.to_mask()
    mask_weight = mask.to_image(data.shape)
    #mask = sky_region.get_mask(hdu=image.img_hdu, shape=np.shape(data))
    #cutout_data = np.extract(mask, data)
    #nncutout = cutout_data[~np.isnan(cutout_data)]
    #cutout = mask.cutout(data)

    return np.nansum(data*mask_weight)/beam_area_pix, beam_area_pix

#get all files as elements of AllImages class
stokes_i_files = sorted(glob.glob(path+'/'+ basename +'0*I-image--conv.fits'))
stokes_q_files = sorted(glob.glob(path+'/'+ basename +'0*Q-image--conv.fits'))
stokes_u_files = sorted(glob.glob(path+'/'+ basename +'0*U-image--conv.fits'))

stokes_i = AllImages(stokes_i_files)
stokes_q = AllImages(stokes_q_files)
stokes_u = AllImages(stokes_u_files)

#get frequency out of freq.txt file
freq_list=np.loadtxt(freq_file)
freq_Ghz = np.array(freq_list)*1e-9 #convert to GHz
c = 2.99792458e8
wavelength_m = c/(freq_Ghz*1e9)

#get rms of p for each image
rms_list=np.loadtxt(rms_file)
rms_p = np.array(rms_list)

#get flux from all images
I_flux = []
Q_flux = []
U_flux = []

P_flux = []
pola_val = []
polf_val = []

for i,q,u,r in zip(stokes_i,stokes_q, stokes_u, rms_p):
    I = flux_measurement(i)[0]
    Q = flux_measurement(q)[0]
    U = flux_measurement(u)[0]

    I_flux.append(I)
    Q_flux.append(Q)
    U_flux.append(U)

    rms = r/flux_measurement(q)[1] #divide by the beam size in pixel
    
    #calculate RM synth values
    P = np.sqrt(Q**2 + U**2)
    P_corr = np.sqrt(P**2 - rms**2)
    polf = P_corr/I if I != 0 else np.nan
    pola = 0.5*np.arctan2(U,Q)

    P_flux.append(P_corr)
    polf_val.append(polf)
    pola_val.append(np.degrees(pola))

P_flux = np.array(P_flux)
pola_val = np.array(pola_val)
polf_val = np.array(polf_val)

#get polarisation fraction from RMsynth for comparison

hdu_polf = fits.open(name_polf)
polf_data = np.array(hdu_polf[0].data.squeeze())
header_polf = hdu_polf[0].header
wcs_polf = WCS(header_polf, naxis=2)

hdu_pol = fits.open(name_pol)
pol = np.array(hdu_pol[0].data.squeeze())
header_pol = hdu_pol[0].header

polf_list = []
for p in polf_data:
    #polf_pix_reg = sky_region.to_pixel(wcs=wcs_polf)
    #mask_polf = polf_pix_reg.to_mask(mode='center').to_image(p.shape)
    #polf_mean= np.nanmean(p[mask_polf.astype(bool)])

    pix_region_polf = sky_region.to_pixel(wcs=wcs_polf)
    mask_polf = pix_region_polf.to_mask(mode='center')
    mask_weight_polf = mask_polf.to_image(p.shape)
    weights = pol[mask_weight_polf.astype(bool)]
    weights_norm = weights/np.nansum(weights)
    weighted_mean = np.nansum(p[mask_weight_polf.astype(bool)]*weights_norm)
    #polf_mean = np.nanmean(mask_weight_polf*p)
    polf_list.append(weighted_mean)


#plot everything
fig, axes = plt.subplots(1, 3, figsize=(30, 8))

axes[0].plot(freq_Ghz, P_flux, 'o',color='tab:orange', label="Polarized Intensity (P)")
axes[0].plot(freq_Ghz, Q_flux, 'o',color='tab:blue', label="Flux in Q")
axes[0].plot(freq_Ghz, U_flux, 'o',color='tab:green', label="Flux in U")
axes[0].set_xlabel("Frequency (GHz)")
axes[0].set_ylabel("Polarised Intensity")
axes[0].legend()
#axes[0].set_title("Polarized Intensity vs Frequency")


axes[1].plot(freq_Ghz, polf_val, 'o', color='tab:blue', label="data")
axes[1].plot(freq_Ghz, polf_list, 'o', color='tab:green', label="RMsynth data")
axes[1].plot(freq_Ghz, model_polf(freq_Ghz),'-', color='tab:red', label='model')
axes[1].set_xlabel("Frequency (GHz)")
axes[1].set_ylabel("polarisation fraction")
axes[1].legend()

axes[2].plot(freq_Ghz, pola_val, 'o', color='tab:blue', label="data")
axes[2].plot(freq_Ghz, pola_val - np.degrees(mean_RM*wavelength_m**2), 'o', color='tab:green', label='corr data')
axes[2].plot(freq_Ghz, model_pola(freq_Ghz),'-', color='tab:red', label='model')
axes[2].set_xlabel("Frequency (GHz)")
axes[2].set_ylabel("polarisation angle")
axes[2].legend()

plt.tight_layout()
plt.savefig(name_out)