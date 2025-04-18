#!/usr/bin/env python3

# excute with python3:
# exec(open("final_RMsynth.py").read())
# or with ipython3:
# %run final_RMsynth.py

import numpy as np
from astropy.io import fits
from sys import exit
import argparse


parser = argparse.ArgumentParser(description='Calculate RM synthesis parameters')
parser.add_argument('basename', type=str, help='Basename of the FDF images to be used e.g obs02_3c286_mask-FDF')
parser.add_argument('name_i', type=str, help='Name of Stokes I cube')
parser.add_argument('--path', type=str, help='Path to folder')
parser.add_argument('--out', type=str, help='output image name')
parser.add_argument('--thresh_p', type=float, help='threshold in sigma for sigma_p', default=6.)
parser.add_argument('--thresh_i', type=float, help='threshold in sigma for sigma_i', default=5.)
parser.add_argument('--sigma_p', type=float, help='Theoretical noise in P')
parser.add_argument('--sigma_i', type=float, help='Theoretical noise in I')
parser.add_argument('--fwhm', type=float, help='Theoretical FWHM of the RMTF', default=64.6494)
args = parser.parse_args()

#names of input images
name_tot = args.path + '/STOKES_CUBES/' + args.basename +'_tot_dirty.fits'
name_q = args.path + '/STOKES_CUBES/' + args.basename +'_real_dirty.fits'
name_u = args.path + '/STOKES_CUBES/' + args.basename +'_im_dirty.fits'
name_i = args.path + '/STOKES_CUBES/' + args.name_i


GRM = 0.5 #Galactic RM in rad/m2, we consider it a constant value without error over the cluster
thresh_p=args.thresh_p #threshold in sigma for sigma_p
thresh_i=args.thresh_i #threshold in sigma for sigma_i

sigma_p = args.sigma_p#Jy/beam,read from the RMsynth_parameters.py script
sigma_i= args.sigma_i

RMSF_FWHM = args.fwhm #in rad/m2,read from the RMSF_FWHM.fits image or from the RMsynth_parameters.py script (theoretical value)

# output images names
name_out = args.path + '/STOKES_CUBES/' + args.out
name_rm_cluster = name_out+'_RM.fits' #... name of RM image corrected for the Milky Way contribution
name_err_rm_cluster = name_out+'_err_RM.fits' # name of error RM image
name_p = name_out+'_P.fits' #... name of polarization image
name_pola = name_out+'_pola.fits' #... name of polarization angle image
name_polf = name_out+'_polf.fits' #... name of polarization fraction image

# END OF INPUTS


#open input images

hdu_tot = fits.open(name_tot)
tot = np.array(hdu_tot[0].data) # [phi,y,x]
head = hdu_tot[0].header

hdu_q = fits.open(name_q)
cube_q = np.array(hdu_q[0].data)

hdu_u = fits.open(name_u)
cube_u = np.array(hdu_u[0].data)


hdu_i = fits.open(name_i)
img_i = np.array(hdu_i[0].data) # [Stokes=1, Frequency=1, y, x]
head_i = hdu_i[0].header

#build de Faraday depth axis

nphi = head['NAXIS3']
dphi = head[ 'CDELT3']
phi_axis = np.linspace(-int(nphi/2)*dphi,int(nphi/2)*dphi,nphi)

#check how many pixels are in one image
nx=head['NAXIS1'] 
ny=head['NAXIS2'] 

#check the observing wavelegth squared (remember shift theorem)
lambda2_0=head['LAMSQ0']

#initialize output images
img_p = np.zeros([1,ny,nx])
img_rm_cluster = np.zeros([1,ny,nx])
img_err_rm_cluster = np.zeros([1,ny,nx])
img_pola = np.zeros([1,ny,nx])

#obtain output values for every pixel

for yy in range (0,ny-1):
	for xx in range (0, nx-1):
		#compute the f, q, u and rm values at the peak position
		f = np.max(tot[:,yy,xx])
		q = cube_q[np.argmax(tot[:,yy,xx]),yy,xx]
		u = cube_u[np.argmax(tot[:,yy,xx]),yy,xx]
		i = img_i[0,yy,xx]
		rm = phi_axis[np.argmax(tot[:,yy,xx])]
		#select only pixels detected in polarization above a certain threshold
		if f>=thresh_p*sigma_p and i>=thresh_i*sigma_i:
			#correct for the ricean bias and write p
			img_p[0,yy,xx] = np.sqrt(f*f-sigma_p*sigma_p)
			#cluster's RM
			img_rm_cluster[0,yy,xx] = rm-GRM
			#error on RM
			img_err_rm_cluster[0,yy,xx] = (RMSF_FWHM/2)/(img_p[0,yy,xx]/sigma_p)
			#polarization angle
			img_pola[0,yy,xx] = ((0.5*np.arctan2(u,q))-rm*lambda2_0)*(180./np.pi)
		else:
			img_p[0,yy,xx]=np.nan
			img_rm_cluster[0,yy,xx]=np.nan
			img_err_rm_cluster[0,yy,xx]=np.nan
			img_pola[0,yy,xx]=np.nan

#compute polarization fraction map
img_polf=img_p/img_i

#Write the results in a fits file. We first modify the header to set the right units for each image

hdu_p = fits.PrimaryHDU(img_p,head_i)
hdu_p.writeto(name_p, overwrite=True) 

head_rm=head_i
head_rm['BUNIT']='rad/m/m'
hdu_rm = fits.PrimaryHDU(img_rm_cluster,head_rm)
hdu_rm.writeto(name_rm_cluster, overwrite=True) 

head_err_rm=head_i
head_err_rm['BUNIT']='rad/m/m'
hdu_err_rm = fits.PrimaryHDU(img_err_rm_cluster,head_err_rm)
hdu_err_rm.writeto(name_err_rm_cluster, overwrite=True) 

head_pola=head_i
head_pola['BUNIT']='deg'
hdu_pola = fits.PrimaryHDU(img_pola,head_pola)
hdu_pola.writeto(name_pola, overwrite=True) 

head_polf=head_i
head_polf['BUNIT']=''
hdu_polf = fits.PrimaryHDU(img_polf,head_polf)
hdu_polf.writeto(name_polf, overwrite=True) 





