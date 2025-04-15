#!/usr/bin/env python3

# excute with python3:
# exec(open("RMsynth_parameters.py").read())
# or with ipython3:
# %run RMsynth_parameters.py

import numpy as np
import scipy.constants 
import argparse

parser = argparse.ArgumentParser(description='Calculate RM synthesis parameters')
parser.add_argument('rms_file', type=str, help='RMS filename')
parser.add_argument('freq_file', type=str, help='Frequency filename')
parser.add_argument('--path', type=str, help='Path to folder')
args = parser.parse_args()

input_freq = args.path+'/STOKES_CUBES/' + args.freq_file
input_rms = args.path+'/STOKES_CUBES/' + args.rms_file

#read from the file the list of frequencies and of rms for each channel
freq_list=np.loadtxt(input_freq)
rms_list=np.loadtxt(input_rms)

#convert frequencies in wavelength squared
lambda2_list=(scipy.constants.c/freq_list)**2

##### compute the RM synthesis parameters #####

d_lambda2=lambda2_list[0]-lambda2_list[1]  #first channel width

D_lambda2=lambda2_list[0]-lambda2_list[-1]  #total bandwidth

W_far = 0.67*(1/lambda2_list[0] + 1/lambda2_list[-1])#calculate Faraday width

d_phi=2.*np.sqrt(3.)/D_lambda2

phi_max=np.sqrt(3.)/d_lambda2

#### compute theoretical noise 

sigma_p=1./np.sqrt(np.sum(1./rms_list**2.))

sigma_RM=(d_phi/2.)/8. # HWHM/signal-to-noise 

#print results

print('Theoretical FWHM of the RMTF:'+str(d_phi)+' rad/m2')
print('Maximum observable phi:'+str(phi_max)+' rad/m2')
print('Faraday width:'+ str(W_far)+' rad/m2')
print('Theoretical noise in P:'+str(sigma_p)+' Jy/beam')
print('Theoretical noise in RM at 8sigma:'+str(sigma_RM)+' rad/m2')

# to run RM synthesis
#rmsynth3d PSZ287_C_64chnQU-Q-cube.fits PSZ287_C_64chnQU-U-cube.fits PSZ287_C_64chnQU-freq.txt -l <maximum observable phi in rad/m2> -d <required resolution in rad/m2> -v




