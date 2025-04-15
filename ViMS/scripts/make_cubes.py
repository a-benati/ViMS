#!/usr/bin/env python3

#Script to select the good channels for RM synthesis and make a cube out of the single Q, U and I images and create a model for the I cube


from astropy.io import fits
import numpy as np
from pyregion import get_mask
import os
import glob
import argparse


parser = argparse.ArgumentParser(description='Make cubes out of Stokes I, Q and U images')
parser.add_argument('image', type=str, help='Basename of the images to be used e.g obs07_cal_3c286-')
parser.add_argument('--path', type=str, help='Path to image folder')
parser.add_argument('--out', type=str, help='output image name')
parser.add_argument('--rms', type=float, help='RMS threshold for noise estimation', default=0.01)
parser.add_argument('--noise_center', type=int, nargs=2, help='Center of the box for noise estimation', default=[1781, 532])
parser.add_argument('--noise_box', type=int, nargs=2, help='Width of the box for noise estimation', default=[123, 82])
parser.add_argument('--x_cut', type=int, nargs=2, help='X cut for the image', default=[0, 2048])
parser.add_argument('--y_cut', type=int, nargs=2, help='Y cut for the image', default=[0, 2048])
parser.add_argument('--model_cube', action='store_true', help='Remove noise from Stokes I cube to create a model cube')
args = parser.parse_args()

#definne image names
path = args.path
imagename = args.image

input_image = path+'/'+imagename
if args.out is None:
   name_out = imagename+'IQUV-Cubes'
else:
   name_out = args.out
output_cube = path+'/STOKES_CUBES/'+name_out

image2D = input_image+'MFS-I-image.fits'

hdu_im = fits.open(image2D)[0]
head = fits.open(image2D)[0].header

#define noise estimation parameters

noise_center = args.noise_center
noise_box = args.noise_box
x_cut = args.x_cut
y_cut = args.y_cut

rms_thresh = args.rms


#***************************************************************************end inputs


if head['NAXIS']==4:
   hdu2D=hdu_im.data[0,0,:,:]

crpix1=head['CRPIX1']
crpix2=head['CRPIX1']



if not os.path.exists(path+'STOKES_CUBES'):
   os.system('mkdir '+path+'STOKES_CUBES')

rms_q=[]
rms_u=[]
img_q=[]
img_u=[]
img_i=[]
freq=[]

files=glob.glob(input_image+'*Q*image--conv.fits')

sorted_list = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))

for i in sorted_list:
   if 'MFS' not in i:
      if (not os.path.exists(i.replace('-Q-','-U-'))) or (not os.path.exists(i.replace('-Q-','-I-'))):
          print('Stokes U Image or I image  not found ',i)
      if (os.path.exists(i.replace('-Q-','-U-'))) and (os.path.exists(i.replace('-Q-','-I-'))):
         print('opening ',i)
         hdu_q = fits.open(i)[0]
         data_q = hdu_q.data.squeeze()
         head_q = hdu_q.header

         hdu_u = fits.open(i.replace('-Q-','-U-'))[0]
         data_u = hdu_u.data.squeeze()
         head_u = hdu_u.header

         hdu_i = fits.open(i.replace('-Q-','-I-'))[0]
         data_i=hdu_i.data.squeeze()


         data_rms_q  =data_q[noise_center[1]-noise_box[1]:noise_center[1]+noise_box[1],noise_center[0]-noise_box[0]:noise_center[0]+noise_box[0]]
         data_rms_u  =data_u[noise_center[1]-noise_box[1]:noise_center[1]+noise_box[1],noise_center[0]-noise_box[0]:noise_center[0]+noise_box[0]]

         if(~np.isnan(np.mean(data_rms_q))) and (~np.isnan(np.mean(data_rms_q))):
            q_noise=np.sqrt(np.mean(data_rms_q*data_rms_q))
            u_noise=np.sqrt(np.mean(data_rms_q*data_rms_q))
            if 0.5*(u_noise+q_noise) <= rms_thresh:
               print('noise q ',q_noise)
               rms_q.append(q_noise)
               img_q.append(data_q[y_cut[0]:y_cut[1],x_cut[0]:x_cut[1]])
               if 'CRVAL3' in head_q:
                  freq.append(head_q['CRVAL3'])
               else:
                  freq.append(head_q['FREQ'])
               rms_u.append(u_noise)
               img_u.append(data_u[y_cut[0]:y_cut[1], x_cut[0]:x_cut[1]])
               img_i.append(data_i[y_cut[0]:y_cut[1], x_cut[0]:x_cut[1]])


array_rms_q = np.array(rms_q)
array_rms_u = np.array(rms_u)
rms_p=0.5 * (array_rms_q + array_rms_u)

array_freq = np.array(freq)

array_q = np.array(img_q)
array_u = np.array(img_u)
array_i= np.array(img_i)

print(np.shape(array_i))


print('writing stokes cubes ',output_cube+'Q_cube.fits',output_cube+'U_cube.fits',output_cube+'I_cube.fits')


head['CRPIX1']=crpix1-x_cut[0]
head['CRPIX2']=crpix2-y_cut[0]

fits.writeto(output_cube+'Q_cube.fits', array_q, header=head, overwrite=True)
fits.writeto(output_cube+'U_cube.fits', array_u, header=head, overwrite=True)
fits.writeto(output_cube+'I_cube.fits', array_i, header=head, overwrite=True)

f_rms = open(output_cube+'-rms.txt', 'w')
np.savetxt(f_rms, rms_p)
f_rms.close()
print('wriritng ',output_cube+'-rms.txt')

f_freq= open(output_cube+'-freq.txt', 'w')
np.savetxt(f_freq, array_freq)
f_freq.close()
print('writing ',output_cube+'-freq.txt')


#remove noise from I cube to create a model cube
if args.model_cube:
   print('removing noise from I cube to create a model cube')
   name_out_masked = output_cube+'I_masked.fits'
   hdul = fits.open(output_cube+'I_cube.fits')
   data_cube = hdul[0].data
   masked_cube = np.empty_like(data_cube)

   for i in range(data_cube.shape[0]):
       slice_2d = data_cube[i, :, :]

       data_rms  =slice_2d[noise_center[1]-noise_box[1]:noise_center[1]+noise_box[1],noise_center[0]-noise_box[0]:noise_center[0]+noise_box[0]]
       noise = np.sqrt(np.mean(data_rms*data_rms))
       threshold = 4*noise

       masked_cube[i, :, :] = np.where(slice_2d >= threshold, slice_2d, np.nan)

   hdul[0].data = masked_cube
   hdul.writeto(name_out_masked, overwrite=True)
   print('writing ',name_out_masked)