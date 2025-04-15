import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
import argparse


parser = argparse.ArgumentParser(description='Plot polarisation properties')
parser.add_argument('--basename', type=str, help='Base name of the polarisation images e.g obs02_3c286_mask-final')
parser.add_argument('--name_stokeI', type=str, help='Name of the Stokes I image')
parser.add_argument('--region_name', type=str, help='Name of the region file')
parser.add_argument('--freq_list', type=str, help='Name of the frequency list file')
parser.add_argument('--name_out', type=str, help='Name of the output image')
parser.add_argument('--path', type=str, help='Path to the folder')
parser.add_argument('--im_size', type=int, nargs=2, help='Cutout size in pixels', default=(100, 100))
parser.add_argument('--weighted', action='store_true', help='display weighted mean value of image?')
args = parser.parse_args()


path = args.path
name_p = args.path +'/STOKES_CUBES/' + args.basename +'_P.fits'
name_polf = args.path +'/STOKES_CUBES/' + args.basename +'_polf.fits'
name_pola = args.path +'/STOKES_CUBES/' + args.basename +'_pola.fits'
name_rm = args.path +'/STOKES_CUBES/' + args.basename +'_RM.fits'
name_err_rm = args.path +'/STOKES_CUBES/' + args.basename +'_err_RM.fits'
name_stokeI = args.path+'/' +args.name_stokeI
region_name = args.path +'/STOKES_CUBES/' + args.region_name
freq_list = np.loadtxt(args.path +'/STOKES_CUBES/' + args.freq_list)

name_out = args.path +'/STOKES_CUBES/' + args.name_out

mean_freq = 1.14e9 #Hz

cutout_size = args.im_size #in pixel


hdu_I = fits.open(name_stokeI)
header_i = hdu_I[0].header
wcs_all = WCS(header_i)
wcs = wcs_all.sub(['longitude', 'latitude'])

center_coord = SkyCoord(ra=header_i['CRVAL1'], dec=header_i['CRVAL2'], unit='deg')  # Central RA, Dec

hdu_p = fits.open(name_p)
p = np.array(hdu_p[0].data.squeeze())
header_p = hdu_p[0].header
#wcs_p = WCS(header_p)

hdu_pola = fits.open(name_pola)
pola = np.array(hdu_pola[0].data.squeeze())
header_pola = hdu_pola[0].header
#wcs_pola = WCS(header_pola)

freq_idx = np.argmin(np.abs(freq_list - mean_freq))

hdu_polf = fits.open(name_polf)
polf = np.array(hdu_polf[0].data.squeeze())
polf_mean = polf[freq_idx]
header_polf = hdu_polf[0].header
#wcs_polf = WCS(header_polf, naxis=2)


hdu_rm = fits.open(name_rm)
rm = np.array(hdu_rm[0].data.squeeze())
header_rm = hdu_rm[0].header
#wcs_rm = WCS(header_rm)

hdu_err_rm = fits.open(name_err_rm)
err_rm = np.array(hdu_err_rm[0].data.squeeze())
header_err_rm = hdu_err_rm[0].header
#wcs_err_rm = WCS(header_err_rm)

region_list = Regions.read(region_name, format='ds9')
sky_region = region_list[0]


cutout_data = []
cutout_regions = []

# Create cutouts for each image
for data, header in zip([p, pola, polf_mean, rm, err_rm], [header_p, header_pola, header_polf, header_rm, header_err_rm]):
    # Create the cutout
    cutout = Cutout2D(data, position=center_coord, size=cutout_size, wcs=wcs)
    cutout_data.append(cutout.data)
    cutout_regions.append(sky_region.to_pixel(cutout.wcs))



fig, axs = plt.subplots(1, 5, figsize=(5 * 5, 5), constrained_layout=True)

for ax, i, name in zip(axs, cutout_data, ['Polarised intensity', 'polarisation angle', 'polarisation fraction','Rotation measure','Rotation measure error']):
    mask = cutout_regions[0].to_mask(mode='center').to_image(i.shape)

    if args.weighted== True:
        weights = cutout_data[0][mask.astype(bool)]
        weights_norm = weights/np.nansum(weights)

        if name == 'polarisation angle':
            i = i % 360
            weighted_mean = np.nansum(i[mask.astype(bool)]*weights_norm)
        else:
            weighted_mean = np.nansum(i[mask.astype(bool)]*weights_norm)

    if name == 'polarisation angle':
        i = i % 360
        mean_value = np.nanmean(i[mask.astype(bool)])
        min = np.nanmin(i[mask.astype(bool)])
        max = np.nanmax(i[mask.astype(bool)])
    else:
        mean_value = np.nanmean(i[mask.astype(bool)])
        min = np.nanmin(i[mask.astype(bool)])
        max = np.nanmax(i[mask.astype(bool)])


    im = ax.imshow(i, origin='lower', cmap='viridis', interpolation='none', vmin=min, vmax=max)
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title(name)

    cutout_regions[0].plot(ax=ax, lw=2)

    if args.weighted == True:
        ax.text(0.5, 0.9, f" weighted mean: {weighted_mean:.3f}", color='black', fontsize=10, transform=ax.transAxes, bbox=dict(facecolor='white'))

    ax.text(0.05, 0.9, f"Mean: {mean_value:.3f}", color='black', fontsize=10, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)

plt.savefig(name_out, dpi=300)
