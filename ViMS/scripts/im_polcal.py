import sys, os
sys.path.insert(0, '/angelina/meerkat_virgo/scripts_fra')
from utils import log, utils
#import utils
import glob
import time

#cal_ms = "/localwork/angelina/meerkat_virgo/Obs01_caldemo_2k/msdir/obs07_1k_demo-cal.ms"

##############################Command definitions#####################################################
def convolve_beam(obs_id, logger, path):
    import glob
    from lib_fits import AllImages
    """
    convolve the beam of all given images to a common size
    """
    im_name = f'{path}/CAL_IMAGES/{obs_id}_cal_3c286-'
    images = glob.glob(im_name + '0*image.fits')
    if not images:
        logger.error(f'Error in convolve_beam: No images found in {path}/CAL_IMAGES/')
        return
    all_images = AllImages([imagefile for imagefile in images])
    all_images.convolve_to()
    all_images.write('-conv')

#---------------------------------------------------------------------------------------

def make_cubes(logger, obs_id, path):
    import glob
    from astropy.io import fits
    import numpy as np
    """
    create image cubes out of the produced images by wsclean for Stokes I, Q and U
    """
    im_name = f'{path}/CAL_IMAGES/{obs_id}_cal_3c286-'
    MFS_I = im_name + 'MFS-I-image.fits'
    cube_name = f'{path}/STOKES_CUBES/{obs_id}_3c286_'
    output_cube = cube_name + 'IQUV-'
    hdu_im = fits.open(MFS_I)[0]
    head = fits.open(MFS_I)[0].header

    if head['NAXIS'] == 4:
        hdu2D = hdu_im.data[0, 0, :, :]

    noise_center = [1781, 532]
    noise_box = [123, 82]

    rms_q=[]
    rms_u=[]
    img_q=[]
    img_u=[]
    img_i=[]
    freq=[]

    files=glob.glob(im_name+'*Q*image--conv.fits')
    sorted_list = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    for i in sorted_list:
        if 'MFS' not in i:
            if (not os.path.exists(i.replace('-Q-','-U-'))) or (not os.path.exists(i.replace('-Q-','-I-'))):
                logger.error('Error in make_cubes: Stokes U Image or I image not found ',i)
                raise FileNotFoundError(f"Stokes U or I image not found for {i}")
            if (os.path.exists(i.replace('-Q-','-U-'))) and (os.path.exists(i.replace('-Q-','-I-'))):
                logger.info(f'make_cubes: opening {i}')
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

                if(~np.isnan(np.mean(data_rms_q))) and (~np.isnan(np.mean(data_rms_u))):
                    q_noise=np.sqrt(np.mean(data_rms_q*data_rms_q))
                    u_noise=np.sqrt(np.mean(data_rms_q*data_rms_q))
                    if 0.5*(u_noise+q_noise) <= 0.01:
                        logger.info(f'make_cubes: noise q: {q_noise}')
                        rms_q.append(q_noise)
                        img_q.append(data_q)
                        if 'CRVAL3' in head_q:
                            freq.append(head_q['CRVAL3'])
                        else:
                            freq.append(head_q['FREQ'])
                        rms_u.append(u_noise)
                        img_u.append(data_u)
                        img_i.append(data_i)
                else:
                    logger.error('Error in make_cubes: RMS calculation failed for Q and U images')
    
    array_rms_q = np.array(rms_q)
    array_rms_u = np.array(rms_u)
    rms_p=0.5 * (array_rms_q + array_rms_u)
    array_freq = np.array(freq)
    array_q = np.array(img_q)
    array_u = np.array(img_u)
    array_i= np.array(img_i)

    logger.info(f'make_cubes: writing Stokes cubes {output_cube}Q_cube.fits, {output_cube}U_cube.fits, {output_cube}I_cube.fits')

    fits.writeto(output_cube+'Q_cube.fits', array_q, header=head, overwrite=True)
    fits.writeto(output_cube+'U_cube.fits', array_u, header=head, overwrite=True)
    fits.writeto(output_cube+'I_cube.fits', array_i, header=head, overwrite=True)

    f_rms = open(output_cube+'rms.txt', 'w')
    np.savetxt(f_rms, rms_p)
    f_rms.close()
    logger.info(f' make_cube: Writing {output_cube}rms.txt')

    f_freq= open(output_cube+'freq.txt', 'w')
    np.savetxt(f_freq, array_freq)
    f_freq.close()
    logger.info(f'make_cube: writing {output_cube}freq.txt')

#-------------------------------------------------------------------

def stokesI_model(obs_id, path):
    import numpy as np
    from astropy.io import fits
    """
    create a background subtracted stokes I image to use for the RM synthesis
    """
    output_cube = f'{path}/STOKES_CUBES/{obs_id}_3c286_'
    hdul = fits.open(output_cube +'IQUV-I_cube.fits')
    data = hdul[0].data
    masked_data = np.empty_like(data)

    noise_center = [1781, 532]
    noise_box = [123, 82]

    for i in range(data.shape[0]):
        slice_2d = data[i, :, :]
        data_rms  =slice_2d[noise_center[1]-noise_box[1]:noise_center[1]+noise_box[1],noise_center[0]-noise_box[0]:noise_center[0]+noise_box[0]]
        noise = np.sqrt(np.mean(data_rms*data_rms))
        thresh = 4*noise
        masked_data[i, :, :] = np.where(slice_2d >= thresh, slice_2d, np.nan)
    
    hdul[0].data = masked_data
    hdul.writeto(output_cube+'IQUV-I_masked.fits', overwrite=True)

#-------------------------------------------------------------------

def rm_synth_param(obs_id, path, logger):
    import numpy as np
    import scipy.constants 
    """
    calculate RM synthesis parameters
    return values needed for rmsynth3d and final_rm_synth
    """
    output_cube = f'{path}/STOKES_CUBES/{obs_id}_3c286_'
    freq_list = np.loadtxt(output_cube + 'IQUV-freq.txt')
    rms_list = np.loadtxt(output_cube + 'IQUV-rms.txt')

    #convert frequencies to lambda squared
    lambda2_list = (scipy.constants.c / freq_list) ** 2

    #calculate RM synthesis parameters
    d_lambda2 = lambda2_list[0] - lambda2_list[1]  # first channel width
    D_lambda2 = lambda2_list[0] - lambda2_list[-1]  # total bandwidth
    W_far = 0.67 * (1 / lambda2_list[0] + 1 / lambda2_list[-1])  # calculate Faraday width
    d_phi = 2. * np.sqrt(3.) / D_lambda2
    phi_max = np.sqrt(3.) / d_lambda2

    # calculate theoretical noise
    sigma_p = 1. / np.sqrt(np.sum(1. / rms_list ** 2.))
    sigma_RM = (d_phi / 2.) / 8.  # HWHM/signal-to-noise

    logger.info(f'RM synthesis parameters: W_far={W_far}, d_phi={d_phi}, phi_max={phi_max}, sigma_p={sigma_p}, sigma_RM={sigma_RM}')

    return d_phi, phi_max, W_far, sigma_p, sigma_RM

#-------------------------------------------------------------------

def StokesI_MFS_noise(obs_id, logger, path):
    import numpy as np
    from astropy.io import fits
    """
    calucate the noise of the Stokes I MFS image
    """

    im_name = f'{path}/CAL_IMAGES/{obs_id}_cal_3c286-'
    MFS_I = im_name + 'MFS-I-image.fits'
    hdu_im = fits.open(MFS_I)[0]

    noise_center = [1781, 532]
    noise_box = [123, 82]

    data_i=hdu_im.data.squeeze()
    data_rms_i  =data_i[noise_center[1]-noise_box[1]:noise_center[1]+noise_box[1],noise_center[0]-noise_box[0]:noise_center[0]+noise_box[0]]
    if (~np.isnan(np.mean(data_rms_i))):
        noise = np.sqrt(np.mean(data_rms_i*data_rms_i))
    else:
        logger.Error('Error in StokesI_MFS_noise: RMS calculation failed for Stokes I image')
        raise ValueError("RMS calculation failed for Stokes I image")
    logger.info(f'StokesI_MFS_noise: calculated noise: {noise}')
    return noise

#-------------------------------------------------------------------

def final_rm_synth(obs_id, sigma_p, d_phi, logger, path):
    from astropy.io import fits
    import numpy as np
    """calculate the polarisation angle, polarisation fraction and RM maps
     from the rmsynth3d output
     """
    GRM = 0.5 #Galactic RM in rad/m2, we consider it a constant value without error over the cluster
    thresh_p = 6. #threshold in sigma for sigma_p
    thresh_i = 5. #threshold in sigma for sigma_i
    sigma_i = StokesI_MFS_noise(obs_id, logger, path)
    RMSF_FWHM = d_phi #theoretical value from RMsynth param 

    # names of output images
    name_out = f'{path}/STOKES_CUBES/{obs_id}_3c286-final'
    name_rm_cluster = name_out+'_RM.fits' #... name of RM image corrected for the Milky Way contribution
    name_err_rm_cluster = name_out+'_err_RM.fits' # name of error RM image
    name_p = name_out+'_P.fits' #... name of polarization image
    name_pola = name_out+'_pola.fits' #... name of polarization angle image
    name_polf = name_out+'_polf.fits' #... name of polarization fraction image

    # name of input images
    name_tot = f'{path}/STOKES_CUBES/{obs_id}_3c286-FDF_tot_dirty.fits'
    name_q = f'{path}/STOKES_CUBES/{obs_id}_3c286-FDF_real_dirty.fits'
    name_u = f'{path}/STOKES_CUBES/{obs_id}_3c286-FDF_im_dirty.fits'
    name_i = f'{path}/STOKES_CUBES/{obs_id}_3c286_IQUV-I_cube.fits'

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

    #build Faraday depth axis
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

    for yy in range(0, ny - 1):
        for xx in range(0, nx - 1):
            # Compute the f, q, u, and rm values at the peak position
            f = np.max(tot[:, yy, xx])
            q = cube_q[np.argmax(tot[:, yy, xx]), yy, xx]
            u = cube_u[np.argmax(tot[:, yy, xx]), yy, xx]
            i = img_i[0, yy, xx]
            rm = phi_axis[np.argmax(tot[:, yy, xx])]

            # Select only pixels detected in polarization above a certain threshold
            if f >= thresh_p * sigma_p and i >= thresh_i * sigma_i:
                # Correct for the Ricean bias and write p
                img_p[0, yy, xx] = np.sqrt(f * f - sigma_p * sigma_p)
                # Cluster's RM
                img_rm_cluster[0, yy, xx] = rm - GRM
                # Error on RM
                img_err_rm_cluster[0, yy, xx] = (RMSF_FWHM / 2) / (img_p[0, yy, xx] / sigma_p)
                # Polarization angle
                img_pola[0, yy, xx] = ((0.5 * np.arctan2(u, q)) - rm * lambda2_0) * (180.0 / np.pi)
            else:
                img_p[0, yy, xx] = np.nan
                img_rm_cluster[0, yy, xx] = np.nan
                img_err_rm_cluster[0, yy, xx] = np.nan
                img_pola[0, yy, xx] = np.nan

    # Compute polarization fraction map
    #img_polf = img_p / img_i
    img_polf = np.divide(img_p, img_i, out=np.full_like(img_i, np.nan), where=img_i != 0)

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

#-------------------------------------------------------------------

def create_region(obs_id, logger, path):
    from astropy.wcs import WCS
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    from skimage.measure import find_contours
    import numpy as np
    from regions import PolygonSkyRegion, Regions
    import matplotlib.pyplot as plt
    """
    create a region file for the polarisation calibrator using a 10sig contour
    of the MFS I image
    """
    #load image
    MFS_I = f'{path}/CAL_IMAGES/{obs_id}_cal_3c286-MFS-I-image.fits'
    hdu_im = fits.open(MFS_I)[0]
    data = hdu_im.data.squeeze()
    header = hdu_im.header
    wcs_all = WCS(header)
    wcs = wcs_all.sub(['longitude', 'latitude'])

    #estimate background RMS
    noise = StokesI_MFS_noise(obs_id, logger, path)

    #determine contour for central source
    contours = find_contours(data, 15*noise)
    #ny, nx = data.shape
    #center_pixel = ((nx-1)/2, (ny-1)/2)
    ra_3c286 = 202.78481  # 13h31m08.354s
    dec_3c286 = 30.50911  # +30d30m32.96s
    source_coord = SkyCoord(ra=ra_3c286, dec=dec_3c286, unit='deg')

    best_contour = None
    best_distance = np.inf

    for contour in contours:
        centroid = np.mean(contour,axis=0) #calculate center of each contour
        centroid_sky  = wcs.pixel_to_world(centroid[1], centroid[0])
        #dist = np.hypot(centroid[1] - center_pixel[0], centroid[0] - center_pixel[1])
        dist = centroid_sky.separation(source_coord).arcsec
        if dist < best_distance:
            best_distance = dist
            best_contour = contour
    
    if best_contour is None:
        logger.error('Error in create_region: No contour found')
        raise ValueError("No contour found")
    
    logger.info(f'create_region: found contour with distance {best_distance} arcsec from poaition of 3C286')

    #convert to sky coordinates
    y_coords = best_contour[:, 0]
    x_coords = best_contour[:, 1]
    sky_coords = wcs.pixel_to_world(x_coords, y_coords)

    region = PolygonSkyRegion(vertices=sky_coords)
    reg = Regions([region])
    reg.write(f'{path}/STOKES_CUBES/{obs_id}_3c286_StokesI_region.reg', format='ds9', overwrite=True)
    logger.info(f'create_region: created region file {path}/STOKES_CUBES/{obs_id}_3c286_StokesI_region.reg')

    #sanity check: plot image with region overlayed
    min = np.nanmin(data)
    max = np.nanmax(data)
    out = f'{path}/PLOTS/{obs_id}_3c286_StokesI_region.png'

    plt.figure(figsize=(8,8))
    plt.imshow(data, origin='lower', cmap='viridis', vmin=min, vmax=max)
    plt.plot(x_coords, y_coords, color='red', linewidth=0.5)
    plt.title("10-sigma Contour for the Central Source")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.savefig(out)

    return reg

#-------------------------------------------------------------------

def ionospheric_RM(obs_id, cal_ms, path):
    import RMextract.getRM as gt
    import numpy as np
    from casatools import table
    """
    calculate the ionospheric RM contribution in direction of the polarisation calibrator
    for the specified observation
    """
    msdir = cal_ms
    ionex_dir = f'{path}/IONEX_DATA/'

    pointing = [3.539257790414, 0.53248520675] #direction of 3C286
    field_id = 1
    ref_ant = 'm000'

    tec = gt.getRM(MS=msdir, ionexPath=ionex_dir, server='ftp://gssc.esa.int/gnss/products/ionex/',earth_rot=0.5,ha_limit=1*np.pi, radec=pointing, prefix='UQRG', out_file=f'{path}/STOKES_CUBES/{obs_id}_3c286-ioncorr.txt')
    times_tot = np.squeeze(tec['times'])
    flags = tec['flags'][ref_ant]
    maskeddata=np.ma.array(tec['RM'][ref_ant],mask=np.logical_not(flags))
    RM = np.squeeze(maskeddata)

    tb = table()
    tb.open(msdir)
    times_pol = tb.query(f'FIELD_ID == {field_id}').getcol('TIME')
    tb.close()

    #calculate the mean RM in the time range of the polarisation calibrator observation
    start_time = times_pol[0]
    end_time = times_pol[-1]

    time_pol = []
    rm_pol = []

    for ind, t in enumerate(times_tot):
        if t <= end_time and t >= start_time:
            time_pol.append(t)
            y = RM[ind]
            rm_pol.append(y)
    
    RM_avg = np.mean(rm_pol)
    return RM_avg

#-------------------------------------------------------------------

def plot_results(obs_id, logger, path):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    import matplotlib.pyplot as plt
    """
    plot the results of RM synthesis for the polarisation calibrator
    and caluclate the mean RM, polarisation angle and polarisation fraction for the source region 
    """
    #input images
    name_p = f'{path}/STOKES_CUBES/{obs_id}_3c286-final_P.fits'
    name_polf = f'{path}/STOKES_CUBES/{obs_id}_3c286-final_polf.fits'
    name_pola = f'{path}/STOKES_CUBES/{obs_id}_3c286-final_pola.fits'
    name_rm = f'{path}/STOKES_CUBES/{obs_id}_3c286-final_RM.fits'
    name_err_rm = f'{path}/STOKES_CUBES/{obs_id}_3c286-final_err_RM.fits'
    name_stokesI = f'{path}/CAL_IMAGES/{obs_id}_cal_3c286-MFS-I-image.fits'
    region = create_region(obs_id, logger, path)

    mean_freq = 1.14e9 #mean frequency in Hz
    cutout_size = (100, 100)

    freq_list = np.loadtxt(f'{path}/STOKES_CUBES/{obs_id}_3c286_IQUV-freq.txt')
    hdu_I = fits.open(name_stokesI)
    header_i = hdu_I[0].header
    wcs_all = WCS(header_i)
    wcs = wcs_all.sub(['longitude', 'latitude'])
    center_coord = SkyCoord(ra=header_i['CRVAL1'], dec=header_i['CRVAL2'], unit='deg')

    hdu_p = fits.open(name_p)
    p = np.array(hdu_p[0].data.squeeze())
    header_p = hdu_p[0].header
    hdu_pola = fits.open(name_pola)
    pola = np.array(hdu_pola[0].data.squeeze())
    header_pola = hdu_pola[0].header

    freq_idx = np.argmin(np.abs(freq_list - mean_freq))
    hdu_polf = fits.open(name_polf)
    polf = np.array(hdu_polf[0].data.squeeze())
    polf_mean = polf[freq_idx]
    header_polf = hdu_polf[0].header

    hdu_rm = fits.open(name_rm)
    rm = np.array(hdu_rm[0].data.squeeze())
    header_rm = hdu_rm[0].header

    hdu_err_rm = fits.open(name_err_rm)
    err_rm = np.array(hdu_err_rm[0].data.squeeze())
    header_err_rm = hdu_err_rm[0].header

    sky_region = region[0]

    cutout_data = []
    cutout_regions = []

    #create cutout for each image
    for data, header in zip([p, pola, polf_mean, rm, err_rm], [header_p, header_pola, header_polf, header_rm, header_err_rm]):
        cutout = Cutout2D(data, position=center_coord, size=cutout_size, wcs=wcs)
        cutout_data.append(cutout.data)
        cutout_regions.append(sky_region.to_pixel(cutout.wcs))

    fig, axs = plt.subplots(1, 5, figsize=(5 * 5, 5), constrained_layout=True)

    for ax, i, name in zip(axs, cutout_data, ['Polarised intensity', 'polarisation angle', 'polarisation fraction','Rotation measure','Rotation measure error']):
        mask = cutout_regions[0].to_mask(mode='center').to_image(i.shape)
        weights = cutout_data[0][mask.astype(bool)]
        weights_norm = weights/np.nansum(weights)

        if name == 'polarisation angle':
            i = i % 360
            weighted_mean = np.nansum(i[mask.astype(bool)]*weights_norm)
            mean_value = np.nanmean(i[mask.astype(bool)])
            min = np.nanmin(i[mask.astype(bool)])
            max = np.nanmax(i[mask.astype(bool)])
        else:
            weighted_mean = np.nansum(i[mask.astype(bool)]*weights_norm)
            mean_value = np.nanmean(i[mask.astype(bool)])
            min = np.nanmin(i[mask.astype(bool)])
            max = np.nanmax(i[mask.astype(bool)])

        #correct rotation measure value for ionospheric contribution
        if name == 'Rotation measure':
            #ionospheric_rm = ionospheric_RM(obs_id, cal_ms, path)
            #weighted_mean_ioncorr = weighted_mean - ionospheric_rm
            #mean_value_ioncorr = mean_value - ionospheric_rm

            im = ax.imshow(i, origin='lower', cmap='viridis', interpolation='none', vmin=min, vmax=max)
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.set_title(name)

            cutout_regions[0].plot(ax=ax, lw=2)
            ax.text(0.5, 0.9, f" weighted mean: {weighted_mean:.3f}", color='black', fontsize=10, transform=ax.transAxes, bbox=dict(facecolor='white'))
            ax.text(0.05, 0.9, f"Mean: {mean_value:.3f}", color='black', fontsize=10, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)

        else:
            im = ax.imshow(i, origin='lower', cmap='viridis', interpolation='none', vmin=min, vmax=max)
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.set_title(name)

            cutout_regions[0].plot(ax=ax, lw=2)
            ax.text(0.5, 0.9, f" weighted mean: {weighted_mean:.3f}", color='black', fontsize=10, transform=ax.transAxes, bbox=dict(facecolor='white'))
            ax.text(0.05, 0.9, f"Mean: {mean_value:.3f}", color='black', fontsize=10, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
 

    plt.savefig(f'{path}/PLOTS/{obs_id}_3c286_RMsynth_param.png')
    return f'{path}/PLOTS/{obs_id}_3c286_RMsynth_param.png'

#-------------------------------------------------------------------

def plot_results_from_im(obs_id, logger, path):
    import numpy as np
    import glob
    from lib_fits import AllImages
    from astropy.io import fits
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt
    """
    calculate the RM synthesis parameters directly from the Q and U images
    for comparision to the values from rmsynth3d
    should be the same
    """
    
    def model_pola(nu):
        """
        model of frequency dep. polarisation angle of 3C286 Hugo & Perley (2024)
        """
        c = 2.99792458e8 #speed of light in m/s
        result = np.zeros_like(nu)
    
        for i,f in enumerate(nu):
            wavelength = c/(f*1e9) #in m
            if f >= 1.7 and f <= 12: #in GHz
                result[i] = 32.64 - 85.37*wavelength**2
            elif f < 1.7: #in GHz
                result[i] = 29.53 + wavelength**2*(4005.88*np.log10(f)**3 - 39.38)
            else:
                logger.error('Error in plot_results_form_im: Model only for frequencies below 12 GHz.')
                result[i] = np.nan
        return result
    
    def model_polf(nu):
        c = 2.99792458e8 #speed of light in m/s
        result = np.zeros_like(nu)

        for i,f in enumerate(nu):
            wavelength = c/(f*1e9)

            if f <= 12 and f >= 1.1: #in GHz
                result[i] = 0.080 - 0.053*wavelength**2 - 0.015*np.log10(wavelength**2)
            elif f < 1.1: #in GHz
                result[i] = 0.029 - 0.172*wavelength**2 - 0.067*np.log10(wavelength**2)
            else:
                logger.error('Error in plot_results_from_im: Model only for Frequencies below 12GHz.')
                result[i] = np.nan
        return result
    
    def flux_measurement(image):
        data = image.img_data
        wcs = image.get_wcs()

        beam_area_pix = image.get_beam_area(unit='pixel')
        pix_region = sky_region.to_pixel(wcs=wcs)
        mask = pix_region.to_mask()
        mask_weight = mask.to_image(data.shape)

        return np.nansum(data*mask_weight)/beam_area_pix, beam_area_pix

    
    #get all image files from wsclean
    basename = f'{path}/CAL_IMAGES/{obs_id}_cal_3c286-'
    stokesI_files = sorted(glob.glob(basename+'0*I-image--conv.fits'))
    stokesQ_files = sorted(glob.glob(basename+'0*Q-image--conv.fits'))
    stokesU_files = sorted(glob.glob(basename+'0*U-image--conv.fits'))

    stokesI = AllImages(stokesI_files)
    stokesQ = AllImages(stokesQ_files)
    stokesU = AllImages(stokesU_files)

    #get freq and rms data
    freq_list = np.loadtxt(f'{path}/STOKES_CUBES/{obs_id}_3c286_IQUV-freq.txt')
    freq_Ghz = np.array(freq_list)*1e-9 #convert to GHz
    c = 2.99792458e8
    wavelength_m = c/(freq_Ghz*1e9)

    rms_list = np.loadtxt(f'{path}/STOKES_CUBES/{obs_id}_3c286_IQUV-rms.txt')
    rms_arr = np.array(rms_list)

    sky_region = create_region(obs_id, logger, path)[0]
    #mean_rm = ionospheric_RM(obs_id, cal_ms, path)

    #get flux from all images
    I_flux = []
    Q_flux = []
    U_flux = []

    P_flux = []
    pola_val = []
    polf_val = []

    for i,q,u,r in zip(stokesI,stokesQ,stokesU,rms_arr):
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

    #get polf from rmsynth3d for comparision

    hdu_polf = fits.open(f'{path}/STOKES_CUBES/{obs_id}_3c286-final_polf.fits')
    polf_data = np.array(hdu_polf[0].data.squeeze())
    header_polf = hdu_polf[0].header
    wcs_polf = WCS(header_polf, naxis=2)

    hdu_pol = fits.open(f'{path}/STOKES_CUBES/{obs_id}_3c286-final_P.fits')
    pol = np.array(hdu_pol[0].data.squeeze())
    header_pol = hdu_pol[0].header

    polf_list = []
    for p in polf_data:
        pix_region_polf = sky_region.to_pixel(wcs=wcs_polf)
        mask_polf = pix_region_polf.to_mask(mode='center')
        mask_weight_polf = mask_polf.to_image(p.shape)
        weights = pol[mask_weight_polf.astype(bool)]
        weights_norm = weights/np.nansum(weights)
        weighted_mean = np.nansum(p[mask_weight_polf.astype(bool)]*weights_norm)
        polf_list.append(weighted_mean)


    #plot everything
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    axes[0].plot(freq_Ghz, P_flux, 'o',color='tab:orange', label="Polarized Intensity (P)")
    axes[0].plot(freq_Ghz, Q_flux, 'o',color='tab:blue', label="Flux in Q")
    axes[0].plot(freq_Ghz, U_flux, 'o',color='tab:green', label="Flux in U")
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Polarised Intensity")
    axes[0].legend()

    axes[1].plot(freq_Ghz, polf_val, 'o', color='tab:blue', label="data")
    axes[1].plot(freq_Ghz, polf_list, 'o', color='tab:green', label="RMsynth data")
    axes[1].plot(freq_Ghz, model_polf(freq_Ghz),'-', color='tab:red', label='model')
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel("polarisation fraction")
    axes[1].legend()

    axes[2].plot(freq_Ghz, pola_val, 'o', color='tab:blue', label="data")
    #axes[2].plot(freq_Ghz, pola_val - np.degrees(mean_rm*wavelength_m**2), 'o', color='tab:green', label='corr data')
    axes[2].plot(freq_Ghz, model_pola(freq_Ghz),'-', color='tab:red', label='model')
    axes[2].set_xlabel("Frequency (GHz)")
    axes[2].set_ylabel("polarisation angle")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{path}/PLOTS/{obs_id}_3c286_RMsynth_param_from_im.png')
    return f'{path}/PLOTS/{obs_id}_3c286_RMsynth_param_from_im.png'
#---------------------------------------------------------------------------

def run(logger, obs_id, pol_ms, path):
    from casatools import msmetadata
    import numpy as np
    """
    Image the polarisation calibrator via WSClean and determine the RM synthesis parameters
    of it with RMsynth3d for the given Observation ID.
    corrected for ionospheric contribution
    """
    logger.info("")
    logger.info("")
    logger.info("##########################################################")
    logger.info("###################### IMAGE POLCAL ######################")
    logger.info("##########################################################")
    logger.info("")
    logger.info("")

    #log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("#################### IMAGE POLCAL ####################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("IMAGE POLCAL", "Started", warnings="", plot_link="")


    # image polarisation calibrator with Wsclean
    try:
        logger.info("\n\n\n\n\n")
        logger.info("IMAGE POLCAL: starting WSClean...")

        im_name = f'{path}/CAL_IMAGES/{obs_id}_cal_3c286'

        msmd = msmetadata()
        msmd.open(pol_ms)
        chan_width = np.mean(msmd.chanwidths(0))*1e-3 # convert to MHz
        msmd.close()
        cutoff = int((1380 - 900)/chan_width) # cutoff at 1380 MHz to avoid off-axis leakage

        cmd = f"wsclean -name {im_name} -size 2048 2048 -scale 1.3asec -mgain 0.8 -niter 30000 -auto-threshold 0.5 -auto-mask 2.5 \
                -field 0 -pol iquv -weight briggs -0.5 -j 32 -abs-mem 100.0 -channels-out 10 -join-channels -gridder wgridder -no-update-model-required \
                -squared-channel-joining -join-polarizations -fit-spectral-pol 4 -multiscale  -multiscale-scales 0,2,3,6 -multiscale-scale-bias 0.75 \
                -parallel-deconvolution 1000 -parallel-gridding 1 -channel-range 0 {cutoff} -nwlayers-factor 3 -minuvw-m 40 -no-mf-weighting -weighting-rank-filter 3 \
                -data-column CORRECTED_DATA {pol_ms}"
        stdout, stderr = utils.run_command(cmd)
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in WSClean: {stderr}")
        
        logger.info('IMAGE POLCAL: finished WSClean')
        logger.info("")
        logger.info("")
        logger.info("")
        #log.append_to_google_doc('IMAGE POLCAL', 'Finished WSClean', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    except Exception as e:
            logger.exception("Error while running WSClean")

    # Comvolve beam to smallest common size
    try:
        logger.info("\n\n\n\n\n")
        logger.info("IMAGE POLCAL: convolving beam...")
        convolve_beam(obs_id, logger, path)
        logger.info('IMAGE POLCAL: finished beam convolution')
        logger.info("")
        logger.info("")
        logger.info("")
        #log.append_to_google_doc('IMAGE POLCAL', 'Finished beam convolution', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    except Exception as e:
        logger.exception("Error while convolving beam to smallest common size")

    # create Image cubes and model of Stokes I image
    try:
        logger.info("\n\n\n\n\n")
        logger.info("IMAGE POLCAL: creating image cubes...")
        make_cubes(logger, obs_id, path)
        logger.info('IMAGE POLCAL: finished creating image cubes')
        logger.info("")
        logger.info("")
        logger.info("")
        #log.append_to_google_doc('IMAGE POLCAL', 'Finished creating image cubes', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    except Exception as e:
        logger.exception("Error while creating image cubes")

    #create background subtratced Stokes I image
    try:
        logger.info("\n\n\n\n\n")
        logger.info("IMAGE POLCAL: creating creating model Stokes I cube...")
        stokesI_model(obs_id, path)
        logger.info('IMAGE POLCAL: finished creating model Stokes I cube')
        logger.info("")
        logger.info("")
        logger.info("")
        #log.append_to_google_doc('IMAGE POLCAL', 'Finished creating model Stokes I cube', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    except Exception as e:
        logger.exception("Error while creating model Stokes I cube")

    #calculate RM synthesis parameters
    try:
        logger.info("\n\n\n\n\n")
        logger.info("IMAGE POLCAL: calculating RM synthesis parameters...")
        d_phi, phi_max, W_far, sigma_p, sigma_RM = rm_synth_param(obs_id, path, logger)
        logger.info('IMAGE POLCAL: finished calculating RM synthesis parameters')
        logger.info("") 
        logger.info("")
        logger.info("")
        #log.append_to_google_doc('IMAGE POLCAL', 'Finished calculating RMsynth paramters', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    except Exception as e:
        logger.exception("Error while calculating RM synthesis parameters")
        
    #run rmsynth3d
    try:
        logger.info("\n\n\n\n\n")
        logger.info("IMAGE POLCAL: running rmsynth3d...")
        cube_name = f'{path}/STOKES_CUBES/{obs_id}_3c286_IQUV-'
        rm_name = f'{obs_id}_3c286-'
        cmd = f"export PYTHONPATH=/opt/RM-Tools:$PYTHONPATH && python3 /opt/RM-Tools/RMtools_3D/do_RMsynth_3D.py {cube_name}Q_cube.fits {cube_name}U_cube.fits {cube_name}freq.txt -i {cube_name}I_masked.fits -n {cube_name}rms.txt -v -l {phi_max} -s 30 -w 'variance' -o {rm_name}"
        logger.info(f"IMAGE POLCAL: Executing command: {cmd}")
        stdout, stderr = utils.run_command(cmd)
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in RMsynth: {stderr}")
        logger.info("IMAGE POLCAL: finished rmsynth3d")
        logger.info("")
        logger.info("")
        logger.info("")
        #log.append_to_google_doc("IMAGE POLCAL", "Finished RMsynth3d", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    except Exception as e:
        logger.exception("Error while running rmsynth3d")

    #create maps of polarisation angle, fraction and RM value for the polarisation calibrator
    try:
        logger.info("\n\n\n\n\n")
        logger.info("IMAGE POLCAL: running final RM synth...")
        final_rm_synth(obs_id, sigma_p, d_phi, logger, path)
        logger.info('IMAGE POLCAL: finished final RM synth')
        logger.info("")
        logger.info("")
        logger.info("")
        #log.append_to_google_doc('IMAGE POLCAL', 'Finished final RM synth', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    except Exception as e:
        logger.exception("Error while running final RM synth")

    #calculate the results for the source region and plot them as a png
    try:
        logger.info("\n\n\n\n\n")
        logger.info("IMAGE POLCAL: plotting results...")
        plot = plot_results(obs_id, logger, path)
        logger.info('IMAGE POLCAL: finished plotting results')
        logger.info("")
        logger.info("")
        logger.info("")
        #log.append_to_google_doc('IMAGE POLCAL', 'Finished plotting results', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    except Exception as e:
        logger.exception("Error while plotting results")

    #plot_links = []
    #plot_links.append(log.upload_plot_to_drive(plot))
    #logger.info(plot_links)
    #for plot_link in plot_links:
        #log.append_to_google_doc("IMAGE POLCAL", "Plotted 3C286 with RM parameters", warnings="", plot_link=plot_link, doc_name="ViMS Pipeline Plots")

    #calculate RM values from image for comparision with RM values from RMsynth3d
    try:
        logger.info("\n\n\n\n\n")
        logger.info("IMAGE POLCAL: calculating RM values from image...")
        plot_im = plot_results_from_im(obs_id, logger, path)
        logger.info('IMAGE POLCAL: finished calculating RM values from image')
        logger.info("")
        logger.info("")
        logger.info("")
        #log.append_to_google_doc('IMAGE POLCAL', 'Finished calculating RM values from image', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    except Exception as e:
        logger.exception("Error while calculating RM values from image")

    #plot_links = []
    #plot_links.append(log.upload_plot_to_drive(plot_im))
    #logger.info(plot_links)
    #for plot_link in plot_links:
        #log.append_to_google_doc("IMAGE POLCAL", "Plotted 3C286 with RM parameters", warnings="", plot_link=plot_link, doc_name="ViMS Pipeline Plots")

        
    logger.info("image polcal step completed successfully!")
    logger.info("######################################################")
    logger.info("################## END IMAGE POLCAL ##################")
    logger.info("######################################################")
    logger.info("")
    logger.info("")
    #log.append_to_google_doc("Image polcal step completed successfully!", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("###################### END IMAGE POLCAL ######################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")


