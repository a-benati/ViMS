import sys, os
sys.path.insert(0, '/angelina/meerkat_virgo/scripts_fra')
from utils import log, utils
#import utils
import glob
import time

##############################Command definitions#####################################################
def get_target_from_obs_id(logger, obs_id):
    from casatools import msmetadata
    """
    Get all traget names from the given obs_id.
    """
    full_ms = glob.glob(f'/beegfs/bba5268/meerkat_virgo/raw/{obs_id}*sdp_l0.ms')[0]

    msmd = msmetadata()
    msmd.open(full_ms)
    fields = msmd.fieldnames()
    msmd.close()
    targets = []

    logger.info(f'determining the targte fields of file {full_ms}')
    for field in fields:
        if field == 'J1939-6342':
            pass
        elif field == 'J1331+3030':
            pass
        elif field == 'J1150-0023':
            pass
        else:
            targets.append(field)
    
    return targets
    
#---------------------------------------------------------------------------------------

def make_mosaic(logger, mosaic_obs_ids, mosaic_targets, output_dir):
    import glob
    """
    Create a mosaic of the specified targets from the given observation IDs.
    If no targets are specified, it will use all targets from the obs_ids.
    """
    if mosaic_targets == []:
        logger.info(f'make_mosaic: No targets specified, using all targets from obs_ids {mosaic_obs_ids}')
        for obs_id in mosaic_obs_ids:
            mosaic_targets.extend(get_target_from_obs_id(logger, obs_id))
    else:
        logger.info(f'make_mosaic: Using specified targets {mosaic_targets}')

    
    def gather_images(mosaic_targets, stokes, num):
        """
        Gather all images for the specified Stokes parameter and image number.
        Done for all targets in the mosaic_targets list.
        """
        images = []
        if isinstance(num, str):
            num_str = num
        else:
            num_str = str(num).zfill(4)
        for obs_id in mosaic_obs_ids:
            for target in mosaic_targets:
                pattern = f'{obs_id}/TARGET_IMAGES/{obs_id}_{target}_pol-{num_str}-{stokes}-image-pb.fits'
                images.extend(glob.glob(pattern))
        return images
    
    obs_id_str = "_".join(mosaic_obs_ids)
    for num in range(100):
        num_str = str(num).zfill(4)
        for stokes in ['Q', 'U', 'I']:
            image_list = gather_images(mosaic_targets, stokes, num)
            logger.info(f'make_mosaic: Found {len(image_list)} images for Stokes {stokes} and image number {num_str}')
            if len(image_list) > 1:
                output_mosaic = f'mosaics/mosaic_{obs_id_str}_{num_str}-{stokes}-image-pb.fits'
                cmd = f'python3.10 /angelina/meerkat_virgo/scripts_fra/mosaic.py --images {image_list} --output {output_mosaic}'
                stdout, stderr = utils.run_command(cmd, logger)
                logger.info(stdout)
                if stderr:
                    logger.error(f"Error in mosaic.py: {stderr}")
    image_list = gather_images(mosaic_targets, 'I', 'MFS')
    logger.info(f'Make mosaic out of the MFS Stokes I images. Found {len(image_list)} images')
    if len(image_list) > 1:
        output_mosaic = f'mosaics/mosaic_{obs_id_str}_MFS-I-image-pb.fits'
        cmd = f'python3.10 /angelina/meerkat_virgo/scripts_fra/mosaic.py --images {image_list} --output {output_mosaic}'
        stdout, stderr = utils.run_command(cmd, logger)
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in mosaic.py: {stderr}")
    
    return obs_id_str
#---------------------------------------------------------------------------------------

def make_cubes(logger, obs_id, target, path, obs_id_str=None, mode='single'):
    import glob
    from astropy.io import fits
    import numpy as np
    """
    create image cubes out of the produced images by wsclean for Stokes I, Q and U
    """

    #noise_center = [816,801]
    #noise_box = [216,143]
    noise_center = [200,200]
    noise_box = [143,143]
    #noise_center = [1781, 532]
    #noise_box = [123, 82]

    if mode == 'single':
        logger.info('make_cubes: processing single Stokes I, Q and U images')

        im_name = f'{path}/TARGET_IMAGES/{obs_id}_{target}_pol-'
        cube_name = f'{path}/STOKES_CUBES/{obs_id}_{target}_'
        files=glob.glob(im_name+'0*Q*image-pb.fits')

    elif mode == 'mosaic':
        logger.info(f'make_cubes: processing mosaic Stokes I, Q and U images')
        if obs_id_str is not None:
            im_name = f'mosaics/mosaic_{obs_id_str}_'
            cube_name = f'mosaics/mosaic_{obs_id_str}_'
            files=glob.glob(im_name+'0*Q*image-pb.fits')
        else:
            logger.error('Error in make_cubes: obs_id_str must be provided for mosaic mode')
            raise ValueError("obs_id_str must be provided for mosaic mode")


    MFS_I = im_name + 'MFS-I-image-pb.fits'           
    output_cube = cube_name + 'IQUV-'
    head = fits.open(MFS_I)[0].header

    rms_q=[]
    rms_u=[]
    img_q=[]
    img_u=[]
    img_i=[]
    freq=[]

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

                hdu_i = fits.open(i.replace('-Q-','-I-'))[0]
                data_i=hdu_i.data.squeeze()

                data_rms_q  =data_q[noise_center[1]-noise_box[1]:noise_center[1]+noise_box[1],noise_center[0]-noise_box[0]:noise_center[0]+noise_box[0]]
                data_rms_u  =data_u[noise_center[1]-noise_box[1]:noise_center[1]+noise_box[1],noise_center[0]-noise_box[0]:noise_center[0]+noise_box[0]]

                if(~np.isnan(np.nanmean(data_rms_q))) and (~np.isnan(np.nanmean(data_rms_u))):
                    q_noise=np.sqrt(np.nanmean(data_rms_q*data_rms_q))
                    u_noise=np.sqrt(np.nanmean(data_rms_u*data_rms_u))
                    if 0.5*(u_noise+q_noise) <= 0.001:
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
                rms_p= np.sqrt(array_rms_q**2 + array_rms_u**2)
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

def stokesI_model(obs_id, target, path, obs_id_str=None, mode='single'):
    import numpy as np
    from astropy.io import fits
    """
    create a background subtracted stokes I image to use for the RM synthesis
    """
    if mode == 'single':
        output_cube = f'{path}/STOKES_CUBES/{obs_id}_{target}_'
    elif mode == 'mosaic':
        output_cube = f'mosaics/mosaic_{obs_id_str}_'
    
    hdul = fits.open(output_cube + 'IQUV-I_cube.fits')
    data = hdul[0].data
    masked_data = np.empty_like(data)

    noise_center = [200,200]
    noise_box = [143,143]
    #noise_center = [816,801]
    #noise_box = [216,143]
    #noise_center = [1781, 532]
    #noise_box = [123, 82]

    for i in range(data.shape[0]):
        slice_2d = data[i, :, :]
        data_rms  =slice_2d[noise_center[1]-noise_box[1]:noise_center[1]+noise_box[1],noise_center[0]-noise_box[0]:noise_center[0]+noise_box[0]]
        noise = np.sqrt(np.nanmean(data_rms*data_rms))
        thresh = 2*noise
        masked_data[i, :, :] = np.where(slice_2d >= thresh, slice_2d, np.nan)
    
    hdul[0].data = masked_data
    hdul.writeto(output_cube+'IQUV-I_masked.fits', overwrite=True)

#-------------------------------------------------------------------

'''def normalize_stokes_cubes(obs_id, target, path, logger,obs_id_str=None, mode='single'):
    from astropy.io import fits
    import numpy as np
    """
    Divide the Stokes Q and U cubes by the Stokes I cubes
    in preparation for RM synthesis.
    """
    name_Qcube = f'{path}/STOKES_CUBES/{obs_id}_{target}_IQUV-Q_cube.fits'
    name_Ucube = f'{path}/STOKES_CUBES/{obs_id}_{target}_IQUV-U_cube.fits'
    name_Icube = f'{path}/STOKES_CUBES/{obs_id}_{target}_IQUV-I_cube.fits'

    if mode == 'mosaic':
        if obs_id_str is not None:
            name_Qcube = f'mosaics/mosaic_{obs_id_str}_IQUV-Q_cube.fits'
            name_Ucube = f'mosaics/mosaic_{obs_id_str}_IQUV-U_cube.fits'
            name_Icube = f'mosaics/mosaic_{obs_id_str}_IQUV-I_cube.fits'

    hdu_q = fits.open(name_Qcube)
    cube_q = np.array(hdu_q[0].data)
    hdu_u = fits.open(name_Ucube)
    cube_u = np.array(hdu_u[0].data)
    hdu_i = fits.open(name_Icube)
    cube_i = np.array(hdu_i[0].data)

    if cube_q.shape != cube_i.shape or cube_u.shape != cube_i.shape:
        logger.error("Error in normalize_stokes_cubes: Stokes Q, U, and I cubes have different shapes.")
        raise ValueError("Stokes Q, U, and I cubes must have the same shape.")
    else:
        logger.info('dividing Stokes Q and U cubes by Stokes I cube')
        norm_q = np.divide(cube_q, cube_i, out=np.full_like(cube_q, np.nan), where=cube_i != 0)
        norm_u = np.divide(cube_u, cube_i, out=np.full_like(cube_q, np.nan), where=cube_i != 0)

        if mode == 'single':
            fits.writeto(f'{path}/STOKES_CUBES/{obs_id}_{target}_IQUV-Q_cube_norm.fits', norm_q, header=hdu_q[0].header, overwrite=True)
            fits.writeto(f'{path}/STOKES_CUBES/{obs_id}_{target}_IQUV-U_cube_norm.fits', norm_u, header=hdu_u[0].header, overwrite=True)
            logger.info(f'wrote normalized Q and U to: {path}/STOKES_CUBES/{obs_id}_{target}_IQUV-Q_cube_norm.fits and {path}/STOKES_CUBES/{obs_id}_{target}_IQUV-U_cube_norm.fits')

        elif mode == 'mosaic':
            fits.writeto(f'mosaics/mosaic_{obs_id_str}_IQUV-Q_cube_norm.fits', norm_q, header=hdu_q[0].header, overwrite=True)
            fits.writeto(f'mosaics/mosaic_{obs_id_str}_IQUV-U_cube_norm.fits', norm_u, header=hdu_u[0].header, overwrite=True)
            logger.info(f'wrote normalized Q and U to: mosaics/mosaic_{obs_id_str}_IQUV-Q_cube_norm.fits and mosaics/mosaic_{obs_id_str}_IQUV-U_cube_norm.fits')'''

#-------------------------------------------------------------------

def rm_synth_param(obs_id, target, path, logger, obs_id_str=None, mode='single'):
    import numpy as np
    import scipy.constants 
    """
    calculate RM synthesis parameters
    return values needed for rmsynth3d and final_rm_synth
    """
    if mode == 'single':
        output_cube = f'{path}/STOKES_CUBES/{obs_id}_{target}_'
        freq_list = np.loadtxt(output_cube + 'IQUV-freq.txt')
        rms_list = np.loadtxt(output_cube + 'IQUV-rms.txt')
    elif mode == 'mosaic':
        output_cube = f'mosaics/mosaic_{obs_id_str}_'
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

def StokesI_MFS_noise(obs_id, target, logger, path, obs_id_str=None, mode='single'):
    import numpy as np
    from astropy.io import fits
    """
    calucate the noise of the Stokes I MFS image
    """
    if mode == 'single':
        im_name = f'{path}/TARGET_IMAGES/{obs_id}_{target}_pol-'
        MFS_I = im_name + 'MFS-I-image-pb.fits'
    elif mode == 'mosaic':
        im_name = f'mosaics/mosaic_{obs_id_str}_'
        MFS_I = im_name + 'MFS-I-image-pb.fits'
    hdu_im = fits.open(MFS_I)[0]

    #noise_center = [816,801]
    #noise_box = [216,143]
    noise_center = [200,200]
    noise_box = [143,143]
    #noise_center = [686, 3866]
    #noise_box = [300, 230]

    data_i=hdu_im.data.squeeze()
    data_rms_i  =data_i[noise_center[1]-noise_box[1]:noise_center[1]+noise_box[1],noise_center[0]-noise_box[0]:noise_center[0]+noise_box[0]]
    if (~np.isnan(np.nanmean(data_rms_i))):
        noise = np.sqrt(np.nanmean(data_rms_i*data_rms_i))
    else:
        logger.error('Error in StokesI_MFS_noise: RMS calculation failed for Stokes I image')
        raise ValueError("RMS calculation failed for Stokes I image")
    logger.info(f'StokesI_MFS_noise: calculated noise: {noise}')
    return noise    

#-------------------------------------------------------------------

def get_grm_map(obs_id, target, logger, path, obs_id_str=None, mode='single'):
    import glob
    from astropy.io import fits
    from reproject import reproject_interp
    import numpy as np
    '''
    determine the Galactic RM map from Hutschenreuther+20 for the region of the given image/mosaic.
    This map will be matched in size and pixel scale to the image fro pixel by pixel subtraction
    '''
    grm_map = glob.glob('/beegfs/bba5268/meerkat_virgo/FaradaySky_SH_image2.fits')
    logger.info(f'found grm map: {grm_map}')

    if mode == 'single':
        image = glob.glob(f'{path}/TARGET_IMAGES/{obs_id}_{target}_pol-MFS-I-image-pb.fits')
        outmap = f'{path}/STOKES_CUBES/FaradaySky_{obs_id}_{target}.fits'
    elif mode == 'mosaic':
        image = glob.glob(f'mosaics/mosaic_{obs_id_str}_MFS-I-image-pb.fits')
        outmap = f'mosaics/FaradaySky_{obs_id_str}.fits'

    ref = fits.open(image)[0]
    map = fits.open(grm_map)[0]

    logger.info(f'reprojecting map to header of {ref}')
    proj, footprint = reproject_interp(map, ref.header)

    hdu_out = fits.PrimaryHDU(data=proj, header=ref.header)
    hdu_out.writeto(outmap, overwrite=True)
    return outmap


#-------------------------------------------------------------------

def final_rm_synth(obs_id, target, sigma_p, d_phi, logger, path, obs_id_str=None, mode='single'):
    from astropy.io import fits
    import numpy as np
    """
    calculate the polarisation angle, polarisation fraction and RM maps
    from the rmsynth3d output
    """

    #GRM = 1.59 #Galactic RM in rad/m2, we consider it a constant value without error over the cluster
    grm_map = fits.open(get_grm_map(obs_id, target, logger, path, obs_id_str, mode))
    GRM = grm_map[0].data
    
    thresh_p = 5. #threshold in sigma for sigma_p
    thresh_i = 0. #threshold in sigma for sigma_i
    sigma_i = StokesI_MFS_noise(obs_id, target, logger, path, obs_id_str, mode)
    RMSF_FWHM = d_phi #theoretical value from RMsynth param 

    # names of output images
    if mode == 'single':
        name_out = f'{path}/STOKES_CUBES/{obs_id}_{target}-final'
    elif mode == 'mosaic':
        name_out = f'mosaics/mosaic_{obs_id_str}-final'
    name_rm_cluster = name_out+'_RM.fits' #... name of RM image corrected for the Milky Way contribution
    name_err_rm_cluster = name_out+'_err_RM.fits' # name of error RM image
    name_p = name_out+'_P.fits' #... name of polarization image
    name_pola = name_out+'_pola.fits' #... name of polarization angle image
    name_polf = name_out+'_polf.fits' #... name of polarization fraction image

    # name of input images
    if mode == 'single':
        name_tot = f'{path}/STOKES_CUBES/{obs_id}_{target}-FDF_clean_tot.fits'
        name_q = f'{path}/STOKES_CUBES/{obs_id}_{target}-FDF_clean_real.fits'
        name_u = f'{path}/STOKES_CUBES/{obs_id}_{target}-FDF_clean_im.fits'
        name_i = f'{path}/STOKES_CUBES/{obs_id}_{target}_IQUV-I_cube.fits'
    elif mode == 'mosaic':
        name_tot = f'mosaics/mosaic_{obs_id_str}-FDF_clean_tot.fits'
        name_q = f'mosaics/mosaic_{obs_id_str}-FDF_clean_real.fits'
        name_u = f'mosaics/mosaic_{obs_id_str}-FDF_clean_im.fits'
        name_i = f'mosaics/mosaic_{obs_id_str}_IQUV-I_cube.fits'

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
            if np.abs(f) >= thresh_p * sigma_p and np.abs(i) >= thresh_i * sigma_i:
                # Correct for the Ricean bias and write p
                img_p[0, yy, xx] = np.sqrt(f * f - 2.3*(sigma_p * sigma_p))
                # Cluster's RM
                img_rm_cluster[0, yy, xx] = rm - GRM[0, yy, xx]
                # Error on RM
                img_err_rm_cluster[0, yy, xx] = (RMSF_FWHM / 2) / (img_p[0, yy, xx] / sigma_p)
                # Polarization angle
                img_pola[0, yy, xx] = ((0.5 * np.arctan2(u, q)) - rm * lambda2_0) * (180.0 / np.pi)
                img_pola[0, yy, xx] = img_pola[0, yy, xx] % 360 #TEST
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

    head_rm=head_i.copy()
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

def plot_results(obs_id, target, logger, path, obs_id_str=None, mode='single', source='all'):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    import matplotlib.pyplot as plt
    """
    plot the results of RM synthesis for the target field
    """
    #input images
    if mode == 'single':
        name_p = f'{path}/STOKES_CUBES/{obs_id}_{target}-final_P.fits'
        name_polf = f'{path}/STOKES_CUBES/{obs_id}_{target}-final_polf.fits'
        name_pola = f'{path}/STOKES_CUBES/{obs_id}_{target}-final_pola.fits'
        name_rm = f'{path}/STOKES_CUBES/{obs_id}_{target}-final_RM.fits'
        name_err_rm = f'{path}/STOKES_CUBES/{obs_id}_{target}-final_err_RM.fits'
        name_stokesI = f'{path}/TARGET_IMAGES/{obs_id}_{target}_pol-0000-I-image-pb.fits'

        freq_list = np.loadtxt(f'{path}/STOKES_CUBES/{obs_id}_{target}_IQUV-freq.txt')

    elif mode == 'mosaic':
        name_p = f'mosaics/mosaic_{obs_id_str}-final_P.fits'
        name_polf = f'mosaics/mosaic_{obs_id_str}-final_polf.fits'
        name_pola = f'mosaics/mosaic_{obs_id_str}-final_pola.fits'
        name_rm = f'mosaics/mosaic_{obs_id_str}-final_RM.fits'
        name_err_rm = f'mosaics/mosaic_{obs_id_str}-final_err_RM.fits'
        name_stokesI = f'mosaics/mosaic_{obs_id_str}_MFS-I-image-pb.fits'

        freq_list = np.loadtxt(f'mosaics/mosaic_{obs_id_str}_IQUV-freq.txt')

    mean_freq = 1.284e9 #mean frequency in Hz
    hdu_I = fits.open(name_stokesI)
    header_i = hdu_I[0].header
    wcs_all = WCS(header_i)
    wcs = wcs_all.sub(['longitude', 'latitude'])

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
    
    if source == 'm49':
        logger.info('creating images for m49')
        cutout_size = (200, 200)
        center_coord = SkyCoord(188.1721479, 14.0479368, unit='deg', frame='icrs') #Center coordinates for the cutout
        xpix, ypix = center_coord.to_pixel(wcs, origin=0)  # Convert to pixel coordinates
        print(f"Center pixel: ({xpix}, {ypix}), image shape: {p.shape}")

        cutout_data = []
        #create cutout for each image
        for data, header in zip([p, pola, polf_mean, rm, err_rm], [header_p, header_pola, header_polf, header_rm, header_err_rm]):
            cutout = Cutout2D(data, position=center_coord, size=cutout_size, wcs=wcs)
            cutout_data.append(cutout.data)

    elif source == 'all':
        logger.info('no source given create image for whole field')
        cutout_data = []
        for data in [p, pola, polf_mean, rm, err_rm]:
            cutout_data.append(data)
            
    fig, axs = plt.subplots(1, 5, figsize=(5 * 5, 5), constrained_layout=True, subplot_kw={'projection': cutout.wcs})

    for ax, i, name in zip(axs, cutout_data, ['Polarised intensity', 'polarisation angle', 'polarisation fraction','Rotation measure','Rotation measure error']):

        if name == 'polarisation angle':
            i = i % 360
            min = np.nanmin(i)
            max = np.nanmax(i)
        else:
            min = np.nanmin(i)
            max = np.nanmax(i)

        im = ax.imshow(i, origin='lower', cmap='viridis', interpolation='none', vmin=min, vmax=max)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.set_title(name)
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        
    if mode == 'single':
        plt.savefig(f'{path}/PLOTS/{obs_id}_{target}_RMsynth_param_conv.png')
    elif mode == 'mosaic':
        plt.savefig(f'mosaics/mosaic_{obs_id_str}_RMsynth_param_conv.png')
#---------------------------------------------------------------------------

def run(logger, obs_ids, targets, path, mode='single'):
    from casatools import msmetadata
    import numpy as np
    """
    determine the RM synthesis parameters of single images or mosaics with RMsynth3d and deconvolve using RMClean.
    """
    logger.info("")
    logger.info("")
    logger.info("##########################################################")
    logger.info("##################### RMSYNTH TARGET #####################")
    logger.info("##########################################################")
    logger.info("")
    logger.info("")

    #log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("#################### IMAGE POLCAL ####################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("IMAGE POLCAL", "Started", warnings="", plot_link="")


    if mode == 'mosaic':
        try:
            logger.info("\n\n\n\n\n")
            logger.info("IMAGE POL TARGET: creating mosaic of images...")
            mosaic_obs_id_str = make_mosaic(logger, obs_ids, targets, path)
            logger.info('IMAGE POL TARGET: finished creating mosaic of images')
            logger.info("")
            logger.info("")
            logger.info("")
        except Exception as e:
            logger.exception("Error while creating mosaic of images")
    
    elif mode == 'single':
        mosaic_obs_id_str = None
        logger.info("IMAGE POL TARGET: running in single mode, no mosaic created")

    if mode == 'single':
        for obs_id in obs_ids:
            for target in targets:

                # create Image cubes and model of Stokes I image
                try:
                    logger.info("\n\n\n\n\n")
                    logger.info("IMAGE POL TARGET: creating image cubes for single targets...")
                    make_cubes(logger, obs_id, target, path, mosaic_obs_id_str, mode)
                    logger.info('IMAGE POL TARGET: finished creating image cubes')
                    logger.info("")
                    logger.info("")
                    logger.info("")
                    #log.append_to_google_doc('IMAGE POLCAL', 'Finished creating image cubes', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
                except Exception as e:
                    logger.exception("Error while creating image cubes")
    
                #normalize Stokes Q and U cubes by Stokes I cube
                try:
                    logger.info("\n\n\n\n\n")
                    logger.info("IMAGE POL TARGET: Creating Stokes I model from Stokes I cube...")
                    stokesI_model(obs_id, target, path, mosaic_obs_id_str, mode)
                    logger.info('IMAGE POL TARGET: finished creating Stokes I model')
                    logger.info("")
                    logger.info("")
                    logger.info("")
                except Exception as e:
                    logger.exception("Error while creating Stokes I model")
                    #log.append_to_google_doc('IMAGE POLCAL', 'Finished normalizing Stokes Q and U cubes by Stokes I cube', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    
                #calculate RM synthesis parameters
                try:
                    logger.info("\n\n\n\n\n")
                    logger.info("IMAGE POL TARGET: calculating RM synthesis parameters...")
                    d_phi, phi_max, W_far, sigma_p, sigma_RM = rm_synth_param(obs_id, target, path, logger, mosaic_obs_id_str, mode)
                    logger.info('IMAGE POL TARGET: finished calculating RM synthesis parameters')
                    logger.info("") 
                    logger.info("")
                    logger.info("")
                    #log.append_to_google_doc('IMAGE POLCAL', 'Finished calculating RMsynth paramters', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
                except Exception as e:
                    logger.exception("Error while calculating RM synthesis parameters")

                        #run rmsynth3d
                try:
                    logger.info("\n\n\n\n\n")
                    logger.info("IMAGE POL TARGET: running rmsynth3d...")
                    cube_name = f'{path}/STOKES_CUBES/{obs_id}_{target}_IQUV-'
                    rm_name = f'{obs_id}_{target}-'
                    cmd = f"export PYTHONPATH=/opt/RM-Tools:$PYTHONPATH && python3.10 /opt/RM-Tools/RMtools_3D/do_RMsynth_3D.py {cube_name}Q_cube.fits {cube_name}U_cube.fits {cube_name}freq.txt -i {cube_name}I_masked.fits -n {cube_name}rms.txt -v -l {phi_max} -s 30 -w 'variance' -o {rm_name}"
                    logger.info(f"IMAGE POL TARGET: Executing command: {cmd}")
                    stdout, stderr = utils.run_command(cmd, logger)
                    logger.info(stdout)
                    if stderr:
                        logger.error(f"Error in RMsynth: {stderr}")
                    logger.info("IMAGE POL TARGET: finished rmsynth3d")
                    logger.info("")
                    logger.info("")
                    logger.info("")
                    #log.append_to_google_doc("IMAGE POLCAL", "Finished RMsynth3d", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
                except Exception as e:
                    logger.exception("Error while running rmsynth3d")

                try:
                    logger.info("\n\n\n\n\n")
                    logger.info("IMAGE POL TARGET: running rmclean3d...")
                    fdf_name = f'{path}/STOKES_CUBES/{obs_id}_{target}-FDF_tot_dirty.fits'
                    rmsf_name = f'{path}/STOKES_CUBES/{obs_id}_{target}-RMSF_tot.fits'
                    out_name = f'{obs_id}_{target}-'
                    cmd = f"export PYTHONPATH=/opt/RM-Tools:$PYTHONPATH && python3.10 /opt/RM-Tools/RMtools_3D/do_RMclean_3D.py {fdf_name} {rmsf_name} -c {7*sigma_p} -v -o {out_name}"
                    logger.info(f"IMAGE POL TARGET: Executing command: {cmd}")
                    stdout, stderr = utils.run_command(cmd, logger)
                    logger.info(stdout)
                    if stderr:
                        logger.error(f"Error in RMsynth: {stderr}")
                    logger.info("IMAGE POL TARGET: finished rmclean3d")
                    logger.info("")
                    logger.info("")
                    logger.info("")
                    #log.append_to_google_doc("IMAGE POLCAL", "Finished RMsynth3d", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
                except Exception as e:
                    logger.exception("Error while running rmclean3d")

                try:
                    logger.info("\n\n\n\n\n")
                    logger.info("IMAGE POL TARGET: running final RM synth...")
                    final_rm_synth(obs_id, target, sigma_p, d_phi, logger, path, mosaic_obs_id_str, mode)
                    logger.info('IMAGE POL TARGET: finished final RM synth')
                    logger.info("")
                    logger.info("")
                    logger.info("")
                    #log.append_to_google_doc('IMAGE POLCAL', 'Finished final RM synth', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
                except Exception as e:
                    logger.exception("Error while running final RM synth")

                #calculate the results for the source region and plot them as a png
                try:
                    logger.info("\n\n\n\n\n")
                    logger.info("IMAGE POL TARGET: plotting results...")
                    plot_results(obs_id, target, logger, path, mosaic_obs_id_str, mode)
                    logger.info('IMAGE POL TARGET: finished plotting results')
                    logger.info("")
                    logger.info("")
                    logger.info("")
                    #log.append_to_google_doc('IMAGE POLCAL', 'Finished plotting results', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
                except Exception as e:
                    logger.exception("Error while plotting results")


    elif mode == 'mosaic':
        try:
            logger.info("\n\n\n\n\n")
            logger.info("IMAGE POL TARGET: creating image cubes for single targets...")
            make_cubes(logger, None, None, path, mosaic_obs_id_str, mode)
            logger.info('IMAGE POL TARGET: finished creating image cubes')
            logger.info("")
            logger.info("")
            logger.info("")
            #log.append_to_google_doc('IMAGE POLCAL', 'Finished creating image cubes', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
        except Exception as e:
            logger.exception("Error while creating image cubes")
    
        #normalize Stokes Q and U cubes by Stokes I cube
        try:
            logger.info("\n\n\n\n\n")
            logger.info("IMAGE POL TARGET: Creating Stokes I model from Stokes I cube...")
            stokesI_model(None, None, path, mosaic_obs_id_str, mode)
            logger.info('IMAGE POL TARGET: finished creating Stokes I model')
            logger.info("")
            logger.info("")
            logger.info("")
        except Exception as e:
            logger.exception("Error while creating Stokes I model")
            #log.append_to_google_doc('IMAGE POLCAL', 'Finished normalizing Stokes Q and U cubes by Stokes I cube', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")

    #calculate RM synthesis parameters
        try:
            logger.info("\n\n\n\n\n")
            logger.info("IMAGE POL TARGET: calculating RM synthesis parameters...")
            d_phi, phi_max, W_far, sigma_p, sigma_RM = rm_synth_param(None, None, path, logger, mosaic_obs_id_str, mode)
            logger.info('IMAGE POL TARGET: finished calculating RM synthesis parameters')
            logger.info("") 
            logger.info("")
            logger.info("")
            #log.append_to_google_doc('IMAGE POLCAL', 'Finished calculating RMsynth paramters', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
        except Exception as e:
            logger.exception("Error while calculating RM synthesis parameters")
        
        #run rmsynth3d
        try:
            logger.info("\n\n\n\n\n")
            logger.info("IMAGE POL TARGET: running rmsynth3d...")
            cube_name = f'mosaics/mosaic_{mosaic_obs_id_str}_IQUV-'
            rm_name = f'mosaic_{mosaic_obs_id_str}-'
            cmd = f"export PYTHONPATH=/opt/RM-Tools:$PYTHONPATH && python3.10 /opt/RM-Tools/RMtools_3D/do_RMsynth_3D.py {cube_name}Q_cube.fits {cube_name}U_cube.fits {cube_name}freq.txt -i {cube_name}I_masked.fits -n {cube_name}rms.txt -v -l {phi_max} -s 30 -w 'variance' -o {rm_name}"
            logger.info(f"IMAGE POL TARGET: Executing command: {cmd}")
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in RMsynth: {stderr}")
            logger.info("IMAGE POL TARGET: finished rmsynth3d")
            logger.info("")
            logger.info("")
            logger.info("")
            #log.append_to_google_doc("IMAGE POLCAL", "Finished RMsynth3d", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
        except Exception as e:
            logger.exception("Error while running rmsynth3d")

        try:
            logger.info("\n\n\n\n\n")
            logger.info("IMAGE POL TARGET: running rmclean3d...")
            fdf_name = f'mosaics/mosaic_{mosaic_obs_id_str}-FDF_tot_dirty.fits'
            rmsf_name = f'mosaics/mosaic_{mosaic_obs_id_str}-RMSF_tot.fits'
            out_name = f'mosaic_{mosaic_obs_id_str}-'
            cmd = f"export PYTHONPATH=/opt/RM-Tools:$PYTHONPATH && python3.10 /opt/RM-Tools/RMtools_3D/do_RMclean_3D.py {fdf_name} {rmsf_name} -c {7*sigma_p} -v -o {out_name}"
            logger.info(f"IMAGE POL TARGET: Executing command: {cmd}")
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in RMsynth: {stderr}")
            logger.info("IMAGE POL TARGET: finished rmclean3d")
            logger.info("")
            logger.info("")
            logger.info("")
            #log.append_to_google_doc("IMAGE POLCAL", "Finished RMsynth3d", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
        except Exception as e:
            logger.exception("Error while running rmclean3d")


        try:
            logger.info("\n\n\n\n\n")
            logger.info("IMAGE POL TARGET: running final RM synth...")
            final_rm_synth(None, None, sigma_p, d_phi, logger, path, mosaic_obs_id_str, mode)
            logger.info('IMAGE POL TARGET: finished final RM synth')
            logger.info("")
            logger.info("")
            logger.info("")
            #log.append_to_google_doc('IMAGE POLCAL', 'Finished final RM synth', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
        except Exception as e:
            logger.exception("Error while running final RM synth")

        #calculate the results for the source region and plot them as a png
        try:
            logger.info("\n\n\n\n\n")
            logger.info("IMAGE POL TARGET: plotting results...")
            plot_results(None, None, logger, path, mosaic_obs_id_str, mode)
            logger.info('IMAGE POL TARGET: finished plotting results')
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

    
    logger.info("image pol target step completed successfully!")
    logger.info("######################################################")
    logger.info("################## END IMAGE POL TARGET ##################")
    logger.info("######################################################")
    logger.info("")
    logger.info("")
    #log.append_to_google_doc("Image polcal step completed successfully!", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("###################### END IMAGE POLCAL ######################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")


