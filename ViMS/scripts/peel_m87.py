def image_m87(logger, obs_id, target, path):
    import glob
    from utils import utils
    '''
    Image M87 using wsclean to create model for peeling.
    '''
    im_name = f'{path}/M87_IMAGES/{obs_id}_{target}_m87'
    target_ms = glob.glob(f'{path}/../{obs_id}_*{target}-avg.ms.copy')[0]

    cmd = f"wsclean -name {im_name} -size 700 700 -scale 2.asec -mgain 0.75 -niter 50000 -shift 12:30:48.1276 +12d22m54.211s \
                -field 0 -pol iquv -weight briggs -0.5 -j 32 -abs-mem 100.0 -channels-out 12 -join-channels -gridder wgridder \
                -reorder -parallel-reordering 4 -squared-channel-joining -join-polarizations -fit-spectral-pol 4 \
                -auto-mask 2.5 -auto-threshold 0.5 -multiscale -multiscale-scale-bias 0.75 -multiscale-scales 0,1,2,6 -wgridder-accuracy 0.001 -no-min-grid-resolution \
                -parallel-deconvolution 1000 -parallel-gridding 4 -nwlayers-factor 3 -no-mf-weighting \
                -taper-gaussian 6 -data-column CORRECTED_DATA {target_ms}"

    stdout, stderr = utils.run_command(cmd, logger)
    logger.info(stdout)
    if stderr:
        logger.error(f"Error in WSClean: {stderr}")

def flux_measurement(logger, obs_id, target, path, region):
    import numpy as np
    from lib_fits import AllImages
    import glob
    ''' 
    Measure the flux of M87 in the image.
    '''
    all_im_m87 = glob.glob(f"{path}/M87_IMAGES/{obs_id}_{target}_m87*MFS-I-image.fits")
    all_im_m87 = AllImages(all_im_m87)
    im_m87 = all_im_m87.images[0]

    data = im_m87.img_data
    wcs = im_m87.get_wcs()

    beam_area_pix = im_m87.get_beam_area(unit='pixel')
    pix_region = region.to_pixel(wcs=wcs)
    mask = pix_region.to_mask()
    mask_weight = mask.to_image(data.shape)

    return np.nansum(data*mask_weight)/beam_area_pix, beam_area_pix

def run(logger, obs_id, targets, path):
    from utils import utils
    import glob
    from regions import Regions
    '''
    Peel M87 from the target field if the flux is higher than a specified value
    '''

    logger.info("")
    logger.info("")
    logger.info("##########################################################")
    logger.info("###################### PEEL M87 #########################")
    logger.info("##########################################################")
    logger.info("")
    logger.info("")

    for target in targets:

        try:
            logger.info("\n\n\n\n\n")
            logger.info("PEEL M87: starting WSClean...")
            image_m87(logger, obs_id, target, path)

            logger.info('PEEL M87: finished WSClean')
            logger.info("")
            logger.info("")
            logger.info("")

        except Exception as e:
            logger.exception("Error while running WSClean for M87")

        try:
            logger.info("\n\n\n\n\n")
            logger.info("PEEL M87: measuring flux...")

            ms_file = glob.glob(f'{path}/../{obs_id}_*{target}-avg.ms.copy')[0]
            region = f"/beegfs/bba5268/meerkat_virgo/m87_offaxis.reg"
            reg = Regions.read(region, format='ds9')[0]
            flux, beam_area_pix = flux_measurement(logger, obs_id, target, path, reg)
            logger.info(f"Flux of M87: {flux:.2f} Jy, Beam area in pixels: {beam_area_pix:.2f} pix")

            if flux > 0.01:  # threshold for peeling
                logger.info(f"Peeling M87 with flux {flux:.2f} Jy")
                cmd = f'taql "UPDATE {ms_file} SET CORRECTED_DATA = CORRECTED_DATA - MODEL_DATA"'
            
                stdout, stderr = utils.run_command(cmd, logger)
                logger.info(stdout)
                if stderr:
                    logger.error(f"Error in WSClean: {stderr}")
                logger.info('PEEL M87: finished peeling')
            else:
                logger.info(f"Flux of M87 is too low ({flux:.2f} Jy), skipping peeling.")

            logger.info('PEEL M87: finished peeling')
            logger.info("")
            logger.info("")
        except Exception as e:
            logger.exception("Error while measuring flux for M87")