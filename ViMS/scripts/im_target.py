def run(logger, obs_id, targets, path):
    import glob
    from casatools import msmetadata
    from utils import utils
    import numpy as np
    from astropy.io import fits
    """
    Image the target fields via WSClean and determine the RM synthesis parameters
    of it with RMsynth3d for the given Observation ID.
    """
    logger.info("")
    logger.info("")
    logger.info("##########################################################")
    logger.info("###################### IMAGE POL TARGET ######################")
    logger.info("##########################################################")
    logger.info("")
    logger.info("")

    #log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("#################### IMAGE POLCAL ####################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    #log.append_to_google_doc("IMAGE POLCAL", "Started", warnings="", plot_link="")

    for target in targets:

        try:
            logger.info("\n\n\n\n\n")
            logger.info("IMAGE POL TARGET: starting WSClean...")

            im_name = f'{path}/TARGET_IMAGES/{obs_id}_{target}_pol'
            target_ms = glob.glob(f'{path}/../{obs_id}_*{target}-avg.ms.copy')[0]
            target_mask = glob.glob(f'{path}/SELFCAL_PRODUCTS/{obs_id}_{target}_selfcal_003-MFS*.mask.fits')[0]

            mask_hdu = fits.open(target_mask)
            mask_data = mask_hdu[0].data
            mask_shape = mask_data.shape
            image_size = 6000

            if mask_shape != (image_size, image_size):
                logger.info(f"Resizing mask from {mask_shape} to {image_size}")
                center_x = mask_shape[2] // 2
                center_y = mask_shape[3] // 2
                cmd = f"python3.10 /angelina/meerkat_virgo/scripts_fra/fitscutout.py --filename {target_mask} -s {image_size} {image_size} -p {center_x} {center_y}"
                stdout, stderr = utils.run_command(cmd, logger)
                logger.info(stdout)
                if stderr:
                    logger.error(f"Error in fitscutout.py: {stderr}")
                target_mask = glob.glob(f'{path}/SELFCAL_PRODUCTS/{obs_id}_{target}_selfcal_003-MFS*.mask-cut.fits')[0]
                mask_hdu = fits.open(target_mask)
                mask_data = mask_hdu[0].data
                logger.info(f'shape of new mask: {mask_data.shape}')
            
            else:
                pass

            #cutoff = int((1380 - 900)/chan_width) # cutoff at 1380 MHz to avoid off-axis leakage

            # toAdd: set image size, parallel-deconvolution 256?, 
            cmd = f"wsclean -name {im_name} -size {image_size} {image_size} -scale 2.asec -mgain 0.75 -niter 50000 -fits-mask {target_mask} \
                -field 0 -pol iquv -weight briggs -0.5 -j 16 -abs-mem 100.0 -channels-out 100 -join-channels -gridder wgridder -no-update-model-required \
                -reorder -parallel-reordering 4 -squared-channel-joining -join-polarizations -fit-spectral-pol 4 -circular-beam -beam-size 13asec -apply-primary-beam \
                -auto-mask 5 -auto-threshold 1 -multiscale -multiscale-scale-bias 0.75 -multiscale-scales 0,1,2,6 -wgridder-accuracy 0.001 -no-min-grid-resolution -primary-beam-limit 0.5\
                -parallel-deconvolution 512 -parallel-gridding 4 -nwlayers-factor 3 -minuvw-m 40 -no-mf-weighting -weighting-rank-filter 3 \
                -taper-gaussian 6 -data-column CORRECTED_DATA {target_ms}"

            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in WSClean: {stderr}")
        
            logger.info('IMAGE POL TARGET: finished WSClean')
            logger.info("")
            logger.info("")
            logger.info("")
            #log.append_to_google_doc('IMAGE POLCAL', 'Finished WSClean', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
        except Exception as e:
            logger.exception("Error while running WSClean")
