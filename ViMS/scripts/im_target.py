

def run(logger, obs_id, targets, path):
    import glob
    from casatools import msmetadata
    from utils import utils
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    import os
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

            cmd = f"wsclean -name {im_name} -size {image_size} {image_size} -scale 2.asec -mgain 0.75 -niter 50000 -fits-mask {target_mask} \
                -field 0 -pol iquv -weight briggs -0.5 -j 16 -abs-mem 100.0 -channels-out 100 -join-channels -gridder wgridder -no-update-model-required \
                -reorder -parallel-reordering 4 -squared-channel-joining -join-polarizations -fit-spectral-pol 4 -circular-beam -beam-size 13asec \
                -auto-mask 3 -auto-threshold 1.5 -multiscale -multiscale-scale-bias 0.75 -multiscale-scales 0,1,2,4\
                -parallel-deconvolution 512 -parallel-gridding 4 -nwlayers-factor 3 -minuvw-m 40 -no-mf-weighting -weighting-rank-filter 3 \
                -taper-gaussian 10 -data-column CORRECTED_DATA {target_ms}"

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

        try:
            logger.info("\n\n\n\n\n")
            logger.info('IMAGE POL TARGET: creating beam...')

            im_name = f'{path}/TARGET_IMAGES/{obs_id}_{target}_pol'
            cmd = f"python3.10 /angelina/meerkat_virgo/scripts_fra/MeerKAT_beam.py --savebeam {im_name}"
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in MeerKAT_beam.py: {stderr}")

            logger.info('IMAGE POL TARGET: finished creating beam')
            logger.info("")
            logger.info("")
            logger.info("")
            #log.append_to_google_doc('IMAGE POLCAL', 'Finished WSClean', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
        except Exception as e:
            logger.exception("Error while running MeerKAT_beam.py")

        try:
            logger.info("\n\n\n\n\n")
            logger.info('IMAGE POL TARGET: clipping image...')

            im_name = f'TARGET_IMAGES/{obs_id}_{target}_pol'
            beam_name = f'TARGET_IMAGES/{obs_id}_{target}_pol'
            beamcut = 0.5 

            cmd = f"python3.10 /angelina/meerkat_virgo/RM_scripts/beamcut.py --path {path} --image_basename {im_name} --crop --beams_base {beam_name} --beamcut {beamcut} --output {path}/TARGET_IMAGES"
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in beamcut.py: {stderr}")

            logger.info('IMAGE POL TARGET: finished clipping image')
            logger.info("")
            logger.info("")
            logger.info("")
            #log.append_to_google_doc('IMAGE POLCAL', 'Finished clipping image', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")

        except Exception as e:
            logger.exception(f"Error while running beamcut.py for target {target}")
            continue

        try:
            logger.info("\n\n\n\n\n")
            logger.info('IMAGE POL TARGET: matching image shapes...')

            im_name = f'{path}/TARGET_IMAGES/{obs_id}_{target}_pol'
            image_list = glob.glob(f'{im_name}*image-cut.fits')

            max_shape = [0, 0]
            valid_images = []
            for image in image_list:
                with fits.open(image) as hdu:
                    data = hdu[0].data
                    if np.all(np.isnan(data)):
                        logger.info(f"Skipping {image}: Fully NaN")
                        continue
                    valid_images.append(image)
                    max_shape[0] = max(max_shape[0], data.shape[0])
                    max_shape[1] = max(max_shape[1], data.shape[1])

            if not valid_images:
                logger.warning("No valid images found (all are fully NaN). Skipping padding.")
                continue
            
            logger.info(f"Maximum shape determined: {max_shape}")
                 
            for image in valid_images:
                #pad images to max shape
                with fits.open(image, mode='update') as hdu:
                    data = hdu[0].data
                    header = hdu[0].header

                    y_offset = (max_shape[0] - data.shape[0]) // 2
                    x_offset = (max_shape[1] - data.shape[1]) // 2

                    padded = np.full(max_shape, np.nan)
                    padded[y_offset:y_offset + data.shape[0], x_offset:x_offset + data.shape[1]] = data

                    if 'CRPIX1' in header and 'CRPIX2' in header:
                        header['CRPIX1'] += x_offset
                        header['CRPIX2'] += y_offset
                    header['NAXIS1'] = max_shape[1]
                    header['NAXIS2'] = max_shape[0]

                    hdu[0].data = padded
                    hdu.flush()
                    logger.info(f"Padded {image} to shape {max_shape}")

                #cut beam files to same size as images
                beam_file = image.replace('-image-cut.fits', '-image_beam.fits')
                if not os.path.isfile(beam_file):
                    logger.warning(f"Beam file {beam_file} not found.")
                    continue
                with fits.open(beam_file) as hdu:
                    beam_data = hdu[0].data
                    beam_hdr = hdu[0].header
                    beam_wcs = WCS(beam_hdr)
                    wcs_new = beam_wcs.sub(['longitude', 'latitude'])
                    if beam_data.shape == tuple(max_shape):
                        logger.info(f"Beam {beam_file} already has correct shape {max_shape}.")
                        continue

                    beam_center = (beam_data.shape[1] // 2, beam_data.shape[0] // 2)
                    beam_cut = Cutout2D(beam_data, position=beam_center, size=max_shape, wcs=wcs_new)
                    new_hdr = beam_hdr.copy()
                    new_hdr.update(beam_cut.wcs.to_header())
                    new_hdr['NAXIS1'] = beam_cut.data.shape[1]
                    new_hdr['NAXIS2'] = beam_cut.data.shape[0]
                    output = beam_file.replace('beam.fits', 'beam-cut.fits')
                    fits.writeto(output, beam_cut.data, new_hdr, overwrite=True)
                    logger.info(f"Cut beam {beam_file} to shape {max_shape} and saved to {output}")


            logger.info('IMAGE POL TARGET: finished matching image shapes')
            logger.info("")
            logger.info("")
            logger.info("")
            #log.append_to_google_doc('IMAGE POLCAL', 'Finished creating polarization image', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")

        except Exception as e:
            logger.exception("Error while matching image size")