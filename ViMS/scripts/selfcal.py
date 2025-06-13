#!/usr/bin/env python3

#from utils import utils, log
import glob
from utils import utils
from casatools import msmetadata
import numpy as np

################################### Helper functions ####################################

#MAYBE CHANGE SO THAT NO NEW FILE FOR EACH TARGET IS CREATED
def quartical_template(logger, obs_id, target, path, run=1, solint='300s', data_column='DATA'):
    """
    Create a quartical selfcalibration template for each target.
    Only phase and delay calibration.
    """
    split_ms = glob.glob(f"{path}/MS_FILES/{obs_id}_*{target}-avg*.ms")[0]
    
    template = f'''
      input_ms:
        path: {split_ms}
        data_column: {data_column}
        weight_column: WEIGHT_SPECTRUM
        select_fields: [0]
      input_model:
        recipe: MODEL_DATA
        beam: /localwork/angelina/meerkat_virgo/ViMS/ViMS/utils/eidos_beams/meerkat_pb_jones_cube_95channels_$(CORR)_$(REIM).fits
        beam_l_axis: X
        beam_m_axis: Y
        apply_p_jones: true
        invert_uvw: true
      output:
        gain_directory: {path}/SELFCAL_PRODUCTS
        log_directory: {path}/LOGS
        log_to_terminal: true
        overwrite: true
        products: [residual, corrected_data]
        columns: [RESIDUAL_DATA, CORRECTED_DATA]
        apply_p_jones_inv: true
      solver:
        terms: [G, D]
        iter_recipe: [15, 10]
        reference_antenna: 0
        propagate_flags: true
      G:
        type: phase
        time_interval: {solint}
        freq_interval: 32
        direction_dependent: false
      D:
        type: delay_and_offset
        time_interval: 120s
        freq_interval: 0
        direction_dependent: false
      '''
    with open(f"{path}/SELFCAL_PRODUCTS/{obs_id}_{target}_selfcal_{run}run.yaml", "w") as f:
        f.write(template)
    logger.info(f"Quartical selfcal yaml file created: {path}/SELFCAL_PRODUCTS/{obs_id}_{target}_selfcal_{run}run.yaml")
    
    return f"{path}/SELFCAL_PRODUCTS/{obs_id}_{target}_selfcal_{run}run.yaml"

#----------------------------------------------------------------------------------

def quartical_template2(logger, obs_id, target, path, solint='60s'):
    """
    Create a quartical selfcalibration template for each target.
    Complex phase and amplitude calibration.
    """
    split_ms = glob.glob(f"{path}/MS_FILES/{obs_id}_*{target}-avg*.ms")[0]
    
    template = f'''
      input_ms:
        path: {split_ms}
        data_column: CORRECTED_DATA
        weight_column: WEIGHT_SPECTRUM
        select_fields: [0]
      input_model:
        recipe: MODEL_DATA
        beam: /localwork/angelina/meerkat_virgo/ViMS/ViMS/utils/eidos_beams/meerkat_pb_jones_cube_95channels_$(CORR)_$(REIM).fits
        beam_l_axis: X
        beam_m_axis: Y
        apply_p_jones: true
        invert_uvw: true
      output:
        gain_directory: {path}/SELFCAL_PRODUCTS
        log_directory: {path}/LOGS
        log_to_terminal: true
        overwrite: true
        products: [corrected_data, corrected_weight]
        columns: [CORRECTED_DATA, WEIGHT_SPECTRUM]
        apply_p_jones_inv: true
      solver:
        terms: [C]
        iter_recipe: [15, 10]
        reference_antenna: 0
        propagate_flags: true
      C:
        type: complex
        time_interval: {solint}
        freq_interval: 1
        direction_dependent: false
      dask:
        threads: 64
      '''
    with open(f"{path}/SELFCAL_PRODUCTS/{obs_id}_{target}_selfcal_3run.yaml", "w") as f:
        f.write(template)
    logger.info(f"Quartical selfcal yaml file created: {path}/SELFCAL_PRODUCTS/{obs_id}_{target}_selfcal_3run.yaml")
    
    return f"{path}/SELFCAL_PRODUCTS/{obs_id}_{target}_selfcal_3run.yaml"

############################ Execution functions ######################################


def run(logger, obs_id, targets, path):
    logger.info("")
    logger.info("")
    logger.info("###########################################################")
    logger.info("######################### SELFCAL #########################")
    logger.info("###########################################################")
    logger.info("")
    logger.info("")

    for target in targets:
        try:
            ms_matches = glob.glob(f"{path}/MS_FILES/{obs_id}*{target}-avg.ms")
            if not ms_matches:
                logger.warning(f"No MS file found for target {target}")
                continue
            ms = ms_matches[0]
            logger.info(f"Found MS file for {target}: {ms}")
            
            logger.info(f"Running selfcal for target {target}")

            '''# Copy the CORRECTED_DATA column to DATA column
            logger.info(f"Copying the CORRECTED_DATA column in the DATA column for {target}")

            ms_splitted = ms.replace(".ms", "_splitted.ms")
            dp3_parset = "split_DATA_column_tmp.dppp"

            # Crea il parset file dinamicamente
            with open(dp3_parset, "w") as f:
                f.write(f"""\
                            msin={ms}
                            msout={ms_splitted}
                            steps=[]
                            msin.datacolumn=CORRECTED_DATA
                            msout.datacolumn=DATA
                            """)

            cmd =f"DP3 {dp3_parset}"
            stdout, stderr = utils.run_command(cmd, logger)
            
            if stderr:
                logger.warning(f"Error in copying the CORRECTED_DATA column in the DATA column for {target}:\n{stderr}")
            
            
            #logger.info(f"CORRECTED_DATA column copied in the DATA column for {target} completed successfully!")
            logger.info(f"Splitted MS file: {ms_splitted}")
            utils.run_command(f"rm {dp3_parset}", logger)
            '''

            # Run facetselfcal
            logger.info(f"Running facetselfcal for target {target}")
            cmd = f"""facetselfcal -i {path}/SELFCAL_PRODUCTS/{obs_id}_{target}_selfcal --noarchive --forwidefield \
            --soltype-list="['scalarphase', 'scalarcomplexgain']" \
            --solint-list="['1min','30min']" --nchan-list=[1,1] \
            --soltypecycles-list=[0,2] --smoothnessconstraint-list=[100.,50.] \
            --imsize=6000 --pixelsize=2. --channelsout=12 --niter=50000 \
            --paralleldeconvolution=1024 --start=0 --stop=4 --multiscale --clipsolutions \
            --multiscale-start=0 --parallelgridding=4 \
            """
            # cmd += f" --DDE --DDE-predict=WSCLEAN --targetFlux=2.0 "
            cmd += f" {ms}"

            # cmd = f"""wsclean -no-update-model-required -minuv-l 80.0 -size 6000 6000 -scale 2.0arcsec \
            #     -reorder -weight briggs -0.5 -parallel-reordering 4 -mgain 0.75 \
            #     -data-column CORRECTED_DATA -join-channels -channels-out 12 \
            #     -parallel-deconvolution 1024 -parallel-gridding 4 -auto-mask 2.5 \
            #     -auto-threshold 0.5 -multiscale \
            #     -multiscale-max-scales 11 -save-source-list \
            #     -fit-spectral-pol 3 -pol i -baseline-averaging 0.8181230868723419 \
            #     -gridder wgridder -wgridder-accuracy 0.0001 -no-min-grid-resolution \
            #     -name /beegfs/bbf4346/OUTPUT/obs26/wsclean/{obs_id}_{target}_wsclean \
            #     -nmiter 12 -niter 50000 \
            #     {ms}"""

            stdout, stderr = utils.run_command(cmd, logger)
            
            if stderr:
                logger.warning(f"Selfcal completed for {target} but with errors:\n{stderr}")

        except IndexError:
            logger.error(f"No MS file found in glob for target {target}")
        except Exception as e:
            logger.exception(f"Unexpected error while processing target {target}: {e}")


    logger.info("Selfcal step completed successfully!")
    logger.info("")
    logger.info("")
    logger.info("#######################################################")
    logger.info("##################### END SELFCAL #####################")
    logger.info("#######################################################")
    logger.info("")
    logger.info("")


def run_selfcal(logger, obs_id, targets, path):
    logger.info("")
    logger.info("")
    logger.info("###########################################################")
    logger.info("######################### SELFCAL #########################")
    logger.info("###########################################################")
    logger.info("")
    logger.info("")

    for target in targets:
        split_ms = glob.glob(f"{path}/MS_FILES/{obs_id}_*{target}-avg*.ms")[0]
        logger.info(f"found {target} ms file: {split_ms}")
        logger.info(f'Running selfcal for target {target}')

        try:
            logger.info('\n\n\n\n\n')
            logger.info('SELFCAL: running WSClean (initial)....')
            im_name = f'{path}/TARGET_IMAGES/image_0/{obs_id}_{target}_0'
            
            
            msmd = msmetadata()
            msmd.open(split_ms)
            chan_width = np.mean(msmd.chanwidths(0))*1e-3 # convert to MHz
            msmd.close()
            cutoff = int((1380 - 900)/chan_width)

            cmd = (f"wsclean -name {im_name} -size 2760 2760 -scale 1.9asec -mgain 0.8 -niter 1000000 -auto-threshold 2 -auto-mask 6 "
                "-pol iquv -weight briggs 0.0 -j 32 -abs-mem 100.0 -channels-out 5 -join-channels -gridder wgridder "
                "-squared-channel-joining -join-polarizations -fit-spectral-pol 4 -multiscale  -multiscale-scales 0,2,3,6 -multiscale-scale-bias 0.75 "
                f"-parallel-deconvolution 1000 -parallel-gridding 1 -channel-range 0 {cutoff} -nwlayers-factor 3 -minuvw-m 40 -no-mf-weighting -weighting-rank-filter 3 "
                f"-data-column DATA {split_ms}")
            
            logger.info(cmd)
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in WSClean: {stderr}")

        except Exception as e:
            logger.error(f'Error while running WSClean: {e}')

        try:
            logger.info('\n\n\n\n\n')
            logger.info('SELFCAL: running masking....')
            mfs_im = f'{path}/TARGET_IMAGES/image_0/{obs_id}_{target}_0-MFS-I-image.fits'

            cmd = (f'breizorro --restored-image {mfs_im} --threshold 6.5 --outfile {path}/TARGET_IMAGES/image_0/{obs_id}_{target}_0-mask.fits')
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in Breizorro: {stderr}")
      
        except Exception as e:
            logger.error(f'Error while running Breizorro: {e}')

        try:
            logger.info('\n\n\n\n\n')
            logger.info('SELFCAL: running WSClean (1st round)....')
            im_name = f'{path}/TARGET_IMAGES/image_1/{obs_id}_{target}_1'
            mask_name = f'{path}/TARGET_IMAGES/image_0/{obs_id}_{target}_0-mask.fits'

            cmd = (f"wsclean -name {im_name} -size 2760 2760 -scale 1.9asec -mgain 0.8 -niter 1000000 -auto-threshold 2 -fits-mask {mask_name} "
                "-pol iquv -weight briggs 0.0 -j 32 -abs-mem 100.0 -channels-out 5 -join-channels -gridder wgridder "
                "-squared-channel-joining -join-polarizations -fit-spectral-pol 4 -multiscale  -multiscale-scales 0,2,3,5 -multiscale-scale-bias 0.75 "
                f"-parallel-deconvolution 1000 -parallel-gridding 1 -channel-range 0 {cutoff} -nwlayers-factor 3 -minuvw-m 40 -no-mf-weighting -weighting-rank-filter 3 "
                f"-data-column DATA {split_ms}")
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in WSClean: {stderr}")
      
        except Exception as e:
            logger.error(f'Error while running WSClean: {e}')

        try:
            logger.info('SELFCAL: starting Quartical (1st round)...')
            template = quartical_template(logger, obs_id, target, path)
            logger.info('SELFCAL: Quartical template created successfully!')
            cmd = f'goquartical {template}'
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in Quartical: {stderr}")
            logger.info('SELFCAL: Quartical run completed successfully!')
            
        except Exception as e:
            logger.error(f"Error in Quartical: {e}")

        try:
            logger.info('\n\n\n\n\n')
            logger.info('SELFCAL: running masking (2nd round)....')
            mfs_im = f'{path}/TARGET_IMAGES/image_1/{obs_id}_{target}_1-MFS-I-image.fits'

            cmd = (f'breizorro --restored-image {mfs_im} --threshold 6.5 --outfile {path}/TARGET_IMAGES/image_1/{obs_id}_{target}_1-mask.fits')
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in Breizorro: {stderr}")
      
        except Exception as e:
            logger.error(f'Error while running Breizorro: {e}')

        try:
            logger.info('\n\n\n\n\n')
            logger.info('SELFCAL: running WSClean (2nd round)....')
            im_name = f'{path}/TARGET_IMAGES/image_2/{obs_id}_{target}_2'
            mask_name = f'{path}/TARGET_IMAGES/image_1/{obs_id}_{target}_1-mask.fits'

            cmd = (f"wsclean -name {im_name} -size 2760 2760 -scale 1.9asec -mgain 0.8 -niter 1000000 -auto-threshold 1 -fits-mask {mask_name} "
                "-pol iquv -weight briggs 0.0 -j 32 -abs-mem 100.0 -channels-out 5 -join-channels -gridder wgridder "
                "-squared-channel-joining -join-polarizations -fit-spectral-pol 4 -multiscale  -multiscale-scales 0,2,3,5 -multiscale-scale-bias 0.75 "
                f"-parallel-deconvolution 1000 -parallel-gridding 1 -channel-range 0 {cutoff} -nwlayers-factor 3 -minuvw-m 40 -no-mf-weighting -weighting-rank-filter 3 "
                f" -data-column CORRECTED_DATA {split_ms}")
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in WSClean: {stderr}")

        except Exception as e:
            logger.error(f'Error while running WSClean: {e}')
        
        try:
            logger.info('SELFCAL: starting Quartical (2nd round)...')
            template = quartical_template(logger, obs_id, target, path, run=2, solint='120s', data_column='CORRECTED_DATA')
            logger.info('SELFCAL: Quartical template created successfully!')
            cmd = (f'goquartical {template}')
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in Quartical: {stderr}")
            logger.info('SELFCAL: Quartical run completed successfully!')
            
        except Exception as e:
            logger.error(f"Error in Quartical: {e}")

        try:
            logger.info('\n\n\n\n\n')
            logger.info('SELFCAL: running masking (3rd round)....')
            mfs_im = f'{path}/TARGET_IMAGES/image_2/{obs_id}_{target}_2-MFS-I-image.fits'

            cmd = (f'breizorro --restored-image {mfs_im} --threshold 4 --outfile {path}/TARGET_IMAGES/image_2/{obs_id}_{target}_2-mask.fits')
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in Breizorro: {stderr}")
      
        except Exception as e:
            logger.error(f'Error while running Breizorro: {e}')

        try:
            logger.info('\n\n\n\n\n')
            logger.info('SELFCAL: running WSClean (3rd round)....')
            im_name = f'{path}/TARGET_IMAGES/image_3/{obs_id}_{target}_3'
            mask_name = f'{path}/TARGET_IMAGES/image_2/{obs_id}_{target}_2-mask.fits'

            cmd = (f"wsclean -name {im_name} -size 2760 2760 -scale 1.9asec -mgain 0.8 -niter 1000000 -auto-threshold 1 -fits-mask {mask_name} "
                "-pol iquv -weight briggs 0.0 -j 32 -abs-mem 100.0 -channels-out 5 -join-channels -gridder wgridder "
                "-squared-channel-joining -join-polarizations -fit-spectral-pol 4 -multiscale  -multiscale-scales 0,2,3,5 -multiscale-scale-bias 0.75 "
                f"-parallel-deconvolution 1000 -parallel-gridding 1 -channel-range 0 {cutoff} -nwlayers-factor 3 -minuvw-m 40 -no-mf-weighting -weighting-rank-filter 3 "
                f"-data-column CORRECTED_DATA {split_ms}")
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in WSClean: {stderr}")

        except Exception as e:
            logger.error(f'Error while running WSClean: {e}')


        try:
            logger.info('SELFCAL: starting Quartical (3rd round)...')
            template = quartical_template2(logger, obs_id, target, path)
            logger.info('SELFCAL: Quartical template created successfully!')
            cmd = (f'goquartical {template}')
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in Quartical: {stderr}")
            logger.info('SELFCAL: Quartical run completed successfully!')
            
        except Exception as e:
            logger.error(f"Error in Quartical: {e}")

        try:
            logger.info('\n\n\n\n\n')
            logger.info('SELFCAL: running masking (final round)....')
            mfs_im = f'{path}/TARGET_IMAGES/image_3/{obs_id}_{target}_3-MFS-I-image.fits'

            cmd = (f'breizorro --restored-image {mfs_im} --threshold 3 --outfile {path}/TARGET_IMAGES/image_3/{obs_id}_{target}_3-mask.fits')
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in Breizorro: {stderr}")
      
        except Exception as e:
            logger.error(f'Error while running Breizorro: {e}')

        try:
            logger.info('\n\n\n\n\n')
            logger.info('SELFCAL: running WSClean (final round)....')
            im_name = f'{path}/TARGET_IMAGES/image_4/{obs_id}_{target}_4'
            mask_name = f'{path}/TARGET_IMAGES/image_3/{obs_id}_{target}_3-mask.fits'

            cmd = (f"wsclean -name {im_name} -size 2760 2760 -scale 1.9asec -mgain 0.8 -niter 1000000 -auto-threshold 1 -fits-mask {mask_name} "
                "-pol iquv -weight briggs 0.0 -j 32 -abs-mem 100.0 -channels-out 5 -join-channels -gridder wgridder "
                "-squared-channel-joining -join-polarizations -fit-spectral-pol 4 -multiscale  -multiscale-scales 0,2,3,5 -multiscale-scale-bias 0.75 "
                f"-parallel-deconvolution 1000 -parallel-gridding 1 -channel-range 0 {cutoff} -nwlayers-factor 3 -minuvw-m 40 -no-mf-weighting -weighting-rank-filter 3 "
                f"-data-column CORRECTED_DATA {split_ms}")
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in WSClean: {stderr}")

        except Exception as e:
            logger.error(f'Error while running WSClean: {e}')
        




    logger.info("Selfcal step completed successfully!")
    logger.info("")
    logger.info("")
    logger.info("#######################################################")
    logger.info("##################### END SELFCAL #####################")
    logger.info("#######################################################")
    logger.info("")
    logger.info("")