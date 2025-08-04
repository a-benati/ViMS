import os, shutil
import glob


#Helper scripts

def file_size(path):
    """Returns the size of the file in MB."""
    return os.path.getsize(path)*1/(1024*1024)

def free_space(path):
    """Returns the free disk space in MB for the given path."""
    total, used, free = shutil.disk_usage(path)
    gb = (1024*1024*1024)
    return (total/gb, used/gb, free/gb)

def cal_lib(logger, obs_id, path):
    """
    create a library containing all the crosscal calibration tables.
    Used for OTF calibration in mstransform (does not contain any gaintables)
    """
    logger.info('Collect calibration tables applied to target fields')
    tables = [
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.kcal.sec" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.gcal_p.sec-selfcal" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.gcal_a2" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.bandpass2" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.T.sec" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.df2" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.kcrosscal" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.xf.ambcorr" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="" spwmap=0',
    ]

    lib = open(f'{path}/CAL_TABLES/{obs_id}_calib_tables.txt','w')
    for table in tables:
        lib.write(f"{table}\n")
    lib.close()
    logger.info(f'Successfully wrote all tables to a file: {path}/CAL_TABLES/{obs_id}_calib_tables.txt')
    return f'{path}/CAL_TABLES/{obs_id}_calib_tables.txt'



###################### preparation of the ms files ##########################

def copy_ms(logger, obs_id, path):
    """
    copy the full ms file from the observatory server to /beegfs
    and unzip it. Delete the original zipped file
    """   
    #TOADD: check if file already in beegfs also check if file alterantively in /lofar2 or lofar1. then no need to copy or unzip. return the path of the full_ms file for further handling
    return('Not implemented yet')

#----------------------------------------------------------------------------------------------------------------

def split_cal(logger, obs_id, full_ms, path):
    from casatasks import mstransform
    """
    check if calibrator ms file already exists
    if not, split data of the calibrators of the full ms-file into a calibrator ms-file
    """

    #full_ms = glob.glob(f'/beegfs/bba5268/meerkat_virgo/{obs_id}_*l0.ms')[0]
    #full_ms = copy_ms(logger, obs_id, path)
    ms = os.path.basename(full_ms)
    base, ext = os.path.splitext(ms)
    split_ms = f'{path}/MS_FILES/{base}-cal{ext}'

    if os.path.isdir(split_ms):
        logger.info(f'Calibrator ms file {split_ms} already exists. Skipping splitting step.')
        return split_ms
    
    else:

        logger.info(f'Splitting the calibrators of file {full_ms}')
        logger.info(f'Creating calibrators ms file {split_ms}')

        mstransform(vis=full_ms, outputvis=split_ms, createmms=False,\
                separationaxis="auto",numsubms="auto", tileshape=[0],field="J1939-6342,J1150-0023,J1331+3030",spw="0:0.9~1.65GHz",scan="",antenna="",\
                correlation="",timerange="",intent="",array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
                usewtspectrum=True,combinespws=False,chanaverage=False,chanbin=1,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
                nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="", veltype="radio",preaverage=False,timeaverage=False,timebin="",\
                timespan="", maxuvwdistance=0.0,docallib=False,callib="",douvcontsub=False,fitspw="", fitorder=0,want_cont=False,denoising_lib=True,\
                nthreads=1,niter=1, disableparallel=False,ddistart=-1,taql="",monolithic_processing=False,reindex=True)
        
        cal_size = file_size(split_ms)
        space_left = free_space(f'{path}')

        logger.info(f'Size of the calibrator file:{cal_size} MB. Space left in target path {path}/MS_FILES/; {space_left[2]}/{space_left[0]} GB ({space_left[2]/space_left[0] *100} %)')
        
        if os.path.exists(f"{path}/LOGS/feedswap.txt"):
            os.remove("feedswap.txt")
        else:
            logger.info("The feedswap file does not exist") 
    
        logger.info('Saved calibrator ms file succecsfully')
        return split_ms

#----------------------------------------------------------------------------------------------------------------

def average_cal(logger, cal_ms, path, nchan=512):
    """
    split the calibrators into flux/gain cal file and into polarisation cal file.
    average both of these files to given channel number.
    """
    from casatasks import mstransform
    from casatools import msmetadata
    from utils import utils
    
    basename = os.path.basename(cal_ms)
    base, ext = os.path.splitext(basename)
    logger.info(f'Found calibrator ms file: {cal_ms}')
    if not cal_ms:
        raise FileNotFoundError(f"No MS file found for {cal_ms}")
    pol_ms_avg = f'{path}/MS_FILES/{base}-pol{ext}'
    flux_ms_avg = f'{path}/MS_FILES/{base}-flux{ext}'

    msmd = msmetadata()
    msmd.open(cal_ms)
    nchan_cal = msmd.nchan(0)
    msmd.close()

    if os.path.isdir(pol_ms_avg):
        msmd = msmetadata()
        msmd.open(pol_ms_avg)
        nchan_in = msmd.nchan(0)
        msmd.close()

        if nchan_in >= nchan -100 and nchan_in <= nchan +100:
            logger.info(f'Polarisation calibrator ms file {pol_ms_avg} already exists with specified channel number. Skipping averaging step.')
        
        else:
            logger.info(f'Polarisation calibrator ms file {pol_ms_avg} already exists but with different channel number. Recreating...')
            chanbin = round(nchan_cal/nchan)

            cmd = f"rm -r {pol_ms_avg} && rm -r {pol_ms_avg}.flagversions"


            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in deletin ms file: {stderr}")
            
            mstransform(vis=cal_ms, outputvis=pol_ms_avg, createmms=False,\
                separationaxis="auto",numsubms="auto", tileshape=[0],field="J1331+3030",spw="0:0.9~1.65GHz",scan="",antenna="",\
                correlation="",timerange="",intent="",array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
                usewtspectrum=True,combinespws=False,chanaverage=True,chanbin=chanbin,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
                nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="", veltype="radio",preaverage=False,timeaverage=False,timebin="",\
                timespan="", maxuvwdistance=0.0,docallib=False,callib="",douvcontsub=False,fitspw="", fitorder=0,want_cont=False,denoising_lib=True,\
                nthreads=1,niter=1, disableparallel=False,ddistart=-1,taql="",monolithic_processing=False,reindex=True)
            
            logger.info(f'Polarisation calibrator ms file {pol_ms_avg} created with {nchan} channels')

    else:
        logger.info(f'Creating polarisation calibrator ms file {pol_ms_avg}')
        chanbin = round(nchan_cal/nchan)
        
        mstransform(vis=cal_ms, outputvis=pol_ms_avg, createmms=False,\
            separationaxis="auto",numsubms="auto", tileshape=[0],field="J1331+3030",spw="0:0.9~1.65GHz",scan="",antenna="",\
            correlation="",timerange="",intent="",array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
            usewtspectrum=True,combinespws=False,chanaverage=True,chanbin=chanbin,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
            nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="", veltype="radio",preaverage=False,timeaverage=False,timebin="",\
            timespan="", maxuvwdistance=0.0,docallib=False,callib="",douvcontsub=False,fitspw="", fitorder=0,want_cont=False,denoising_lib=True,\
            nthreads=1,niter=1, disableparallel=False,ddistart=-1,taql="",monolithic_processing=False,reindex=True)
        
        logger.info(f'Polarisation calibrator ms file {pol_ms_avg} created with {nchan} channels')

    if os.path.isdir(flux_ms_avg):
        msmd = msmetadata()
        msmd.open(flux_ms_avg)
        nchan_in = msmd.nchan(0)
        msmd.close()

        if nchan_in >= nchan -100 and nchan_in <= nchan +100:
            logger.info(f'flux/gain calibrator ms file {flux_ms_avg} already exists with specified channel number. Skipping averaging step.')
        
        else:
            logger.info(f'Polarisation calibrator ms file {flux_ms_avg} already exists but with different channel number. Recreating...')
            chanbin = round(nchan_cal/nchan)

            cmd = f"rm -r {flux_ms_avg} && rm -r {flux_ms_avg}.flagversions"

            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in deletin ms file: {stderr}")
            
            mstransform(vis=cal_ms, outputvis=flux_ms_avg, createmms=False,\
                separationaxis="auto",numsubms="auto", tileshape=[0],field="J1939-6342,J1150-0023",spw="0:0.9~1.65GHz",scan="",antenna="",\
                correlation="",timerange="",intent="",array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
                usewtspectrum=True,combinespws=False,chanaverage=True,chanbin=chanbin,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
                nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="", veltype="radio",preaverage=False,timeaverage=False,timebin="",\
                timespan="", maxuvwdistance=0.0,docallib=False,callib="",douvcontsub=False,fitspw="", fitorder=0,want_cont=False,denoising_lib=True,\
                nthreads=1,niter=1, disableparallel=False,ddistart=-1,taql="",monolithic_processing=False,reindex=True)
            
            logger.info(f'Flux and gain calibrator ms file {flux_ms_avg} created with {nchan} channels')

    else:
        logger.info(f'Creating flux/gain calibrator ms file {flux_ms_avg}')
        chanbin = round(nchan_cal/nchan)
        
        mstransform(vis=cal_ms, outputvis=flux_ms_avg, createmms=False,\
            separationaxis="auto",numsubms="auto", tileshape=[0],field="J1939-6342,J1150-0023",spw="0:0.9~1.65GHz",scan="",antenna="",\
            correlation="",timerange="",intent="",array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
            usewtspectrum=True,combinespws=False,chanaverage=True,chanbin=chanbin,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
            nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="", veltype="radio",preaverage=False,timeaverage=False,timebin="",\
            timespan="", maxuvwdistance=0.0,docallib=False,callib="",douvcontsub=False,fitspw="", fitorder=0,want_cont=False,denoising_lib=True,\
            nthreads=1,niter=1, disableparallel=False,ddistart=-1,taql="",monolithic_processing=False,reindex=True)
        
        logger.info(f'Polarisation calibrator ms file {flux_ms_avg} created with {nchan} channels')


    logger.info('Saved all calibrator ms files successfully')
    return pol_ms_avg, flux_ms_avg

#----------------------------------------------------------------------------------------------------------------

def split_targets(logger, obs_id, full_ms, path):
    from casatasks import mstransform
    from casatools import msmetadata
    """
    split data of the fields of the full ms-file into individual field ms-files. 
    does NOT apply any calibration tables or averaging
    """

    #path_ms = f'/beegfs/bba5268/meerkat_virgo/'
    #full_ms = glob.glob(f'{path_ms}{obs_id}_*l0.ms')[0]
    #full_ms = copy_ms(logger, obs_id, path)
    if full_ms:
        logger.info(f'Found full ms file: {full_ms}')
    
    msmd = msmetadata()
    msmd.open(full_ms)
    fields = msmd.fieldnames()
    msmd.close()
    targets = []

    logger.info(f'Splitting the target fields of file {full_ms}')

    for field in fields:
        if field == 'J1939-6342':
            pass
        elif field == 'J1331+3030':
            pass
        elif field == 'J1150-0023':
            pass
        else:
            targets.append(field)

    filename = os.path.basename(full_ms)
    base, ext = os.path.splitext(filename)

    for target in targets:
        split_ms = f'{path}/MS_FILES/{base}-{target}{ext}'

        if os.path.isdir(split_ms):
            logger.info(f'Target ms file {split_ms} already exists. Skipping splitting step.')
            continue
        else:
            logger.info(f'Creating target ms file {split_ms}')

            mstransform(vis=full_ms,outputvis=split_ms,createmms=False,\
                separationaxis="auto",numsubms="auto",tileshape=[0],field=target,spw="",scan="",antenna="", correlation="",timerange="",intent="",\
                array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
                usewtspectrum=True,combinespws=False,chanaverage=False,chanbin=1,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
                nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="",veltype="radio",preaverage=False,timeaverage=False,timebin="",\
                timespan="",maxuvwdistance=0.0,docallib=False,\
                douvcontsub=False,fitspw="",fitorder=0,want_cont=False,denoising_lib=True,nthreads=1,niter=1,disableparallel=False,ddistart=-1,taql="",\
                monolithic_processing=False,reindex=True)
        
            target_size = file_size(split_ms)
            space_left = free_space(f'{path}')
            logger.info(f'Size of the target file:{target_size} MB. Space left in target path {path}; {space_left[2]}/{space_left[0]} GB. ({space_left[2]/space_left[0]*100} %)')
    
    logger.info('Saved all target ms files successfully!')
    return targets

#----------------------------------------------------------------------------------------------------------------

def average_targets(logger, obs_id, targets, path, nchan=512, chanbin=None, force=False):
    from casatasks import mstransform, applycal
    from casatools import msmetadata
    import glob
    from utils import utils
    """
    average each target ms file to the specified number of channels.
    If the target ms file already exists, it will check if the number of channels is correct.
    """


    for target in targets:
        split_ms = glob.glob(f'{path}/MS_FILES/*{target}.ms')[0]
        if split_ms:
            logger.info(f'Found target ms file: {split_ms}')
        else:
            raise FileNotFoundError(f"No MS file found for {path}/MS_FILES/*{target}.ms")
        
        base, ext = os.path.splitext(split_ms)
        split_ms_avg = f"{base}-avg{ext}"

        msmd = msmetadata()
        msmd.open(split_ms)
        nchan_tar = msmd.nchan(0)
        logger.info(f'Number of channels of {target}: {nchan_tar}')
        msmd.close()

        if force == True and os.path.isdir(split_ms_avg):
            logger.info(f'Forcing recreation of target ms file {split_ms_avg}')

            cmd = f"rm -r {split_ms_avg}"
            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in deleting ms file: {stderr}")
                
            if chanbin is None:
                if nchan_tar < 3000 and nchan_tar != 2048:
                    chanbin=int(4)
                    dp3chan = 1860
                elif nchan_tar >= 3000 and nchan_tar != 4096:
                    chanbin = int(8)
                    dp3chan = 3720
                else:
                    chanbin = int(nchan_tar/nchan)
                    dp3chan = nchan_tar

            else:
                chanbin = chanbin
                dp3chan = nchan_tar
            
            cmd = f"rm -r {split_ms_avg}"

            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in deleting ms file: {stderr}")

            # mstransform(vis=split_ms,outputvis=split_ms_avg,createmms=False,\
            #         separationaxis="auto",numsubms="auto",tileshape=[0],field=target,spw="",scan="",antenna="", correlation="",timerange="",intent="",\
            #         array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
            #         usewtspectrum=True,combinespws=False,chanaverage=True,chanbin=chanbin,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
            #         nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="",veltype="radio",preaverage=False,timeaverage=True,timebin="16s",\
            #         timespan="",maxuvwdistance=0.0,docallib=False, callib='',\
            #         douvcontsub=False,fitspw="",fitorder=0,want_cont=False,denoising_lib=True,nthreads=1,niter=1,disableparallel=False,ddistart=-1,taql="",\
            #         monolithic_processing=False,reindex=True)
            
            dp3_parset = "avg_DP3_tmp.dppp"

            # Crea il parset file dinamicamente
            with open(dp3_parset, "w") as f:
                f.write(f"""\
                            msin={split_ms}
                            msin.datacolumn=CORRECTED_DATA
                            msin.nchan={dp3chan}
                            msout={split_ms_avg}
                            msout.storagemanager=dysco
                            steps=[average]
                            average.timeresolution=16
                            average.freqstep={chanbin}
                            """)

            cmd =f"DP3 {dp3_parset}"
            stdout, stderr = utils.run_command(cmd, logger)
            
            if stderr:
                logger.warning(f"Error in averaging the ms file for {target}:\n{stderr}")

            utils.run_command(f"rm {dp3_parset}", logger)
        
            cal_size = file_size(split_ms_avg)
            space_left = free_space(f'{path}')
            logger.info(f'Size of the target file:{cal_size} MB. Space left in target path {path}; {space_left[2]}/{space_left[0]} GB. ({space_left[2]/space_left[0]*100} %)')

        
        elif os.path.isdir(split_ms_avg) and force == False:
            msmd = msmetadata()
            msmd.open(split_ms_avg)
            nchan_in = msmd.nchan(0)
            msmd.close()

            if nchan_in >= nchan -100 and nchan_in <= nchan +100:
                logger.info(f'Target ms file {split_ms_avg} averaged to {nchan} channels already exists. Skipping averaging step.')
                continue
            else:
                logger.info(f'Target ms file {split_ms_avg} already exists but with different channel number. Recreating...')
            
                if chanbin is None:
                    if nchan_tar < 3000 and nchan_tar != 2048:
                        chanbin=int(4)
                        dp3chan = 1860
                    elif nchan_tar >= 3000 and nchan_tar != 4096:
                        chanbin = int(8)
                        dp3chan = 3720
                    else:
                        chanbin = int(nchan_tar/nchan)
                        dp3chan = nchan_tar
                else:
                    chanbin = chanbin
                    dp3chan = nchan_tar

                cmd = f"rm -r {split_ms_avg}"

                stdout, stderr = utils.run_command(cmd, logger)
                logger.info(stdout)
                if stderr:
                    logger.error(f"Error in deleting ms file: {stderr}")

                # mstransform(vis=split_ms,outputvis=split_ms_avg,createmms=False,\
                #     separationaxis="auto",numsubms="auto",tileshape=[0],field=target,spw="",scan="",antenna="", correlation="",timerange="",intent="",\
                #     array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
                #     usewtspectrum=True,combinespws=False,chanaverage=True,chanbin=chanbin,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
                #     nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="",veltype="radio",preaverage=False,timeaverage=True,timebin="16s",\
                #     timespan="",maxuvwdistance=0.0,docallib=False, callib='',\
                #     douvcontsub=False,fitspw="",fitorder=0,want_cont=False,denoising_lib=True,nthreads=1,niter=1,disableparallel=False,ddistart=-1,taql="",\
                #     monolithic_processing=False,reindex=True)
                
                dp3_parset = "avg_DP3_tmp.dppp"

                # Crea il parset file dinamicamente
                with open(dp3_parset, "w") as f:
                    f.write(f"""\
                                msin={split_ms}
                                msin.datacolumn=CORRECTED_DATA
                                msin.nchan={dp3chan}
                                msout={split_ms_avg}
                                msout.storagemanager=dysco
                                steps=[average]
                                average.timeresolution=16
                                average.freqstep={chanbin}
                                """)

                cmd =f"DP3 {dp3_parset}"
                stdout, stderr = utils.run_command(cmd, logger)
                
                if stderr:
                    logger.warning(f"Error in averaging the ms file for {target}:\n{stderr}")

                utils.run_command(f"rm {dp3_parset}", logger)
        
                cal_size = file_size(split_ms_avg)
                space_left = free_space(f'{path}')
                logger.info(f'Size of the target file:{cal_size} MB. Space left in target path {path}; {space_left[2]}/{space_left[0]} GB. ({space_left[2]/space_left[0]*100} %)')
        else:
            logger.info(f'Creating target ms file {split_ms_avg}')
            if chanbin is None:
                if nchan_tar < 3000 and nchan_tar != 2048:
                    chanbin=int(4)
                    dp3chan = 1860
                elif nchan_tar >= 3000 and nchan_tar != 4096:
                    chanbin = int(8)
                    dp3chan = 3720
                else:
                    chanbin = int(nchan_tar/nchan)
                    dp3chan = nchan_tar
            else:
                chanbin = chanbin
                dp3chan = nchan_tar

            # mstransform(vis=split_ms,outputvis=split_ms_avg,createmms=False,\
            #     separationaxis="auto",numsubms="auto",tileshape=[0],field=target,spw="",scan="",antenna="", correlation="",timerange="",intent="",\
            #     array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
            #     usewtspectrum=True,combinespws=False,chanaverage=True,chanbin=chanbin,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
            #     nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="",veltype="radio",preaverage=False,timeaverage=True,timebin="16s",\
            #     timespan="",maxuvwdistance=0.0,docallib=False, callib='',\
            #     douvcontsub=False,fitspw="",fitorder=0,want_cont=False,denoising_lib=True,nthreads=1,niter=1,disableparallel=False,ddistart=-1,taql="",\
            #     monolithic_processing=False,reindex=True)
            
            dp3_parset = "avg_DP3_tmp.dppp"

            # Crea il parset file dinamicamente
            with open(dp3_parset, "w") as f:
                f.write(f"""\
                            msin={split_ms}
                            msin.datacolumn=CORRECTED_DATA
                            msin.nchan={dp3chan}
                            msout={split_ms_avg}
                            msout.storagemanager=dysco
                            steps=[average]
                            average.timeresolution=16
                            average.freqstep={chanbin}
                            """)

            cmd =f"DP3 {dp3_parset}"
            stdout, stderr = utils.run_command(cmd, logger)
            
            if stderr:
                logger.warning(f"Error in averaging the ms file for {target}:\n{stderr}")

            utils.run_command(f"rm {dp3_parset}", logger)
        
            cal_size = file_size(split_ms_avg)
            space_left = free_space(f'{path}')
            logger.info(f'Size of the target file:{cal_size} MB. Space left in target path {path}; {space_left[2]}/{space_left[0]} GB. ({space_left[2]/space_left[0]*100} %)')
    
    logger.info(f'Averaged target ms file {target} successfully!')


#----------------------------------------------------------------------------------------------------------------

def apply_cal(logger, obs_id, targets, path):
    from casatasks import applycal
    """
    Apply the calibration tables from cross and polcal to the target fields
    """

    for target in targets:
        target = glob.glob(f"{path}/MS_FILES/{obs_id}*{target}.ms")[0]
        if not os.path.exists(target):
            logger.error(f"Target ms file {target} does not exist.")

        else:
            logger.info(f'Applying calibration tables to target {target}')
            applycal(vis=target, gaintable=[f'{path}/CAL_TABLES/{obs_id}_calib.kcal.sec', f'{path}/CAL_TABLES/{obs_id}_calib.gcal_p.sec-selfcal',\
                     f'{path}/CAL_TABLES/{obs_id}_calib.gcal_a2', f'{path}/CAL_TABLES/{obs_id}_calib.bandpass2',f'{path}/CAL_TABLES/{obs_id}_calib.T.sec',\
                    f'{path}/CAL_TABLES/{obs_id}_calib.df2', f'{path}/CAL_TABLES/{obs_id}_calib.kcrosscal', f'{path}/CAL_TABLES/{obs_id}_calib.xf.ambcorr'], parang=True, flagbackup=False)

        
        logger.info(f'Successfully applied calibration tables to target {target}')

#----------------------------------------------------------------------------------------------------------------

def ionosphere_corr_target(logger, obs_id, targets, path):
    from pathlib import Path
    from spinifex import h5parm_tools
    from spinifex.vis_tools import ms_tools
    from utils import utils
    """
    Calculate the ionospheric RM for the target fields and apply it as a calibration table to the data.
    """
    for target in targets:
        logger.info(f'Calculating ionospheric RM for target {target}')
        target_avg = glob.glob(f"{path}/MS_FILES/{obs_id}*{target}-avg.ms")[0]
        
        ms_path = Path(target_avg)
        ms_metadata = ms_tools.get_metadata_from_ms(ms_path)

        rms = ms_tools.get_rm_from_ms(ms_path, use_stations=ms_metadata.station_names)
        h5parm_name = f"{path}/CAL_TABLES/{obs_id}_{target}.h5parm"
        if os.path.exists(h5parm_name):
            os.remove(h5parm_name)
            logger.info(f"ionosphere_rm: Removed existing h5parm file {h5parm_name}")
        h5parm_tools.write_rm_to_h5parm(rms=rms, h5parm_name=h5parm_name)
        logger.info(f"ionosphere_rm: Created h5parm file {h5parm_name} with ionospheric RM.")

        cmd = f"DP3 msin={target_avg} msout=. msin.datacolumn=DATA msout.datacolumn=DATA steps=[cor] cor.type=correct cor.parmdb={h5parm_name} cor.correction=rotationmeasure000 cor.invert=True"
        stdout, stderr = utils.run_command(cmd, logger)
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in DP3: {stderr}")
 