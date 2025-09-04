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

def get_ms(logger, obs_id, delete_zipped=False):
    import glob
    import os
    import tarfile
    """
    checks whether the full ms file with the given Obs_id already exists.
    If not, it will check for the zipped version on /lofar and unzip it. 
    Deletes the original zipped file if needed.
    Returns the path of the full ms file.
    """   
    search_folders = ['/lofar/bba5268/meerkat_virgo/raw_ms_files', '/lofar/bba5268/meerkat_virgo/raw_ms_files/uhf', '/lofar/p1uy068/meerkat-virgo/raw']
    zipped_folder = '/lofar/bba5268/meerkat_virgo/raw_ms_files'

    for folder in search_folders:
        ms_file = glob.glob(f'{folder}/{obs_id}*l0.ms')
        if ms_file:
            logger.info(f'Found full ms files for obs_id {obs_id}: {ms_file}')
            return ms_file[0]
        else:
            logger.info(f'No full ms files found for obs_id {obs_id} in {folder}.')
    zipped_files = glob.glob(f'{zipped_folder}/{obs_id}*.ms.tar.gz')
    if not zipped_files:
        logger.error(f'No zipped ms files found for obs_id {obs_id} in {zipped_folder}')
        raise FileNotFoundError(f'No zipped ms files found for obs_id {obs_id} in {zipped_folder}')

    zipped_file = zipped_files[0]
    logger.info(f'Found zipped ms file for obs_id {obs_id}: {zipped_file}')

    with tarfile.open(zipped_file, 'r') as tar:
        tar.extractall(path= zipped_folder)
        extracted_files = tar.getnames()
        logger.info(f'Extracted {zipped_file} to {zipped_folder}')

    ms_file = None
    for file in extracted_files:
        if file.endswith('.ms'):
            ms_file = os.path.join(zipped_folder, file)
            break

    if ms_file is None:
        logger.error(f'No .ms file found after extraction for obs_id {obs_id}. Something went wrong.')
        raise FileNotFoundError(f'No .ms file found after extraction for obs_id {obs_id}. Something went wrong.')

    basename = os.path.basename(ms_file)
    if basename.startswith(obs_id):
        logger.info(f'extracted file name already starts with {obs_id}. Will not rename')
    else: 
        new_msfile = os.path.join(zipped_folder, f'{obs_id}_{basename}')
        os.rename(ms_file, new_msfile)
        ms_file = new_msfile
        logger.info(f'Renamed extracted file to {ms_file}')
    
    if delete_zipped == True:
        try:
            os.remove(zipped_file)
            logger.info(f'Deleted zipped file {zipped_file}')
        except Exception as e:
            logger.error(f'Error deleting zipped file {zipped_file}: {e}')
    
    return ms_file

def get_cal_and_band(logger, ms_file):
    from casatools import msmetadata, table
    """
    Determine all calibrator fields and the frequency band from the ms file.
    Return calibrator list and band type (currently 'L' or 'UHF').
    """

    known_calibrators = {
        'J1939-6342': ['flux', 'bandpass'],  # PKS 1934-638 (flux/bandpass cal)
        'J1331+3030': ['polarization'],  # 3C286 (polarization cal)
        'J1150-0023': ['gain'],  # gain cal
        'J0408-6545': ['flux', 'bandpass'],  # UHF flux cal
        #add as needed
    }
    msmd = msmetadata()
    tb = table()

    try:
        msmd.open(ms_file)
        tb.open(ms_file + '/SPECTRAL_WINDOW')

        all_fields = msmd.fieldnames()
        calibrators = [field for field in all_fields if field in known_calibrators]

        cal_roles = {}
        for cal in calibrators:
            cal_roles[cal] = known_calibrators[cal]

        ref_freq = tb.getcol('REF_FREQUENCY')
        mean_freq = ref_freq.mean()/1e9 #in GHz

        if mean_freq < 1.0:
            band = 'UHF'
        else:
            band = 'L'
        logger.info(f'detected band: {band} with mean frequency {mean_freq:.2f} GHz')
        logger.info(f'found calibrators: {calibrators}')

        return calibrators, band, cal_roles
    
    finally:
        msmd.close()
        tb.close()

#----------------------------------------------------------------------------------------------------------------

def split_cal(logger, full_ms, path):
    from casatasks import mstransform
    """
    check if calibrator ms file already exists
    if not, split data of the calibrators of the full ms-file into a calibrator ms-file
    """

    ms = os.path.basename(full_ms)
    base, ext = os.path.splitext(ms)
    split_ms = f'{path}/MS_FILES/{base}-cal{ext}'
    calibrators, band, cal_roles = get_cal_and_band(logger, full_ms)
    
    if not calibrators:
        logger.error(f'No calibrators found in {full_ms}. Cannot split into calibrator ms file.')
        raise ValueError(f'No calibrators found in {full_ms}. Cannot split into calibrator ms file.')

    if os.path.isdir(split_ms):
        logger.info(f'Calibrator ms file {split_ms} already exists. Skipping splitting step.')
        return split_ms, calibrators, band, cal_roles
    
    else:
        if band == 'L':
            freq_range = '0:0.9~1.65GHz'
        elif band == 'UHF':
            freq_range = '0:0.55~1.08GHz'

        logger.info(f'Splitting the {band}-band calibrators of file {full_ms}')
        logger.info(f'Creating calibrators ms file {split_ms}')

        cals = ','.join(calibrators)

        mstransform(vis=full_ms, outputvis=split_ms, createmms=False,\
                separationaxis="auto",numsubms="auto", tileshape=[0],field=cals,spw=freq_range,scan="",antenna="",\
                correlation="",timerange="",intent="",array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
                usewtspectrum=True,combinespws=False,chanaverage=False,chanbin=1,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
                nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="", veltype="radio",preaverage=False,timeaverage=False,timebin="",\
                timespan="", maxuvwdistance=0.0,docallib=False,callib="",douvcontsub=False,fitspw="", fitorder=0,want_cont=False,denoising_lib=True,\
                nthreads=1,niter=1, disableparallel=False,ddistart=-1,taql="",monolithic_processing=False,reindex=True)
        
        cal_size = file_size(split_ms)
        space_left = free_space(f'{path}')

        logger.info(f'Size of the calibrator file:{cal_size} MB. Space left in target path {path}/MS_FILES/; {space_left[2]}/{space_left[0]} GB ({space_left[2]/space_left[0] *100} %)')
        
        if os.path.exists(f"{path}/LOGS/feedswap_cal.txt"):
            os.remove(f"{path}/LOGS/feedswap_cal.txt")
        else:
            logger.info("The feedswap file does not exist") 

        logger.info('Saved calibrator ms file successfully')
        return split_ms, calibrators, band, cal_roles

#----------------------------------------------------------------------------------------------------------------

def average_cal(logger, cal_ms, path, band, cal_roles, nchan=512, force=False):
    from casatasks import mstransform
    from casatools import msmetadata
    from utils import utils
    """
    split the calibrators into flux/gain cal file and into polarisation cal file.
    average both of these files to given channel number.
    Will automatically skip if the averaged files already exist except if force is set to True.
    """
    
    basename = os.path.basename(cal_ms)
    base, ext = os.path.splitext(basename)
    logger.info(f'Found calibrator ms file: {cal_ms}')
    if not cal_ms:
        raise FileNotFoundError(f"No MS file found for {cal_ms}")
    
    pol_cals = [cal for cal, roles in cal_roles.items() if 'polarization' in roles]
    flux_cals = [cal for cal, roles in cal_roles.items() if any(role in ['flux', 'bandpass', 'gain'] for role in roles)]
    
    pol_ms_avg = f'{path}/MS_FILES/{base}-pol{ext}'
    flux_ms_avg = f'{path}/MS_FILES/{base}-flux{ext}'

    msmd = msmetadata()
    msmd.open(cal_ms)
    nchan_cal = msmd.nchan(0)
    msmd.close()

    if band == 'L':
        freq_range = '0:0.9~1.65GHz'
    elif band == 'UHF':
        freq_range = '0:0.55~1.08GHz'

    if pol_cals:
        pol_field_str = ','.join(pol_cals)
    if flux_cals:
        flux_field_str = ','.join(flux_cals)

    if os.path.isdir(pol_ms_avg):
        msmd = msmetadata()
        msmd.open(pol_ms_avg)
        nchan_in = msmd.nchan(0)
        msmd.close()

        if force==True:
            logger.info(f'Force option set to True. Recreating polarisation calibrator ms file {pol_ms_avg}...')
            chanbin = round(nchan_cal/nchan)

            cmd = f"rm -r {pol_ms_avg} && rm -r {pol_ms_avg}.flagversions"


            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in deleting ms file: {stderr}")
            
            mstransform(vis=cal_ms, outputvis=pol_ms_avg, createmms=False,\
                separationaxis="auto",numsubms="auto", tileshape=[0],field=pol_field_str,spw=freq_range,scan="",antenna="",\
                correlation="",timerange="",intent="",array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
                usewtspectrum=True,combinespws=False,chanaverage=True,chanbin=chanbin,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
                nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="", veltype="radio",preaverage=False,timeaverage=False,timebin="",\
                timespan="", maxuvwdistance=0.0,docallib=False,callib="",douvcontsub=False,fitspw="", fitorder=0,want_cont=False,denoising_lib=True,\
                nthreads=1,niter=1, disableparallel=False,ddistart=-1,taql="",monolithic_processing=False,reindex=True)
            
            logger.info(f'Polarisation calibrator ms file {pol_ms_avg} created with {nchan} channels')

        elif nchan_in >= nchan -50 and nchan_in <= nchan +50 and force==False:
            logger.info(f'Polarisation calibrator ms file {pol_ms_avg} already exists with specified channel number. Skipping averaging step.')
        
        else:
            logger.info(f'Polarisation calibrator ms file {pol_ms_avg} already exists but with different channel number. Recreating...')
            chanbin = round(nchan_cal/nchan)

            cmd = f"rm -r {pol_ms_avg} && rm -r {pol_ms_avg}.flagversions"


            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in deleting ms file: {stderr}")
            
            mstransform(vis=cal_ms, outputvis=pol_ms_avg, createmms=False,\
                separationaxis="auto",numsubms="auto", tileshape=[0],field=pol_field_str,spw=freq_range,scan="",antenna="",\
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
            separationaxis="auto",numsubms="auto", tileshape=[0],field=pol_field_str,spw=freq_range,scan="",antenna="",\
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

        if force==True:
            logger.info(f'Force option set to True. Recreating flux/gain calibrator ms file {flux_ms_avg}...')
            chanbin = round(nchan_cal/nchan)

            cmd = f"rm -r {flux_ms_avg} && rm -r {flux_ms_avg}.flagversions"

            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in deleting ms file: {stderr}")
            
            mstransform(vis=cal_ms, outputvis=flux_ms_avg, createmms=False,\
                separationaxis="auto",numsubms="auto", tileshape=[0],field=flux_field_str,spw=freq_range,scan="",antenna="",\
                correlation="",timerange="",intent="",array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
                usewtspectrum=True,combinespws=False,chanaverage=True,chanbin=chanbin,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
                nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="", veltype="radio",preaverage=False,timeaverage=False,timebin="",\
                timespan="", maxuvwdistance=0.0,docallib=False,callib="",douvcontsub=False,fitspw="", fitorder=0,want_cont=False,denoising_lib=True,\
                nthreads=1,niter=1, disableparallel=False,ddistart=-1,taql="",monolithic_processing=False,reindex=True)
            
            logger.info(f'Flux and gain calibrator ms file {flux_ms_avg} created with {nchan} channels')

        elif nchan_in >= nchan -50 and nchan_in <= nchan +50 and force==False:
            logger.info(f'flux/gain calibrator ms file {flux_ms_avg} already exists with specified channel number. Skipping averaging step.')
        
        else:
            logger.info(f'Polarisation calibrator ms file {flux_ms_avg} already exists but with different channel number. Recreating...')
            chanbin = round(nchan_cal/nchan)

            cmd = f"rm -r {flux_ms_avg} && rm -r {flux_ms_avg}.flagversions"

            stdout, stderr = utils.run_command(cmd, logger)
            logger.info(stdout)
            if stderr:
                logger.error(f"Error in deleting ms file: {stderr}")
            
            mstransform(vis=cal_ms, outputvis=flux_ms_avg, createmms=False,\
                separationaxis="auto",numsubms="auto", tileshape=[0],field=flux_field_str,spw=freq_range,scan="",antenna="",\
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
            separationaxis="auto",numsubms="auto", tileshape=[0],field=flux_field_str,spw=freq_range,scan="",antenna="",\
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


    if full_ms:
        logger.info(f'Found full ms file: {full_ms}')
        calibrators, band, cal_roles = get_cal_and_band(logger, full_ms)
    
    msmd = msmetadata()
    msmd.open(full_ms)
    fields = msmd.fieldnames()
    msmd.close()
    
    targets = [field for field in fields if field not in calibrators]

    logger.info(f'Splitting the target fields of file {full_ms}')


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

            if nchan_in >= nchan -50 and nchan_in <= nchan +50:
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
 