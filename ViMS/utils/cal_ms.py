import os, shutil
import glob


def file_size(path):
    """Returns the size of the file in MB."""
    return os.path.getsize(path)*1/(1024*1024)

def free_space(path):
    """Returns the free disk space in MB for the given path."""
    total, used, free = shutil.disk_usage(path).free
    mb = (1024*1024)
    return (total/mb, used/mb, free/mb)

def cal_lib(obs_id, logger, target, path):
    """
    create a library containing all the crosscal calibration tables.
    Used for OTF calibration in mstransform (does not contain any gaintables)
    """
    logger.info('Collect calibration tables applied to to target fields')
    tables = [
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.Kcal" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="{target}" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.kcrosscal" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="{target}" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.xf.ambcorr" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="{target}" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.bandpass2" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="{target}" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.T.pol" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="{target}" spwmap=0',
        f'caltable="{path}/CAL_TABLES/{obs_id}_calib.df2" calwt=False tinterp="nearest" finterp="linear" fldmap="nearest" field="{target}" spwmap=0',
    ]

    lib = open(f'{path}/CAL_TABLES/{obs_id}_calib_tables.txt','w')
    for table in tables:
        lib.write(f"{table}\n")
    lib.close()
    logger.info(f'Successfully wrote all tables to a file: {path}/CAL_TABLES/{obs_id}_calib_tables.txt')


def split_cal(logger, obs_id, path):
    from casatasks import mstransform
    """
    split data of the calibrators of the full ms-file into a calibrator ms-file
    """

    ms = glob.glob(f'{obs_id}_*l0.ms')
    full_ms = f'/lofar2/p1uy068/meerkat-virgo/raw/{ms}'
    base, ext = os.path.splitext(ms)
    split_ms = f'{path}/MS_FILES/{base}-cal{ext}'

    logger.info(f'Splitting the calibrators of file {full_ms}')
    logger.info(f'Creating calibrators ms file {split_ms}')

    cal_size = file_size(split_ms)
    space_left = free_space(f'{path}')

    logger.info(f'Size of the calibrator file:{cal_size} MB. Space left in target path {path}/MS_FILES/; {space_left[2]}/{space_left[0]} MB')

    mstransform(vis=full_ms, outputvis=split_ms, createmms=False,\
                separationaxis="auto",numsubms="auto", tileshape=[0],field="J1939-6342,J1150-0023,J1331+3030",spw="0:0.9~1.65GHz",scan="",antenna="",\
                correlation="",timerange="",intent="",array="",uvrange="",observation="",feed="",datacolumn="data",realmodelcol=False,keepflags=True,\
                usewtspectrum=True,combinespws=False,chanaverage=False,chanbin=1,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
                nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="", veltype="radio",preaverage=False,timeaverage=False,timebin="",\
                timespan="", maxuvwdistance=0.0,docallib=False,callib="",douvcontsub=False,fitspw="", fitorder=0,want_cont=False,denoising_lib=True,\
                nthreads=1,niter=1, disableparallel=False,ddistart=-1,taql="",monolithic_processing=False,reindex=True)
    
    logger.info('Saved calibrator ms file succecsfully')
    return split_ms

    
def split_targets(obs_id, logger, path):
    from casatasks import mstransform
    from casatools import msmetadata
    """
    split data of the fields of the full ms-file into individual field ms-files
    """
    ms = glob.glob(f'{obs_id}_*l0.ms')
    full_ms = f'/lofar4/bba5268/meerkat_virgo/{ms}'
    
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

    base, ext = os.path.splitext(ms)

    for target in targets:
        split_ms = f'{path}/MS_FILES/{base}-{target}{ext}'
        logger.info(f'Creating target ms file {split_ms}')

        cal_size = file_size(split_ms)
        space_left = free_space(f'{path}')
        logger.info(f'Size of the target file:{cal_size} MB. Space left in target path {path}; {space_left[2]}/{space_left[0]} MB')

        mstransform(vis=full_ms,outputvis=split_ms,createmms=False,\
                separationaxis="auto",numsubms="auto",tileshape=[0],field=target,spw="",scan="",antenna="", correlation="",timerange="",intent="",\
                array="",uvrange="",observation="",feed="",datacolumn="corrected",realmodelcol=False,keepflags=True,\
                usewtspectrum=True,combinespws=False,chanaverage=False,chanbin=1,hanning=False, regridms=False,mode="channel",nchan=-1,start=0,width=1,\
                nspw=1,interpolation="linear",phasecenter="",restfreq="",outframe="",veltype="radio",preaverage=False,timeaverage=False,timebin="",\
                timespan="",maxuvwdistance=0.0,docallib=True, callib=cal_lib(obs_id, logger, target),\
                douvcontsub=False,fitspw="",fitorder=0,want_cont=False,denoising_lib=True,nthreads=1,niter=1,disableparallel=False,ddistart=-1,taql="",\
                monolithic_processing=False,reindex=True)
    
    logger.info('Saved all target ms files succecsfully')


