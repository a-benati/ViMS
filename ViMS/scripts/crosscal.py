#!/usr/bin/env python3

import numpy as np
from casatasks import *
from casatools import table
from utils import utils


#---------------------------------------------------------------------
def add_column(table, col_name, like_col="DATA", like_type=None):
    """
    Lifted from ratt-ru/cubical
    Inserts a new column into the measurement set.
    Args:
        col_name (str):
            Name of target column.
        like_col (str, optional):
            Column will be patterned on the named column.
        like_type (str or None, optional):
            If set, column type will be changed.
    Returns:
        bool:
            True if a new column was inserted, else False.
    """

    if col_name not in table.colnames():
        # new column needs to be inserted -- get column description from column 'like_col'
        desc = table.getcoldesc(like_col)

        desc[str('name')] = str(col_name)
        desc[str('comment')] = str(desc['comment'].replace(" ", "_"))  # got this from Cyril, not sure why
        dminfo = table.getdminfo(like_col)
        dminfo[str("NAME")] =  "{}-{}".format(dminfo["NAME"], col_name)

        # if a different type is specified, insert that
        if like_type:
            desc[str('valueType')] = like_type
        table.addcols(desc, dminfo)
        return True
    return False

#----------------------------------------------------------------------
def xyamb(logger, xytab, xyout=''):
    import time
    from casatools import table
    import numpy as np
    """
    Resolve the 180-degree cross-hand phase ambiguity in a CASA calibration table.
    CAlculates the mean phase and shifts every point deviating more then 90 degrees from the mean phase by 180 degrees.

    Parameters:
    xytab : str
        Path to the input calibration table.
    xyout : str, optional
        Path to the output calibration table. If not specified, the input table is modified in place.
    """
    tb=table()

    if xyout == '':
        xyout = xytab
    if xyout != xytab:
        tb.open(xytab)
        tb.copy(xyout)
        tb.clearlocks()
        tb.close()

    tb.open(xyout, nomodify=False)

    spw_ids = np.unique(tb.getcol('SPECTRAL_WINDOW_ID'))

    for spw in spw_ids:
        st = tb.query('SPECTRAL_WINDOW_ID=='+str(spw))
        if st.nrows() > 0:
            c = st.getcol('CPARAM')
            fl = st.getcol('FLAG')

            num_channels = c.shape[1]
            flipped_channels=0
            avg_phase = np.angle(np.mean(c[0, :, :][~fl[0,:,:]]), True)
            logger.info('xyamb: Average phase = '+str(avg_phase))
            for ch in range(num_channels):
                valid_data = c[0,ch,:][~fl[0,ch,:]]
                if valid_data.size > 0:
                    xyph0 = np.angle(np.mean(valid_data), True)

                    # Calculate the phase difference
                    phase_diff =  np.abs(((xyph0 - avg_phase)))

                    if phase_diff >= 100.0:
                        flipped_channels += 1
                        c[0, ch, :] *= -1.0
                        st.putcol('CPARAM', c)
            
            logger.info('xyamb: Flipped '+str(flipped_channels)+' channels in SPW '+str(spw))


            st.close()
            time.sleep(1)
    
    tb.clearlocks()
    tb.flush()
    tb.close()
    time.sleep(1)

#----------------------------------------------------------------------------------
def ionosphere_rm(logger, pol_ms, obs_id, path):
    from pathlib import Path
    from spinifex import h5parm_tools
    from spinifex.vis_tools import ms_tools
    """
    Calculate the ionospheric RM for the polarisation calibrator and create a h5param file.
    """

    ms_path = Path(pol_ms)
    ms_metadata = ms_tools.get_metadata_from_ms(ms_path)
    ionex_dir = f'{path}/IONEX_DATA'

    rms = ms_tools.get_rms_from_ms(ms_path, use_stations=ms_metadata.station_names, prefix='cod', output_directory=ionex_dir)
    h5parm_name = f"{path}/CAL_TABLES/{obs_id}_polcal.h5parm"
    h5parm_tools.write_rm_to_h5parm(rms=rms, h5parm_name=h5parm_name)
    logger.info(f"ionosphere_rm: Created h5parm file {h5parm_name} with ionospheric RM.")
    return h5parm_name

#----------------------------------------------------------------------------------
def crosscal(logger, obs_id, cal_ms, pol_ms, path, ref_ant='m000'):
    import numpy as np
    import os,sys
    from scripts import flag
    from casatasks import gaincal, setjy, bandpass, clearcal, polcal, flagmanager, applycal, flagdata, tclean
    from casatools import table
    """
    do crosscalibration and polarisation calibration for the given ms file.
    Adjusted from strategy of Benjamin Hugo and Annalisa Bonafede
    """

    calms = cal_ms
    obs = obs_id

    # Name your gain tables
    ktab = path+'/CAL_TABLES/'+obs+'_calib.kcal'
    gtab_p = path+'/CAL_TABLES/'+obs+'_calib.gcal_p'
    gtab_a = path+'/CAL_TABLES/'+obs+'_calib.gcal_a'
    btab = path+'/CAL_TABLES/'+obs+'_calib.bandpass'
    ktab2 = path+'/CAL_TABLES/'+obs+'_calib.kcal2'
    gtab_p2 = path+'/CAL_TABLES/'+obs+'_calib.gcal_p2'
    gtab_a2 = path+'/CAL_TABLES/'+obs+'_calib.gcal_a2'
    btab2 = path+'/CAL_TABLES/'+obs+'_calib.bandpass2'
    ktab_sec = path+'/CAL_TABLES/'+obs+'_calib.kcal.sec'
    gtab_sec_p = path+'/CAL_TABLES/'+obs+'_calib.gcal_p.sec'
    Ttab_sec = path+'/CAL_TABLES/'+obs+'_calib.T.sec'
    ktab_pol = path+'/CAL_TABLES/'+obs+'_calib.kcal.pol'
    gtab_pol_p = path+'/CAL_TABLES/'+obs+'_calib.gcal_p.pol'
    Ttab_pol = path+'/CAL_TABLES/'+obs+'_calib.T.pol'

    # Name your polcal tables
    ptab_df = path+'/CAL_TABLES/'+obs+'_calib.df'
    ptab_df2 = path+'/CAL_TABLES/'+obs+'_calib.df2'
    kxtab = path+'/CAL_TABLES/'+obs+'_calib.kcrosscal'
    ptab_xf = path+'/CAL_TABLES/'+obs+'_calib.xf'
    ptab_xfcorr = path+'/CAL_TABLES/'+obs+'_calib.xf.ambcorr'


    ###################################################################
    #get names of the fields from table

    tb = table()
    tb.open(calms+ "/FIELD")
    field_names = tb.getcol("NAME")
    tb.close()

    fcal = field_names[1]
    bpcal = field_names[1]
    gcal = field_names[0]

    #get name of the polcal
    tb.open(pol_ms+ "/FIELD")
    field_names_pol = tb.getcol("NAME")
    tb.close()
    xcal = field_names_pol[0]


    ################################################################
    # Change RECEPTOR_ANGLE FOR FLUXCAL: DEFAULT IS -90DEG 

    tb.open(calms+'/FEED', nomodify=False)
    feed_angle = tb.getcol('RECEPTOR_ANGLE')
    new_feed_angle = np.zeros(feed_angle.shape)
    tb.putcol('RECEPTOR_ANGLE', new_feed_angle)
    tb.close()
#-----------------------------------------------------

    # Change RECEPTOR_ANGLE FOR POLCAL: DEFAULT IS -90DEG 

    tb.open(pol_ms+'/FEED', nomodify=False)
    feed_angle = tb.getcol('RECEPTOR_ANGLE')
    new_feed_angle = np.zeros(feed_angle.shape)
    tb.putcol('RECEPTOR_ANGLE', new_feed_angle)
    tb.close()

    
    #get flagging state after initial flagging and clear calibrations

    flag_versions = flagmanager(vis=calms, mode='list')

    if isinstance(flag_versions, dict):
        initial_flag = any(entry['name'] == obs+'_flag_after' for entry in flag_versions.values())
    else:
        initial_flag = False

    if initial_flag:
        flagmanager(vis=calms, mode='restore', versionname=obs+'_flag_after', merge='replace')
        logger.info("crosscal: Found '"+obs+"_flag_after'. Restoring it.")

    else:
        flagmanager(vis=calms, mode='save', versionname=obs+'_flag_after', merge='replace')
        logger.info("crosscal: No 'flag_after' found. Save current flagging state.")

    logger.info('')
    logger.info('Clearing calibrations')
    clearcal(vis=calms)
    logger.info('')

#-------------------------------------------------------------------    
    
    #get flagging state after initial flagging and clear calibrations for the polarisation calibrator

    flag_versions = flagmanager(vis=pol_ms, mode='list')

    if isinstance(flag_versions, dict):
        initial_flag = any(entry['name'] == obs+'_flag_after' for entry in flag_versions.values())
    else:
        initial_flag = False

    if initial_flag:
        flagmanager(vis=calms, mode='restore', versionname=obs+'_flag_after', merge='replace')
        logger.info("crosscal: Found '"+obs+"_flag_after' for the polcal. Restoring it.")

    else:
        flagmanager(vis=calms, mode='save', versionname=obs+'_flag_after', merge='replace')
        logger.info("crosscal: No 'flag_after' found for the polcal. Save current flagging state.")

    logger.info('')
    logger.info('Clearing calibrations of the polcal')
    clearcal(vis=pol_ms)
    logger.info('')

    ###################################################################
    # Set model for flux and polarisation calibrator


    for cal in set(fcal.split(',')+bpcal.split(',')):

        if cal == 'J1939-6342':
            logger.info('crosscal: setting model for flux calibrator J1939-6342')
            setjy(vis = calms, field = "{0}".format(fcal), spw = "", selectdata = False, timerange = "", scan = "", \
                standard = 'Stevens-Reynolds 2016', scalebychan = True, useephemdir = False, usescratch = True)

        else:
            logger.error("corsscal: Unknown calibrator, insert model in the script please ", cal)
            sys.exit()

#-------------------------------------------------------------

    for cal in set(xcal.split(',')):
        
        if cal =='J1331+3030':
            logger.info('crosscal: setting model for polarisation calibrator J1331+3030')

            reffreq = '1.284GHz'
            # Stokes flux density in Jy
            I =  15.74331687
            Q = 0.8628247336
            U = 1.248991241
            V = 0   
            # Spectral Index in 2nd order
            alpha =   [-0.4881438633844231, -0.17025582235426978]

            setjy(vis=pol_ms,
                field=xcal,
                selectdata=False,
                scalebychan=True,
                standard="manual",
                listmodels=False,
                fluxdensity=[I,Q,U,V],
                spix=alpha,
                reffreq=reffreq,
                polindex=[0.0964244115642966, 0.018277345381024372, -0.07332409550519345, 0.3253188415787851, -0.37228554528542385],
                polangle=[0.48312994184873537, 0.12063061722082152, -0.04180094935296229, 0.12832951565823608],
                rotmeas=0.12,
                useephemdir=False,
                interpolation="nearest",
                usescratch=True,
                ismms=False,
                )
        else:
            logger.error("crosscal: Unknown polarisation calibrator, insert model in the script please ", cal)
            sys.exit()

    ####################################################################
    # Ionospehric corruption of the polcal model
    #calculate ionospheric roatation measure with spinifix and corrupt the model with dp3

    h5parm = ionosphere_rm(logger, pol_ms, obs_id, path)

    cmd = f"DP3 msin={pol_ms} msout=. msin.datacolumn=MODEL_DATA msout.datacolumn=MODEL_DATA steps=[cor] cor.type=correct cor.parmdb={h5parm} corr.correction=rotationmeasure000 cor.invert=False"
    stdout, stderr = utils.run_command(cmd)
    logger.info(stdout)
    if stderr:
        logger.error(f"Error in DP3: {stderr}")

    ###################################################################
    # crosscalibration of the primary calibrator

    logger.info("")
    logger.info('crosscal: Starting crosscalibration of the primary following the calibration scheme KGBFKGB')
    logger.info('crosscal: using the following reference antenna: ', ref_ant)

    # Delay calibration
    gaincal(vis = calms, caltable = ktab, selectdata = True,\
        solint = "inf", field = bpcal, combine = "scan",uvrange='',\
        refant = ref_ant, solnorm = False, gaintype = "K",\
        minsnr=3,parang = False)

    os.system(f'/opt/ragavi-env/bin/ragavi-gains --table {ktab} --field 1 --htmlname {path}/PLOTS/{obs}_Kcal --plotname {path}/PLOTS/{obs}_Kcal.png')

    # phase cal on bandpass calibrator
    gaincal(vis = calms, caltable = gtab_p, selectdata = True,\
        solint = "60s", field = bpcal, combine = "",\
        refant = ref_ant, gaintype = "G", calmode = "p",uvrange='',\
        gaintable = [ktab], gainfield = [''], interp = [''],parang = False)

    # amplitude cal on bandpass calibrator
    gaincal(vis = calms, caltable = gtab_a, selectdata = True,\
        solint = "inf", field = bpcal, combine = "",\
        refant = ref_ant, gaintype = "T", calmode = "a",uvrange='', refantmode='strict',\
        gaintable = [ktab,gtab_p], gainfield = ['',''], interp = ['',''],parang = False)

    os.system(f'/opt/ragavi-env/bin/ragavi-gains --table {gtab_a} --field 1 -o {path}/PLOTS/{obs}_Gcal_amp -p {path}/PLOTS/{obs}_Gcal_amp.png')
    os.system(f'/opt/ragavi-env/bin/ragavi-gains --table {gtab_p} --field 1 -o {path}/PLOTS/{obs}_Gcal_phase -p {path}/PLOTS/{obs}_Gcal_phase.png')

    # bandpass cal on bandpass calibrator
    bandpass(vis = calms, caltable = btab, selectdata = True,\
            solint = "inf", field = bpcal, combine = "scan", uvrange='',\
            refant = ref_ant, solnorm = False, bandtype = "B",\
        gaintable = [ktab,gtab_p,gtab_a], gainfield = ['','',''],\
        interp = ['','',''], parang = False)

    os.system(f'/opt/ragavi-env/bin/ragavi-gains --table {btab} --field 1 -o {path}/PLOTS/{obs}_Bpcal -p {path}/PLOTS/{obs}_Bpcal.png')
 
    applycal(vis=calms, field=bpcal, gaintable=[ktab,gtab_p,gtab_a,btab], applymode='calflag', flagbackup=False)

    os.system(f'/opt/shadems-env/bin/shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_crosscal_XXYY.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x FREQ -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_crosscal_XYYX.png')

    #add flagging of Data and redoing crosscal on primary again
    flagdata(vis=calms, mode="rflag", field=bpcal, datacolumn="corrected", quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False)
    flagdata(vis=calms, mode='extend', field=bpcal, datacolumn='corrected', growtime=80, growfreq=80, flagbackup=False)

    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_crosscal' for entry in flag_versions.values()):
        flagmanager(vis=calms, mode='delete', versionname=obs+'_flag_crosscal', merge='replace')
        print("Found 'flag_crosscal'. Deleting it.")    

    flagmanager(vis=calms, mode='save', versionname=obs+'_flag_crosscal', merge='replace')
    crosscal_flag = flagdata(vis=calms, mode='summary')
    print()
    print('Flagging summary after crosscalibration:')
    flag.log_flagsum(crosscal_flag, logger)

    # Refined delay calibration
    gaincal(vis = calms, caltable = ktab2, selectdata = True,\
        solint = "inf", field = bpcal, combine = "scan",uvrange='',\
        refant = ref_ant, solnorm = False, gaintype = "K",\
        minsnr=3,parang = False, gaintable=[btab, gtab_p, gtab_a])

    os.system(f'/opt/ragavi-env/bin/ragavi-gains --table {ktab2} --field 1 -o {path}/PLOTS/{obs}_Kcal2 -p {path}/PLOTS/{obs}_Kcal2.png')

    # refined phase cal on bandpass calibrator
    gaincal(vis = calms, caltable = gtab_p2, selectdata = True,\
        solint = "60s", field = bpcal, combine = "",\
        refant = ref_ant, gaintype = "G", calmode = "p",uvrange='',refantmode='strict',\
        gaintable = [btab, ktab2], gainfield = ['',''], interp = ['',''],parang = False)

    gaincal(vis = calms, caltable = gtab_a2, selectdata = True,\
        solint = "inf", field = bpcal, combine = "",\
        refant = ref_ant, gaintype = "T", calmode = "a",uvrange='',\
        gaintable = [btab, ktab2, gtab_p2], gainfield = ['','',''], interp = ['','',''],parang = False)

    os.system(f'/opt/ragavi-env/bin/ragavi-gains --table {gtab_a2} --field 1 -o {path}/PLOTS/{obs}_Gcal_amp2 -p {path}/PLOTS/{obs}_Gcal_amp2.png')
    os.system(f'/opt/ragavi-env/bin/ragavi-gains --table {gtab_p2} --field 1 -o {path}/PLOTS/{obs}_Gcal_phase2 -p {path}/PLOTS/{obs}_Gcal_phase2.png')

    # refined bandpass cal on bandpass calibrator
    bandpass(vis = calms, caltable = btab2, selectdata = True,\
            solint = "inf", field = bpcal, combine = "scan", uvrange='',\
            refant = ref_ant, solnorm = False, bandtype = "B",\
        gaintable = [ktab2,gtab_p2,gtab_a2], gainfield = ['','',''],\
        interp = ['','',''], parang = False)

    os.system(f'/opt/ragavi-env/bin/ragavi-gains --table {btab2} --field 1 -o {path}/PLOTS/{obs}_Bpcal2 -p {path}/PLOTS/{obs}_Bpcal2.png')
    
    applycal(vis=calms, field=bpcal, gaintable=[ktab2,gtab_p2,gtab_a2,btab2], applymode='calflag', flagbackup=False)

    os.system(f'/opt/shadems-env/bin/shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_crosscal_XXYY_flagged.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x FREQ -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_crosscal_XYYX_flagged.png')

    #####################################################################
    #Do two iterations of leakage calibration

    # flag corrceted data in XYYX and redo dfcal
    flagdata(vis=calms, mode="rflag", datacolumn="corrected", field=bpcal, quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False, correlation='XY,YX')
    flagdata(vis=calms, mode='extend', datacolumn="corrected", field=bpcal, growtime=80, growfreq=80, flagbackup=False, growaround=True, flagnearfreq=True, correlation='XY,YX')

    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_before_df' for entry in flag_versions.values()):
        flagmanager(vis=calms, mode='delete', versionname=obs+'_flag_before_df', merge='replace')
        logger.info("crosscal: Found 'flag_before_df'. Deleting it.")
    flagmanager(vis=calms, mode='save', versionname=obs+'_flag_before_df', merge='replace')

    # Calibrate Df
    polcal(vis = calms, caltable = ptab_df, selectdata = True,\
            solint = 'inf', field = bpcal, combine = '',\
            refant = ref_ant, poltype = 'Dflls', preavg= 200.0,\
            gaintable = [ktab2, gtab_p2,gtab_a2,btab2],\
            gainfield = ['', '', '',''],\
            smodel=[14.8,0,0,0],\
            interp = ['', '', '', ''])

    # flag df solutions 
    flagdata(vis=ptab_df, mode='tfcrop',datacolumn="CPARAM", quackinterval=0.0,ntime="60s",combinescans=True,timecutoff=5.0, freqcutoff=3.0, usewindowstats="both", flagbackup=False)
    df_flagversions = flagmanager(vis=ptab_df, mode='list')
    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_df' for entry in df_flagversions.values()):
        flagmanager(vis=ptab_df, mode='delete', versionname=obs+'_flag_df', merge='replace')
        logger.info("crosscal: Found 'flag_df'. Deleting it.")
    flagmanager(vis=ptab_df, mode='save', versionname=obs+'_flag_df', merge='replace')

    os.system(f'/opt/ragavi-env/bin/ragavi-gains -x channel --table {ptab_df} --field 1 -o {path}/PLOTS/{obs}_Dfcal_flagged -p {path}/PLOTS/{obs}_Dfcal_flagged.png')

    # Apply Df to bpcal
    applycal(vis=calms,field=bpcal,gaintable=[ktab2,gtab_p2,gtab_a2,btab2,ptab_df],parang=True, flagbackup=False)

    #flag corrected data in XYYX and redo dfcal
    flagdata(vis=calms, mode="rflag", datacolumn="corrected", field=bpcal, quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False, correlation='XY,YX')
    flagdata(vis=calms, mode='extend', datacolumn="corrected", field=bpcal, growtime=80, growfreq=80, flagbackup=False, growaround=True, flagnearfreq=True, correlation='XY,YX')

    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_df_sec_iter' for entry in flag_versions.values()):
        flagmanager(vis=calms, mode='delete', versionname=obs+'_flag_df_sec_iter', merge='replace')
        logger.info("crosscal: Found 'flag_df_sec_iter'. Deleting it.")
    flagmanager(vis=calms, mode='save', versionname=obs+'_flag_df_sec_iter', merge='replace')


    polcal(vis = calms, caltable = ptab_df2, selectdata = True,\
            solint = 'inf', field = bpcal, combine = '',\
            refant = ref_ant, poltype = 'Dflls', preavg= 200.0,\
            gaintable = [ktab2, gtab_p2,gtab_a2,btab2],\
            gainfield = ['', '', '',''],\
            smodel=[14.8,0,0,0],\
            interp = ['', '', '', ''])

    os.system(f'/opt/ragavi-env/bin/ragavi-gains -x channel --table {ptab_df2} --field 1 -o {path}/PLOTS/{obs}_Dfcal_preflag2 -p {path}/PLOTS/{obs}_Dfcal_preflag2.png')
    
    flagdata(vis=ptab_df2, mode='tfcrop',datacolumn="CPARAM", quackinterval=0.0,ntime="60s",combinescans=True,timecutoff=5.0, freqcutoff=3.0, usewindowstats="both", flagbackup=False)
    df_flagversions2 = flagmanager(vis=ptab_df2, mode='list')
    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_df2' for entry in df_flagversions2.values()):
        flagmanager(vis=ptab_df2, mode='delete', versionname=obs+'_flag_df2', merge='replace')
        logger.info("corsscal: Found 'flag_df2'. Deleting it.")
    flagmanager(vis=ptab_df2, mode='save', versionname=obs+'_flag_df2', merge='replace')

    os.system(f'/opt/ragavi-env/bin/ragavi-gains -x channel --table {ptab_df2} --field 1 -o {path}/PLOTS/{obs}_Dfcal_flagged2 -p {path}/PLOTS/{obs}_Dfcal_flagged2.png')
     
    applycal(vis=calms,field=gcal,gaintable=[ktab2,gtab_p2,gtab_a2,btab2,ptab_df2],parang=True, flagbackup=False)

    flagdata(vis=calms, mode="rflag", field=gcal, datacolumn="corrected", quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False)
    flagdata(vis=calms, mode='extend', field=gcal, datacolumn='corrected', growtime=80, growfreq=80, flagbackup=False, growaround=True, flagnearfreq=True)

    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_after_df' for entry in flag_versions.values()):
        flagmanager(vis=calms, mode='delete', versionname=obs+'_flag_after_df', merge='replace')
        logger.info("crosscal: Found 'flag_after_df'. Deleting it.")
    flagmanager(vis=calms, mode='save', versionname=obs+'_flag_after_df', merge='replace')
    df_cal_flag = flagdata(vis=calms, mode='summary')
    logger.info("")
    logger.info('crosscal: Flagging summary after Df calibration:')
    flag.log_flagsum(df_cal_flag, logger)

    # Check that amplitude of leakage cal is gone down (few %) after calibration
    #os.system(f'/opt/shadems-env/bin/shadems {calms} -x FREQ -y DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_Df-DATA.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x FREQ -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_Df-CORRECTED.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_crosscal_XXYY.png')

    logger.info("")
    logger.info("crosscal: Finished calibration of the primary calibrator")
    # log.append_to_google_doc(' CROSSCAL', 'Finished calibration of the primary', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")


    #####################################################################################################
    # Crosscalibration of the secondary calibrator
    logger.info("")
    logger.info('crosscal: Starting crosscalibration of the secondary calibrator')

    #  Calibrate Secondary -p  and T - amplitude normalized to 1
    gaincal(vis = calms, caltable = ktab_sec, selectdata = True,\
        solint = "inf", field = gcal, combine = "",uvrange='',\
        refant = ref_ant, solnorm = False, gaintype = "K",\
        minsnr=3,parang = True, gaintable=[gtab_a2, btab2, ptab_df2])

    gaincal(vis = calms, caltable = gtab_sec_p, selectdata = True,\
        solint = "inf", field = gcal, combine = "",\
        refant = ref_ant, gaintype = "G", calmode = "p",uvrange='',refantmode='strict',\
        gaintable = [ktab_sec, gtab_a2,btab2,ptab_df2], parang = True)

    gaincal(vis = calms, caltable = Ttab_sec, selectdata = True,\
        solint = "inf", field = gcal, combine = "",\
        refant = ref_ant, gaintype = "T", calmode = "ap",uvrange='',refantmode='strict',\
        solnorm=True, gaintable = [ktab_sec, gtab_sec_p,gtab_a2,btab2,ptab_df2], append=False, parang=True)


    # Check calibration of secondary
    applycal(vis=calms, field=gcal, gaintable=[ktab_sec, gtab_sec_p,gtab_a2,btab2,Ttab_sec,ptab_df2], parang=True, flagbackup=False)
    #os.system(f'/opt/ragavi-env/bin/ragavi-vis --ms {calms} -x frequency -y amplitude -dc CORRECTED tbin 12000 -ca antenna1 --corr XY,YX --field {gcal} -o {path}/PLOTS/{obs}_{gcal}-Df-CORRECTED.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x UV -y CORRECTED_DATA:amp -c ANTENNA1 --corr XX,YY --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_amp_XXYY.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x UV -y CORRECTED_DATA:phase -c ANTENNA1 --corr XX,YY --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_phase_XXYY.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY  --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_phaserefine_XXYY.png')


    #apply calibration up to  now to xcal: XY and YX will vary with time due to pang
    applycal(vis=pol_ms,field=xcal,gaintable=[ktab2,gtab_p2,gtab_a2,btab2,ptab_df2],parang=True, flagbackup=False)

    flagdata(vis=pol_ms, mode="rflag", field=xcal, datacolumn="corrected", quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False)
    flagdata(vis=pol_ms, mode='extend', field=xcal, datacolumn='corrected', growtime=80, growfreq=80, flagbackup=False, growaround=True, flagnearfreq=True)
    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_before_xf' for entry in flag_versions.values()):
        flagmanager(vis=pol_ms, mode='delete', versionname=obs+'_flag_before_xf', merge='replace')
        logger.info("crosscal: Found 'flag_before_xf'. Deleting it.")
    flagmanager(vis=pol_ms, mode='save', versionname=obs+'_flag_before_xf', merge='replace')


    os.system(f'/opt/shadems-env/bin/shadems {pol_ms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XXYY.png')
    os.system(f'/opt/shadems-env/bin/shadems {pol_ms} -x TIME -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XYYX.png')

    logger.info("")
    logger.info("crosscal: Finished calibration of the secondary calibrator")
    # log.append_to_google_doc(' CROSSCAL', 'Finished calibration of the secondary', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")

    #############################################################################################################
    #calibration of the polarisation calibrator
    logger.info("")
    logger.info('crosscal: Starting crosscalibration of the polarisation calibrator')

    #  Calibrate XY phase: calibrate P on 3C286 - refine the phase
    gaincal(vis = pol_ms, caltable = ktab_pol, selectdata = True,\
        solint = "inf", field = xcal, combine = "",uvrange='',\
        refant = ref_ant, solnorm = False, gaintype = "K",\
        minsnr=3,parang = True, gaintable=[gtab_a2, btab2, ptab_df2])

    gaincal(vis = pol_ms, caltable = gtab_pol_p, selectdata = True,\
        solint = 'inf', field = xcal, combine = "",scan='',\
        refant = ref_ant, gaintype = "G", calmode = "p",uvrange='',refantmode='strict',\
        gaintable = [ktab_pol, gtab_a2,btab2,ptab_df2], parang = True)

    #selfcal on polarisation calibrator
    tclean(vis=pol_ms,field=xcal,cell='0.5arcsec',imsize=512,niter=1000,imagename=path+'/CAL_IMAGES/'+obs+'_'+xcal+'-selfcal',weighting='briggs',robust=-0.2,datacolumn= 'corrected',deconvolver= 'mtmfs',\
        nterms=2,specmode='mfs',interactive=False)
    gaincal(vis=pol_ms,field=xcal, calmode='p', solint='30s',caltable=gtab_pol_p+'-selfcal',refantmode='strict',\
        refant=ref_ant,gaintype='G',gaintable = [ktab_pol, gtab_a2,btab2,ptab_df2], parang = True)
    gtab_pol_p=gtab_pol_p+"-selfcal"


    gaincal(vis = pol_ms, caltable = Ttab_pol, selectdata = True,\
        solint = "inf", field = xcal, combine = "",\
        refant = ref_ant, gaintype = "T", calmode = "ap",uvrange='', refantmode='strict',\
        solnorm=True, gaintable = [ktab_pol, gtab_pol_p,gtab_a2,btab2,ptab_df2], append=False, parang=True)

    #apply calibration up to  now, including phase refinement to xcal - crosshands should be real vaue dominated, imaginary will give idea of induced elliptcity. change in real axis due to parang

    applycal(vis=pol_ms,field=xcal,gaintable=[ktab_pol, gtab_pol_p, gtab_a2, btab2, Ttab_pol, ptab_df2],parang=False, flagbackup=False)
    os.system(f'/opt/shadems-env/bin/shadems {pol_ms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XXYY_refinephase.png')
    os.system(f'/opt/shadems-env/bin/shadems {pol_ms} -x CORRECTED_DATA:imag -y CORRECTED_DATA:real -c CORR --corr XY,YX --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XYYX_real_im.png')


    # Cross-hand delay calibration - 
    gaincal(vis = pol_ms, caltable = kxtab, selectdata = True,\
                solint = "inf", field = xcal, combine = "scan", scan='',\
                refant = ref_ant, gaintype = "KCROSS",\
                gaintable = [ktab_pol, gtab_pol_p, btab2, ptab_df2],\
                #smodel=[15.7433,0.8628247336,1.248991241,0],\
                parang = True)


    # Calibrate XY phase
    polcal(vis = pol_ms, caltable = ptab_xf, selectdata = True,scan='',combine='scan',\
        solint = "1200s,20MHz", field = xcal, uvrange='',\
        refant = ref_ant, poltype = "Xf",  gaintable = [ktab_pol, gtab_pol_p, kxtab,btab2,ptab_df2],\
              #smodel= [15.7433,0.8628247336,1.248991241,0]\
              )

    
    os.system(f'ragavi-gains --table {ptab_xf} --field 0 -o {path}/PLOTS/{obs}_Xfcal -p {path}/PLOTS/{obs}_Xfcal.png')
    
    logger.info("")
    logger.info('crosscal: Correcting for phase ambiguity')
    #exec(open('/localwork/angelina/meerkat_virgo/ViMS/ViMS/scripts/xyamb_corr.py').read())
    S=xyamb(logger, xytab=ptab_xf ,xyout=ptab_xfcorr)

    os.system(f'ragavi-gains --table {ptab_xfcorr} --field 0 -o {path}/PLOTS/{obs}_Xfcal_ambcorr -p {path}/PLOTS/{obs}_Xfcal_ambcorr.png')

    logger.info("")
    logger.info("crosscal: Finished calibration of the polarisation calibrator")
    # log.append_to_google_doc(' CROSSCAL', 'Finished calibration of the polcal', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")

    applycal(vis=pol_ms, field=xcal,gaintable=[ktab_pol, gtab_pol_p, gtab_a2, btab2, Ttab_pol, ptab_df2, kxtab, ptab_xfcorr],parang=True, flagbackup=False)
    applycal(vis=calms, field=gcal,gaintable=[ktab_sec, gtab_sec_p, gtab_a2, btab2, Ttab_sec, ptab_df2, kxtab, ptab_xfcorr], parang=True, flagbackup=False)
    applycal(vis=calms, field=fcal,gaintable=[ktab2, gtab_p2, gtab_a2, btab2, ptab_df2, kxtab, ptab_xfcorr], parang=True, flagbackup=False)

    final_flag = flagdata(vis=calms, mode='summary')
    logger.info("")
    logger.info('crosscal: Flagging summary after cross and pol calibration:')
    flag.log_flagsum(final_flag, logger)

    #-------------------------------------------------------
    final_flag_pol = flagdata(vis=pol_ms, mode='summary')
    logger.info("")
    logger.info('crosscal: Flagging summary after cross and pol calibration for the polarisation calibrator:')
    flag.log_flagsum(final_flag_pol, logger)   

    # Check: plot imaginary versis real and compare to previous plot

    os.system(f'/opt/shadems-env/bin/shadems {pol_ms} -x CORRECTED_DATA:imag -y CORRECTED_DATA:real -c CORR --corr XY,YX --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_aftercalXf_XYYX_real_im.png')
    os.system(f'ragavi-vis --ms {pol_ms} -x frequency -y phase -dc CORRECTED tbin 12000 -ca antenna1 --corr XY,YX --field {xcal} -o {path}/PLOTS/{obs}_{xcal}_aftercalXf_XYYX_phase.png')

#------------------------------------------------------

def run(logger, obs_id, cal_ms, pol_ms, path):
    # log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("###################### CROSSCAL ######################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    
    # do cross- and polarisation calibration
    try:
        logger.info("\n\n\n\n\n")
        logger.info("CROSSCAL: starting crosscalibration...")
        crosscal(logger, obs_id, cal_ms, pol_ms, path)
        logger.info("Crosscal step completed successfully!")
        logger.info("")
        logger.info("")
        logger.info("######################################################")
        logger.info("#################### END CROSSCAL ####################")
        logger.info("######################################################")
        logger.info("")
        logger.info("")
    except Exception as e:
        logger.exception("CROSSCAL: crosscalibration failed")
    