#!/usr/bin/env python3

import numpy as np
import ephem
from astropy.coordinates import SkyCoord
from pyrap.tables import table as tbl
from pyrap.tables import taql
from pyrap.quanta import quantity
from pyrap.measures import measures
from astropy import units
import datetime
import pytz
import os,sys
from utils import utils, log
from casatasks import *
from casatools import table

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

#-------------------------------------------------------------------------------------------------
def correct_parang(logger, ms_file, fields, ddid=0, storecolumn='DATA', rawcolumn='DATA'):
    """
    swaps feeds of the given fields. Saves swapped feeds in DATA column per default

    Args:
        ms_file (str): ms file to correct
        fields (list): list of field IDs to correct

    Returns:

    """
    ephemobservatory = ephem.Observer()
    ephemobservatory.epoch = ephem.J2000

    with tbl(ms_file+"::FIELD", ack=False) as t:
        fieldnames = t.getcol("NAME")
        pos = t.getcol("PHASE_DIR")

    for f in fields:
        logger.info("Processing field {}: {}".format(f, fieldnames[f]))

        with tbl(ms_file, ack=False) as t:
            with taql("select * from $t where FIELD_ID=={}".format(f)) as tt:
                def __to_datetimestr(t):
                    dt = datetime.datetime.utcfromtimestamp(quantity("{}s".format(t)).to_unix_time())
                    return dt.strftime("%Y/%m/%d %H:%M:%S")
                dbtime = tt.getcol("TIME_CENTROID")
                start_time_Z = __to_datetimestr(dbtime.min())
                end_time_Z = __to_datetimestr(dbtime.max())
                logger.info("Observation spans '{}' and '{}' UTC".format(
                        start_time_Z, end_time_Z))
        dm = measures()
        ephemobservatory.date = start_time_Z
        st = ephemobservatory.date
        ephemobservatory.date = end_time_Z
        et = ephemobservatory.date
        TO_SEC = 3600*24.0
        nstep = int(np.round((float(et)*TO_SEC - float(st)*TO_SEC) / (1*60.)))
        timepa = time = np.linspace(st,et,nstep)
        timepadt = list(map(lambda x: ephem.Date(x).datetime(), time))

        with tbl(ms_file+"::ANTENNA", ack=False) as t:
            anames = t.getcol("NAME")
            apos = t.getcol("POSITION")
            aposdm = list(map(lambda pos: dm.position('itrf',*[ quantity(x,'m') for x in pos ]),
                            apos))

        with tbl(ms_file+"::FIELD", ack=False) as t:
            fieldnames = t.getcol("NAME")
            pos = t.getcol("PHASE_DIR")
        skypos = SkyCoord(pos[f][0,0]*units.rad, pos[f][0,1]*units.rad, frame="fk5")
        rahms = "{0:.0f}:{1:.0f}:{2:.5f}".format(*skypos.ra.hms)
        decdms = "{0:.0f}:{1:.0f}:{2:.5f}".format(skypos.dec.dms[0], abs(skypos.dec.dms[1]), abs(skypos.dec.dms[2]))
        fieldEphem = ephem.readdb(",f|J,{},{},0.0".format(rahms, decdms))
        logger.info("Using coordinates of field '{}' for body: J2000, {}, {}".format(fieldnames[f],
                                                                                    np.rad2deg(pos[f][0,0]),
                                                                                    np.rad2deg(pos[f][0,1])))
        
        with tbl(ms_file+"::DATA_DESCRIPTION", ack=False) as t:
            if ddid < 0 or ddid >= t.nrows():
                raise RuntimeError("Invalid DDID selected")
            spwsel = t.getcol("SPECTRAL_WINDOW_ID")[ddid]
            poldescsel = t.getcol("POLARIZATION_ID")[ddid]
        

        az = np.zeros(nstep, dtype=np.float32)
        el = az.copy()
        ra = az.copy()
        #racc = az.copy()
        dec = az.copy()
        #deccc = az.copy()
        arraypa = az.copy()
        pa = np.zeros((len(anames), nstep), np.float32)

        zenith = dm.direction('AZELGEO','0deg','90deg')
        
        with tbl(ms_file+"::POLARIZATION", ack=False) as t:
            poltype = t.getcol("CORR_TYPE")[poldescsel]
            # must be linear
            if any(poltype - np.array([9,10,11,12]) != 0):
                raise RuntimeError("Must be full correlation linear system being corrected")


        with tbl(ms_file+"::SPECTRAL_WINDOW", ack=False) as t:
            chan_freqs = t.getcol("CHAN_FREQ")[spwsel]
            chan_width = t.getcol("CHAN_WIDTH")[spwsel]
            nchan = chan_freqs.size
            logger.info("Will apply to SPW {0:d} ({3:d} channels): {1:.2f} to {2:.2f} MHz".format(
                spwsel, chan_freqs.min()*1e-6, chan_freqs.max()*1e-6, nchan))
        list_apply = []

        logger.info("Will flip the visibility hands per user request")
        list_apply.append("Anti-diagonal Jones")
        
        logger.info("Arranging to apply (inversion):")
        for j in list_apply:
            logger.info("\t{}".format(j))
        
        logger.info("Storing corrected data into '{}'".format(storecolumn))
        timepaunix = np.array(list(map(lambda x: x.replace(tzinfo=pytz.UTC).timestamp(), timepadt)))
        nrowsput = 0
        with tbl(ms_file, ack=False, readonly=False) as t:
            if storecolumn not in t.colnames():
                logger.info(f"Inserting column {storecolumn}. Do not interrupt")
                add_column(t, storecolumn)
                logger.info(f"Inserted column {storecolumn}")
            with taql("select * from $t where FIELD_ID=={} and DATA_DESC_ID=={}".format(f, ddid)) as tt:
                nrow = tt.nrows()
                nchunk = nrow // 1000 + int(nrow % 1000 > 0)
                for ci in range(nchunk):
                    cl = ci * 1000
                    crow = min(nrow - ci * 1000, 1000)
                    data = tt.getcol(rawcolumn, startrow=cl, nrow=crow)
                    if data.shape[2] != 4:
                        raise RuntimeError("Data must be full correlation")
                    data = data.reshape(crow, nchan, 2, 2)

                    def __casa_to_unixtime(t):
                        dt = quantity("{}s".format(t)).to_unix_time()
                        return dt
                    mstimecentroid = tt.getcol("TIME", startrow=cl, nrow=crow)
                    msuniqtime = np.unique(mstimecentroid)
                    # expensive quanta operation -- do only for unique values
                    uniqtimemsunix = np.array(list(map(__casa_to_unixtime, msuniqtime)))
                    timemsunixindex = np.array(list(map(lambda t: np.argmin(np.abs(msuniqtime-t)),
                                                            mstimecentroid)))
                    timemsunix = uniqtimemsunix[timemsunixindex]
                    a1 = tt.getcol("ANTENNA1", startrow=cl, nrow=crow)
                    a2 = tt.getcol("ANTENNA2", startrow=cl, nrow=crow)

                    def give_lin_Rmat(paA, nchan, conjugate=False):
                        N = paA.shape[0] # nrow
                        c = np.cos(paA).repeat(nchan)
                        s = np.sin(paA).repeat(nchan)
                        if conjugate:
                            return np.array([c,s,-s,c]).T.reshape(N, nchan, 2, 2)
                        else:
                            return np.array([c,-s,s,c]).T.reshape(N, nchan, 2, 2)

                    def give_crossphase_mat(phase, nrow, nchan, conjugate=False):
                        ones = np.ones(nchan*nrow)
                        zeros = np.zeros(nchan*nrow)
                        e = np.exp((1.j if not conjugate else -1.j) * np.deg2rad(phase)) * ones
                        return np.array([e,zeros,zeros,ones]).T.reshape(nrow, nchan, 2, 2)

                    # need to apply anti-diagonal

                    FVmat = np.array([np.zeros(nchan*crow),
                                        np.ones(nchan*crow),
                                        np.ones(nchan*crow),
                                        np.zeros(nchan*crow)]).T.reshape(crow, nchan, 2, 2)
                    
                    # cojugate exp for left antenna
                    XA1 = give_crossphase_mat(0.0, nrow=crow, nchan=nchan,conjugate=True)
                    XA2 = give_crossphase_mat(0.0, nrow=crow, nchan=nchan,conjugate=False)

                    JA1 = np.matmul(FVmat, XA1)
                    JA2 = np.matmul(XA2, FVmat)

                    corr_data = np.matmul(JA1, np.matmul(data, JA2)).reshape(crow, nchan, 4)
                    tt.putcol(storecolumn, corr_data, startrow=cl, nrow=crow)
                    logger.info("\tCorrected chunk {}/{}".format(ci+1, nchunk))
                    nrowsput += crow
            assert nrow == nrowsput


#----------------------------------------------------------------------------------
def crosscal(logger, obs_id, cal_ms, path, fields=[2,2,0,1], ref_ant='m000'):
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
    #get names of the fields from table (fields are in order fcal, bpcal, gcal, xcal)

    tb = table()
    tb.open(calms+ "/FIELD")
    field_names = tb.getcol("NAME")
    tb.close()

    fcal = field_names[fields[0]]
    bpcal = field_names[fields[1]]
    gcal = field_names[fields[2]]
    xcal = field_names[fields[3]]


    ################################################################
    # Change RECEPTOR_ANGLE : DEFAULT IS -90DEG 

    tb.open(calms+'/FEED', nomodify=False)
    feed_angle = tb.getcol('RECEPTOR_ANGLE')
    new_feed_angle = np.zeros(feed_angle.shape)
    tb.putcol('RECEPTOR_ANGLE', new_feed_angle)
    tb.close()


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

    ###################################################################
    # Set model for flux and polarisation calibrator


    for cal in set(fcal.split(',')+bpcal.split(',')+xcal.split(',')):

        if cal == 'J1939-6342':
            logger.info('crosscal: setting model for flux calibrator J1939-6342')
            setjy(vis = calms, field = "{0}".format(fcal), spw = "", selectdata = False, timerange = "", scan = "", \
                standard = 'Stevens-Reynolds 2016', scalebychan = True, useephemdir = False, usescratch = True)


        elif cal =='J1331+3030':
            logger.info('crosscal: setting model for polarisation calibrator J1331+3030')

            
            reffreq = '1.284GHz'
            # Stokes flux density in Jy
            I =  15.74331687
            Q = 0.8628247336
            U = 1.248991241
            V = 0   
            # Spectral Index in 2nd order
            alpha =   [-0.4881438633844231, -0.17025582235426978]

            setjy(vis=calms,
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
            logger.error("corsscal: Unknown calibrator, insert model in the script please ", cal)
            sys.exit()

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

    os.system(f'ragavi-gains --table {ktab} --field 2 --htmlname {path}/PLOTS/{obs}_Kcal --plotname {path}/PLOTS/{obs}_Kcal.png')

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

    os.system(f'ragavi-gains --table {gtab_a} --field 2 -o {path}/PLOTS/{obs}_Gcal_amp -p {path}/PLOTS/{obs}_Gcal_amp.png')
    os.system(f'ragavi-gains --table {gtab_p} --field 2 -o {path}/PLOTS/{obs}_Gcal_phase -p {path}/PLOTS/{obs}_Gcal_phase.png')

    # bandpass cal on bandpass calibrator
    bandpass(vis = calms, caltable = btab, selectdata = True,\
            solint = "inf", field = bpcal, combine = "scan", uvrange='',\
            refant = ref_ant, solnorm = False, bandtype = "B",\
        gaintable = [ktab,gtab_p,gtab_a], gainfield = ['','',''],\
        interp = ['','',''], parang = False)

    os.system(f'ragavi-gains --table {btab} --field 2 -o {path}/PLOTS/{obs}_Bpcal -p {path}/PLOTS/{obs}_Bpcal.png')
 
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

    os.system(f'ragavi-gains --table {ktab2} --field 2 -o {path}/PLOTS/{obs}_Kcal2 -p {path}/PLOTS/{obs}_Kcal2.png')

    # refined phase cal on bandpass calibrator
    gaincal(vis = calms, caltable = gtab_p2, selectdata = True,\
        solint = "60s", field = bpcal, combine = "",\
        refant = ref_ant, gaintype = "G", calmode = "p",uvrange='',refantmode='strict',\
        gaintable = [btab, ktab2], gainfield = ['',''], interp = ['',''],parang = False)

    gaincal(vis = calms, caltable = gtab_a2, selectdata = True,\
        solint = "inf", field = bpcal, combine = "",\
        refant = ref_ant, gaintype = "T", calmode = "a",uvrange='',\
        gaintable = [btab, ktab2, gtab_p2], gainfield = ['','',''], interp = ['','',''],parang = False)

    os.system(f'ragavi-gains --table {gtab_a2} --field 2 -o {path}/PLOTS/{obs}_Gcal_amp2 -p {path}/PLOTS/{obs}_Gcal_amp2.png')
    os.system(f'ragavi-gains --table {gtab_p2} --field 2 -o {path}/PLOTS/{obs}_Gcal_phase2 -p {path}/PLOTS/{obs}_Gcal_phase2.png')

    # refined bandpass cal on bandpass calibrator
    bandpass(vis = calms, caltable = btab2, selectdata = True,\
            solint = "inf", field = bpcal, combine = "scan", uvrange='',\
            refant = ref_ant, solnorm = False, bandtype = "B",\
        gaintable = [ktab2,gtab_p2,gtab_a2], gainfield = ['','',''],\
        interp = ['','',''], parang = False)

    os.system(f'ragavi-gains --table {btab2} --field 2 -o {path}/PLOTS/{obs}_Bpcal2 -p {path}/PLOTS/{obs}_Bpcal2.png')
    
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

    os.system(f'ragavi-gains -x channel --table {ptab_df} --field 2 -o {path}/PLOTS/{obs}_Dfcal_flagged -p {path}/PLOTS/{obs}_Dfcal_flagged.png')

    # Apply Df to bpcal
    applycal(vis=calms,field=fcal,gaintable=[ktab2,gtab_p2,gtab_a2,btab2,ptab_df],parang=True, flagbackup=False)

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

    os.system(f'ragavi-gains -x channel --table {ptab_df2} --field 2 -o {path}/PLOTS/{obs}_Dfcal_preflag2 -p {path}/PLOTS/{obs}_Dfcal_preflag2.png')
    
    flagdata(vis=ptab_df2, mode='tfcrop',datacolumn="CPARAM", quackinterval=0.0,ntime="60s",combinescans=True,timecutoff=5.0, freqcutoff=3.0, usewindowstats="both", flagbackup=False)
    df_flagversions2 = flagmanager(vis=ptab_df2, mode='list')
    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_df2' for entry in df_flagversions2.values()):
        flagmanager(vis=ptab_df2, mode='delete', versionname=obs+'_flag_df2', merge='replace')
        logger.info("corsscal: Found 'flag_df2'. Deleting it.")
    flagmanager(vis=ptab_df2, mode='save', versionname=obs+'_flag_df2', merge='replace')

    os.system(f'ragavi-gains -x channel --table {ptab_df2} --field 2 -o {path}/PLOTS/{obs}_Dfcal_flagged2 -p {path}/PLOTS/{obs}_Dfcal_flagged2.png')
     
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
    os.system(f'ragavi-vis --ms {calms} -x frequency -y amplitude -dc CORRECTED tbin 12000 -ca antenna1 --corr XY,YX --field {gcal} -o {path}/PLOTS/{obs}_{gcal}-Df-CORRECTED.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x UV -y CORRECTED_DATA:amp -c ANTENNA1 --corr XX,YY --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_amp_XXYY.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x UV -y CORRECTED_DATA:phase -c ANTENNA1 --corr XX,YY --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_phase_XXYY.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY  --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_phaserefine_XXYY.png')


    #apply calibration up to  now to xcal: XY and YX will vary with time due to pang
    applycal(vis=calms,field=xcal,gaintable=[ktab,gtab_p2,gtab_a2,btab2,ptab_df2],parang=True, flagbackup=False)

    flagdata(vis=calms, mode="rflag", field=xcal, datacolumn="corrected", quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False)
    flagdata(vis=calms, mode='extend', field=xcal, datacolumn='corrected', growtime=80, growfreq=80, flagbackup=False, growaround=True, flagnearfreq=True)
    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_before_xf' for entry in flag_versions.values()):
        flagmanager(vis=calms, mode='delete', versionname=obs+'_flag_before_xf', merge='replace')
        logger.info("crosscal: Found 'flag_before_xf'. Deleting it.")
    flagmanager(vis=calms, mode='save', versionname=obs+'_flag_before_xf', merge='replace')


    os.system(f'/opt/shadems-env/bin/shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XXYY.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x TIME -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XYYX.png')

    logger.info("")
    logger.info("crosscal: Finished calibration of the secondary calibrator")
    # log.append_to_google_doc(' CROSSCAL', 'Finished calibration of the secondary', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")

    #############################################################################################################
    #calibration of the polarisation calibrator
    logger.info("")
    logger.info('crosscal: Starting crosscalibration of the polarisation calibrator')

    #  Calibrate XY phase: calibrate P on 3C286 - refine the phase
    gaincal(vis = calms, caltable = ktab_pol, selectdata = True,\
        solint = "inf", field = xcal, combine = "",uvrange='',\
        refant = ref_ant, solnorm = False, gaintype = "K",\
        minsnr=3,parang = True, gaintable=[gtab_a2, btab2, ptab_df2])

    gaincal(vis = calms, caltable = gtab_pol_p, selectdata = True,\
        solint = 'inf', field = xcal, combine = "",scan='',\
        refant = ref_ant, gaintype = "G", calmode = "p",uvrange='',refantmode='strict',\
        gaintable = [ktab_pol, gtab_a2,btab2,ptab_df2], parang = True)

    #selfcal on polarisation calibrator
    tclean(vis=calms,field=xcal,cell='0.5arcsec',imsize=512,niter=1000,imagename=path+'/CAL_IMAGES/'+obs+'_'+xcal+'-selfcal',weighting='briggs',robust=-0.2,datacolumn= 'corrected',deconvolver= 'mtmfs',\
        nterms=2,specmode='mfs',interactive=False)
    gaincal(vis=calms,field=xcal, calmode='p', solint='30s',caltable=gtab_pol_p+'-selfcal',refantmode='strict',\
        refant=ref_ant,gaintype='G',gaintable = [ktab_pol, gtab_a2,btab2,ptab_df2], parang = True)
    gtab_pol_p=gtab_pol_p+"-selfcal"


    gaincal(vis = calms, caltable = Ttab_pol, selectdata = True,\
        solint = "inf", field = xcal, combine = "",\
        refant = ref_ant, gaintype = "T", calmode = "ap",uvrange='', refantmode='strict',\
        solnorm=True, gaintable = [ktab_pol, gtab_pol_p,gtab_a2,btab2,ptab_df2], append=False, parang=True)

    #apply calibration up to  now, including phase refinement to xcal - crosshands should be real vaue dominated, imaginary will give idea of induced elliptcity. change in real axis due to parang

    applycal(vis=calms,field=xcal,gaintable=[ktab_pol, gtab_pol_p, gtab_a2, btab2, Ttab_pol, ptab_df2],parang=False, flagbackup=False)
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XXYY_refinephase.png')
    os.system(f'/opt/shadems-env/bin/shadems {calms} -x CORRECTED_DATA:imag -y CORRECTED_DATA:real -c CORR --corr XY,YX --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XYYX_real_im.png')


    # Cross-hand delay calibration - 
    gaincal(vis = calms, caltable = kxtab, selectdata = True,\
                solint = "inf", field = xcal, combine = "scan", scan='',\
                refant = ref_ant, gaintype = "KCROSS",\
                gaintable = [ktab_pol, gtab_pol_p, btab2, ptab_df2],\
                #smodel=[15.7433,0.8628247336,1.248991241,0],\
                parang = True)


    # Calibrate XY phase
    polcal(vis = calms, caltable = ptab_xf, selectdata = True,scan='',combine='scan',\
        solint = "1200s,20MHz", field = xcal, uvrange='',\
        refant = ref_ant, poltype = "Xf",  gaintable = [ktab_pol, gtab_pol_p, kxtab,btab2,ptab_df2],\
              #smodel= [15.7433,0.8628247336,1.248991241,0]\
              )

    
    os.system(f'ragavi-gains --table {ptab_xf} --field 1 -o {path}/PLOTS/{obs}_Xfcal -p {path}/PLOTS/{obs}_Xfcal.png')
    
    logger.info("")
    logger.info('crosscal: Correcting for phase ambiguity')
    #exec(open('/localwork/angelina/meerkat_virgo/ViMS/ViMS/scripts/xyamb_corr.py').read())
    S=xyamb(logger, xytab=ptab_xf ,xyout=ptab_xfcorr)

    os.system(f'ragavi-gains --table {ptab_xfcorr} --field 1 -o {path}/PLOTS/{obs}_Xfcal_ambcorr -p {path}/PLOTS/{obs}_Xfcal_ambcorr.png')

    logger.info("")
    logger.info("crosscal: Finished calibration of the polarisation calibrator")
    # log.append_to_google_doc(' CROSSCAL', 'Finished calibration of the polcal', warnings="", plot_link="", doc_name="ViMS Pipeline Plots")

    applycal(vis=calms, field=xcal,gaintable=[ktab_pol, gtab_pol_p, gtab_a2, btab2, Ttab_pol, ptab_df2, kxtab, ptab_xfcorr],parang=True, flagbackup=False)
    applycal(vis=calms, field=gcal,gaintable=[ktab_sec, gtab_sec_p, gtab_a2, btab2, Ttab_sec, ptab_df2, kxtab, ptab_xfcorr], parang=True, flagbackup=False)
    applycal(vis=calms, field=fcal,gaintable=[ktab2, gtab_p2, gtab_a2, btab2, ptab_df2, kxtab, ptab_xfcorr], parang=True, flagbackup=False)

    final_flag = flagdata(vis=calms, mode='summary')
    logger.info("")
    logger.info('crosscal: Flagging summary after cross and pol calibration:')
    flag.log_flagsum(final_flag, logger)

    # Check: plot imaginary versis real and compare to previous plot

    os.system(f'/opt/shadems-env/bin/shadems {calms} -x CORRECTED_DATA:imag -y CORRECTED_DATA:real -c CORR --corr XY,YX --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_aftercalXf_XYYX_real_im.png')
    os.system(f'ragavi-vis --ms {calms} -x frequency -y phase -dc CORRECTED tbin 12000 -ca antenna1 --corr XY,YX --field {xcal} -o {path}/PLOTS/{obs}_{xcal}_aftercalXf_XYYX_phase.png')

#------------------------------------------------------

def run(logger, obs_id, cal_ms, path):
    # log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("###################### CROSSCAL ######################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    
    # swap feeds
    try:
        logger.info("\n\n\n\n\n")
        logger.info("CROSSCAL: starting Feedswap...")
        correct_parang(logger, cal_ms, [0, 1, 2])
        logger.info('CROSSCAL: finished Feedswap')
        logger.info("")
        logger.info("")
        logger.info("")
    except Exception as e:
        logger.exception("CROSSCAL: Feedswap failed")
    
    # do cross- and polarisation calibration
    try:
        logger.info("\n\n\n\n\n")
        logger.info("CROSSCAL: starting crosscalibration...")
        crosscal(logger, obs_id, cal_ms, path)
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
    