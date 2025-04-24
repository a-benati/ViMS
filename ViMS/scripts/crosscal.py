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

def correct_parang(logger, ms_file, fields):
    """
    Correction of the parallactic angle

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
        nstep = int(np.round((float(et)*TO_SEC - float(st)*TO_SEC) / (args.stepsize*60.)))
        if not args.noparang:
            logger.info("Computing PA in {} steps of {} mins each".format(nstep, args.stepsize))
        timepa = time = np.linspace(st,et,nstep)
        timepadt = list(map(lambda x: ephem.Date(x).datetime(), time))

        with tbl(ms_file+"::ANTENNA", ack=False) as t:
            anames = t.getcol("NAME")
            apos = t.getcol("POSITION")
            aposdm = list(map(lambda pos: dm.position('itrf',*[ quantity(x,'m') for x in pos ]),
                            apos))

        if args.ephem:
            with tbl(ms_file+"::FIELD", ack=False) as t:
                fieldnames = t.getcol("NAME")
            fieldEphem = getattr(ephem, args.ephem, None)()
            if not fieldEphem:
                raise RuntimeError("Body {} not defined by PyEphem".format(args.ephem))
            logger.info("Overriding stored ephemeris in database '{}' field '{}' by special PyEphem body '{}'".format(
                ms_file, fieldnames[f], args.ephem))
        else:
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
            if args.ddid < 0 or args.ddid >= t.nrows():
                raise RuntimeError("Invalid DDID selected")
            spwsel = t.getcol("SPECTRAL_WINDOW_ID")[args.ddid]
            poldescsel = t.getcol("POLARIZATION_ID")[args.ddid]
        

        az = np.zeros(nstep, dtype=np.float32)
        el = az.copy()
        ra = az.copy()
        #racc = az.copy()
        dec = az.copy()
        #deccc = az.copy()
        arraypa = az.copy()
        pa = np.zeros((len(anames), nstep), np.float32)

        zenith = dm.direction('AZELGEO','0deg','90deg')
        if not args.noparang:
            for ti, t in enumerate(time):
                ephemobservatory.date = t
                t_iso8601 = ephemobservatory.date.datetime().strftime("%Y-%m-%dT%H:%M:%S.%f")
                fieldEphem.compute(ephemobservatory)
                az[ti] = fieldEphem.az
                el[ti] = fieldEphem.alt
                ra[ti] = fieldEphem.a_ra
                dec[ti] = fieldEphem.a_dec
                arraypa[ti] = fieldEphem.parallactic_angle()
                # compute PA per antenna
                field_centre = dm.direction('J2000', quantity(ra[ti],"rad"), quantity(dec[ti],"rad"))
                dm.do_frame(dm.epoch("UTC", quantity(t_iso8601)))
                #dm.doframe(aposdm[0])
                #field_centre = dm.measure(dm.direction('moon'), "J2000")
                #racc[ti] = field_centre["m0"]["value"]
                #deccc[ti] = field_centre["m1"]["value"]
                for ai in range(len(anames)):
                    dm.doframe(aposdm[ai])
                    pa[ai, ti] = dm.posangle(field_centre,zenith).get_value("rad")
            if args.plot:
                def __angdiff(a, b):
                    return ((a-b) + 180) % 360 - 180
                for axl, axd in zip(["Az", "El", "RA", "DEC", "ParAng"],
                                    [az, el, ra, dec, pa]):
                    hfmt = mdates.DateFormatter('%H:%M')
                    fig = plt.figure(figsize=(16,8))
                    ax = fig.add_subplot(111)
                    ax.set_xlabel("Time UTC")
                    ax.set_ylabel("{} [deg]".format(axl))
                    if axl == "ParAng":
                        ax.errorbar(timepadt,
                                    np.rad2deg(np.mean(axd, axis=0)),
                                    capsize=2,
                                    yerr=0.5*__angdiff(np.rad2deg(axd.max(axis=0)),
                                                    np.rad2deg(axd.min(axis=0))), label="CASACORE")
                        plt.plot(timepadt, np.rad2deg(arraypa), label="PyEphem")
                    else:
                        ax.plot(timepadt, np.rad2deg(axd))
                    ax.xaxis.set_major_formatter(hfmt)
                    ax.grid(True)
                    plt.show()

            with tbl(ms_file+"::FEED", ack=False) as t:
                with taql("select * from $t where SPECTRAL_WINDOW_ID=={}".format(spwsel)) as tt:
                    receptor_aid = tt.getcol("ANTENNA_ID")
                    if len(receptor_aid) != len(anames):
                        raise RuntimeError("Receptor angles not all filed for the antennas in the ::FEED keyword table")
                    receptor_angles = dict(zip(receptor_aid, tt.getcol("RECEPTOR_ANGLE")[:,0]))
                    if args.fa is not None:
                        receptor_angles[...] = float(args.fa)
                        logger.info("Overriding F Jones angle to {0:.3f} for all antennae".format(float(args.fa)))
                    else:
                        logger.info("Applying the following feed angle offsets to parallactic angles:")
                        for ai, an in enumerate(anames):
                            logger.info("\t {0:s}: {1:.3f} degrees".format(an, np.rad2deg(receptor_angles.get(ai, 0.0))))

                raarr = np.empty(len(anames), dtype=int)
                for aid in range(len(anames)):
                    raarr[aid] = receptor_angles[aid]


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
        if abs(args.crossphase) > 1.0e-6:
            logger.info("Applying crosshand phase matrix (X) with {0:.3f} degrees".format(args.crossphase))
            list_apply.append("X Jones")
        if not args.noparang:
            list_apply.append("P+F Jones")
        if args.flipfeeds:
            logger.info("Will flip the visibility hands per user request")
            list_apply.append("Anti-diagonal Jones")
        logger.info("Arranging to apply (inversion):")
        for j in list_apply:
            logger.info("\t{}".format(j))
        if args.invertpa:
            log.warning("Note: Applying corrupting P+F Jones, instead of correction per user request")

        if not args.sim:
            logger.info("Storing corrected data into '{}'".format(args.storecolumn))
            timepaunix = np.array(list(map(lambda x: x.replace(tzinfo=pytz.UTC).timestamp(), timepadt)))
            nrowsput = 0
            with tbl(ms_file, ack=False, readonly=False) as t:
                if args.storecolumn not in t.colnames():
                    logger.info(f"Inserting column {args.storecolumn}. Do not interrupt")
                    add_column(t, args.storecolumn)
                    logger.info(f"Inserted column {args.storecolumn}")
                with taql("select * from $t where FIELD_ID=={} and DATA_DESC_ID=={}".format(f, args.ddid)) as tt:
                    nrow = tt.nrows()
                    nchunk = nrow // args.chunksize + int(nrow % args.chunksize > 0)
                    for ci in range(nchunk):
                        cl = ci * args.chunksize
                        crow = min(nrow - ci * args.chunksize, args.chunksize)
                        data = tt.getcol(args.rawcolumn, startrow=cl, nrow=crow)
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
                        if args.flipfeeds:
                            FVmat = np.array([np.zeros(nchan*crow),
                                            np.ones(nchan*crow),
                                            np.ones(nchan*crow),
                                            np.zeros(nchan*crow)]).T.reshape(crow, nchan, 2, 2)
                        else: # ignore step
                            FVmat = np.array([np.ones(nchan*crow),
                                            np.zeros(nchan*crow),
                                            np.zeros(nchan*crow),
                                            np.ones(nchan*crow)]).T.reshape(crow, nchan, 2, 2)
                        # cojugate exp for left antenna
                        XA1 = give_crossphase_mat(args.crossphase, nrow=crow, nchan=nchan,conjugate=True)
                        XA2 = give_crossphase_mat(args.crossphase, nrow=crow, nchan=nchan,conjugate=False)

                        if not args.noparang:
                            # nearest neighbour interp to computed ParAng
                            pamap = np.array(list(map(lambda x: np.argmin(np.abs(x - timepaunix)), timemsunix)))

                            # apply receptor angles and get a PA to apply per row
                            # assume same PA for all antennas, different F Jones per antenna possibly
                            paA1 = pa[a1, pamap] + raarr[a1]
                            paA2 = pa[a2, pamap] + raarr[a2]

                            PA1 = give_lin_Rmat(paA1, nchan=nchan, conjugate=(args.invertpa))
                            PA2 = give_lin_Rmat(paA2, nchan=nchan, conjugate=(not args.invertpa))
                            JA1 = np.matmul(FVmat, np.matmul(PA1, XA1))
                            JA2 = np.matmul(np.matmul(XA2, PA2), FVmat)
                        else:
                            JA1 = np.matmul(FVmat, XA1)
                            JA2 = np.matmul(XA2, FVmat)

                        corr_data = np.matmul(JA1, np.matmul(data, JA2)).reshape(crow, nchan, 4)
                        tt.putcol(args.storecolumn, corr_data, startrow=cl, nrow=crow)
                        logger.info("\tCorrected chunk {}/{}".format(ci+1, nchunk))
                        nrowsput += crow
                assert nrow == nrowsput
        else:
            logger.info("Simulating correction only -- no changes applied to data")


def crosscal(logger, obs_id, cal_ms, path, fields=[2,2,0,1], ref_ant='m000'):
    import numpy as np
    import os,sys
    from scripts import flag, xyamb_corr
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

    if isinstance(flag_versions, dict) and 'version' in flag_versions.values():
        initial_flag = any(entry['name'] == obs+'_flag_after' for entry in flag_versions.values())
    else:
        initial_flag = False

    if initial_flag:
        flagmanager(vis=calms, mode='restore', versionname=obs+'_flag_after', merge='replace')
        logger.info("crosscal: Found '"+obs+"_flag_after'. Restoring it.")

    else:
        flagmanager(vis=calms, mode='save', versionname=obs+'_flag_after', merge='replace')
        logger.info("crosscal: No 'flag_after' found. Save current flagging state.")

    print()
    print('Clearing calibrations')
    clearcal(vis=calms)
    print()

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
        refant = ref_ant, gaintype = "G", calmode = "a",uvrange='', refantmode='strict',\
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

    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_crosscal_XXYY.png')
    os.system(f'shadems {calms} -x FREQ -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_crosscal_XYYX.png')

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
        refant = ref_ant, gaintype = "G", calmode = "a",uvrange='',\
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

    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_crosscal_XXYY_flagged.png')
    os.system(f'shadems {calms} -x FREQ -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_crosscal_XYYX_flagged.png')

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

    os.system(f'ragavi-gains --table {ptab_df} --field 2 -o {path}/PLOTS/{obs}_Dfcal_flagged -p {path}/PLOTS/{obs}_Dfcal_flagged.png')

    # Apply Df to bpcal
    applycal(vis=calms,field=fcal,gaintable=[ktab2,gtab_p2,gtab_a2,btab2,ptab_df],parang=False, flagbackup=False)

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

    os.system(f'ragavi-gains --table {ptab_df2} --field 2 -o {path}/PLOTS/{obs}_Dfcal_preflag2 -p {path}/PLOTS/{obs}_Dfcal_preflag2.png')
    
    flagdata(vis=ptab_df2, mode='tfcrop',datacolumn="CPARAM", quackinterval=0.0,ntime="60s",combinescans=True,timecutoff=5.0, freqcutoff=3.0, usewindowstats="both", flagbackup=False)
    df_flagversions2 = flagmanager(vis=ptab_df2, mode='list')
    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_df2' for entry in df_flagversions2.values()):
        flagmanager(vis=ptab_df2, mode='delete', versionname=obs+'_flag_df2', merge='replace')
        logger.info("corsscal: Found 'flag_df2'. Deleting it.")
    flagmanager(vis=ptab_df2, mode='save', versionname=obs+'_flag_df2', merge='replace')

    os.system(f'ragavi-gains --table {ptab_df2} --field 2 -o {path}/PLOTS/{obs}_Dfcal_flagged2 -p {path}/PLOTS/{obs}_Dfcal_flagged2.png')
     
    applycal(vis=calms,field=gcal,gaintable=[ktab2,gtab_p2,gtab_a2,btab2,ptab_df2],parang=False, flagbackup=False)

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
    os.system(f'shadems {calms} -x FREQ -y DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_Df-DATA.png')
    os.system(f'shadems {calms} -x FREQ -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/PLOTS --png {obs}_{bpcal}_Df-CORRECTED.png')
    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_crosscal_XXYY.png')

    logger.info("")
    logger.info("crosscal: Finished calibration of the primary calibrator")
    log.append_to_google_doc(' CROSSCAL', 'Finished calibration of the primary', warnings="", plot_link="")


    #####################################################################################################
    # Crosscalibration of the secondary calibrator
    logger.info("")
    logger.info('crosscal: Starting crosscalibration of the secondary calibrator')

    #  Calibrate Secondary -p  and T - amplitude normalized to 1
    gaincal(vis = calms, caltable = ktab_sec, selectdata = True,\
        solint = "inf", field = gcal, combine = "",uvrange='',\
        refant = ref_ant, solnorm = False, gaintype = "K",\
        minsnr=3,parang = False, gaintable=[gtab_a2, btab2, ptab_df2])

    gaincal(vis = calms, caltable = gtab_sec_p, selectdata = True,\
        solint = "inf", field = gcal, combine = "",\
        refant = ref_ant, gaintype = "G", calmode = "p",uvrange='',refantmode='strict',\
        gaintable = [ktab_sec, gtab_a2,btab2,ptab_df2])

    gaincal(vis = calms, caltable = Ttab_sec, selectdata = True,\
        solint = "inf", field = gcal, combine = "",\
        refant = ref_ant, gaintype = "T", calmode = "ap",uvrange='',refantmode='strict',\
        solnorm=True, gaintable = [ktab_sec, gtab_sec_p,gtab_a2,btab2,ptab_df2], append=False)


    # Check calibration of secondary
    applycal(vis=calms, field=gcal, gaintable=[ktab_sec, gtab_sec_p,gtab_a2,btab2,Ttab_sec,ptab_df2], parang=False, flagbackup=False)
    os.system(f'ragavi-vis --ms {calms} -x frequency -y amplitude -dc CORRECTED tbin 12000 -ca antenna1 --corr XY,YX --field {gcal} -o {path}/PLOTS/{obs}_{gcal}-Df-CORRECTED.png')
    os.system(f'shadems {calms} -x UV -y CORRECTED_DATA:amp -c ANTENNA1 --corr XX,YY --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_amp_XXYY.png')
    os.system(f'shadems {calms} -x UV -y CORRECTED_DATA:phase -c ANTENNA1 --corr XX,YY --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_phase_XXYY.png')
    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY  --field {gcal} --dir {path}/PLOTS --png {obs}_{gcal}_phaserefine_XXYY.png')


    #apply calibration up to  now to xcal: XY and YX will vary with time due to pang
    applycal(vis=calms,field=xcal,gaintable=[ktab,gtab_p2,gtab_a2,btab2,ptab_df2],parang=False, flagbackup=False)

    flagdata(vis=calms, mode="rflag", field=xcal, datacolumn="corrected", quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False)
    flagdata(vis=calms, mode='extend', field=xcal, datacolumn='corrected', growtime=80, growfreq=80, flagbackup=False, growaround=True, flagnearfreq=True)
    if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_before_xf' for entry in flag_versions.values()):
        flagmanager(vis=calms, mode='delete', versionname=obs+'_flag_before_xf', merge='replace')
        logger.info("crosscal: Found 'flag_before_xf'. Deleting it.")
    flagmanager(vis=calms, mode='save', versionname=obs+'_flag_before_xf', merge='replace')


    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XXYY.png')
    os.system(f'shadems {calms} -x TIME -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XYYX.png')

    logger.info("")
    logger.info("crosscal: Finished calibration of the secondary calibrator")
    log.append_to_google_doc(' CROSSCAL', 'Finished calibration of the secondary', warnings="", plot_link="")

    #############################################################################################################
    #calibration of the polarisation calibrator
    logger.info("")
    logger.info('crosscal: Starting crosscalibration of the polarisation calibrator')

    #  Calibrate XY phase: calibrate P on 3C286 - refine the phase
    gaincal(vis = calms, caltable = ktab_pol, selectdata = True,\
        solint = "inf", field = xcal, combine = "",uvrange='',\
        refant = ref_ant, solnorm = False, gaintype = "K",\
        minsnr=3,parang = False, gaintable=[gtab_a2, btab2, ptab_df2])

    gaincal(vis = calms, caltable = gtab_pol_p, selectdata = True,\
        solint = 'inf', field = xcal, combine = "",scan='',\
        refant = ref_ant, gaintype = "G", calmode = "p",uvrange='',refantmode='strict',\
        gaintable = [ktab_pol, gtab_a2,btab2,ptab_df2], parang = False)

    #selfcal on polarisation calibrator
    tclean(vis=calms,field=xcal,cell='0.5arcsec',imsize=512,niter=1000,imagename=path+'/output/pol_selfcal/'+obs+'_'+xcal+'-selfcal',weighting='briggs',robust=-0.2,datacolumn= 'corrected',deconvolver= 'mtmfs',\
        nterms=2,specmode='mfs',interactive=False)
    gaincal(vis=calms,field=xcal, calmode='p', solint='30s',caltable=gtab_pol_p+'-selfcal',refantmode='strict',\
        refant=ref_ant,gaintype='G',gaintable = [ktab_pol, gtab_a2,btab2,ptab_df2], parang = False)
    gtab_pol_p=gtab_pol_p+"-selfcal"


    gaincal(vis = calms, caltable = Ttab_pol, selectdata = True,\
        solint = "inf", field = xcal, combine = "",\
        refant = ref_ant, gaintype = "T", calmode = "ap",uvrange='',\
        solnorm=False, gaintable = [ktab_pol, gtab_pol_p,gtab_a2,btab2,ptab_df2], append=False, parang=False)

    #apply calibration up to  now, including phase refinement to xcal - crosshands should be real vaue dominated, imaginary will give idea of induced elliptcity. change in real axis due to parang

    applycal(vis=calms,field=xcal,gaintable=[ktab_pol, gtab_pol_p, gtab_a2, btab2, Ttab_sec, ptab_df2],parang=False, flagbackup=False)
    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XXYY_refinephase.png')
    os.system(f'shadems {calms} -x CORRECTED_DATA:imag -y CORRECTED_DATA:real -c CORR --corr XY,YX --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_precalXf_XYYX_real_im.png')


    # Cross-hand delay calibration - 
    gaincal(vis = calms, caltable = kxtab, selectdata = True,\
                solint = "inf", field = xcal, combine = "scan", scan='',\
                refant = ref_ant, gaintype = "KCROSS",\
                gaintable = [ktab_pol, gtab_pol_p, gtab_a2, btab2, ptab_df2],\
                smodel=[15.7433,0.8628247336,1.248991241,0],\
                parang = True)


    # Calibrate XY phase
    polcal(vis = calms, caltable = ptab_xf, selectdata = True,scan='',combine='scan',\
        solint = "1200s,20MHz", field = xcal, uvrange='',\
        refant = ref_ant, poltype = "Xf",  gaintable = [ktab_pol, gtab_pol_p, kxtab,gtab_a2,btab2,ptab_df2], smodel= [15.7433,0.8628247336,1.248991241,0])

    
    os.system(f'ragavi-gains --table {ptab_xf} --field 1 -o {path}/PLOTS/{obs}_Xfcal -p {path}/PLOTS/{obs}_Xfcal.png')
    
    logger.info("")
    logger.info('crosscal: Correcting for phase ambiguity')
    #exec(open('/localwork/angelina/meerkat_virgo/ViMS/ViMS/scripts/xyamb_corr.py').read())
    S=xyamb_corr.xyamb(logger, xytab=ptab_xf ,xyout=ptab_xfcorr)

    os.system(f'ragavi-gains --table {ptab_xfcorr} --field 1 -o {path}/PLOTS/{obs}_Xfcal_ambcorr -p {path}/PLOTS/{obs}_Xfcal_ambcorr.png')

    logger.info("")
    logger.info("crosscal: Finished calibration of the polarisation calibrator")
    log.append_to_google_doc(' CROSSCAL', 'Finished calibration of the polcal', warnings="", plot_link="")

    applycal(vis=calms, field=xcal,gaintable=[ktab_pol, gtab_pol_p, kxtab, ptab_xfcorr, gtab_a2, btab2, Ttab_pol, ptab_df2],parang=True, flagbackup=False)
    applycal(vis=calms, field=gcal,gaintable=[ktab_sec, gtab_sec_p, kxtab, ptab_xfcorr, gtab_a2, btab2, Ttab_sec, ptab_df2], parang=True, flagbackup=False)
    applycal(vis=calms, field=fcal,gaintable=[ktab2, gtab_p2, kxtab, ptab_xfcorr, gtab_a2, btab2, ptab_df2], parang=True, flagbackup=False)

    final_flag = flagdata(vis=calms, mode='summary')
    logger.info("")
    logger.info('crosscal: Flagging summary after cross and pol calibration:')
    flag.log_flagsum(final_flag, logger)

    # Check: plot imaginary versis real and compare to previous plot

    os.system(f'shadems {calms} -x CORRECTED_DATA:imag -y CORRECTED_DATA:real -c CORR --corr XY,YX --field {xcal} --dir {path}/PLOTS --png {obs}_{xcal}_aftercalXf_XYYX_real_im.png')
    os.system(f'ragavi-vis --ms {calms} -x frequency -y phase -dc CORRECTED tbin 12000 -ca antenna1 --corr XY,YX --field {xcal} -o {path}/PLOTS/{obs}_{xcal}_aftercalXf_XYYX_phase.png')

#------------------------------------------------------

def run(logger, obs_id, cal_ms, path):
    #cal_ms = "/a.benati/lw/victoria/tests/flag/obs01_1662797070_sdp_l0-cal_copy.ms"
    log.append_to_google_doc("######################################################", "", warnings="", plot_link="")
    log.append_to_google_doc("######################## CROSSCAL ########################", "", warnings="", plot_link="")
    log.append_to_google_doc("######################################################", "", warnings="", plot_link="")
    log.append_to_google_doc("CROSSCAL", "Started", warnings="", plot_link="")
    
    # swap feeds
    logger.info("\n\n\n\n\n")
    logger.info("CROSSCAL: starting Feedswap...")
    correct_parang(logger, cal_ms, [0, 1, 2])
    logger.info('CROSSCAL: finished Feedswap')
    logger.info("")
    logger.info("")
    logger.info("")
    log.append_to_google_doc('CROSSCAL', 'Finished Feedswap', warnings="", plot_link="")
    
    # do cross- and polarisation calibration
    logger.info("\n\n\n\n\n")
    logger.info("CROSSCAL: starting crosscalibration...")
    crosscal(logger, obs_id, cal_ms, path)
    logger.info('CROSSCAL: finished crosscalibration')
    logger.info("")
    logger.info("")
    logger.info("")
    log.append_to_google_doc('CROSSCAL', 'Finished crosscalibration', warnings="", plot_link="")
    