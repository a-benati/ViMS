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


def run(logger, cal_ms, path, fields=[0,1,2], filename="feedswap_cal.txt"):
    try:
        if os.path.exists(f"{path}/LOGS/{filename}"):
            logger.info("FEEDSWAP: Feedswap already done, skipping...")
        
        else:
            logger.info("\n\n\n\n\n")
            logger.info("FEEDSWAP: starting Feedswap...")
            correct_parang(logger, cal_ms, fields)
            logger.info('FEEDSWAP: finished Feedswap')
            with open(f"{path}/LOGS/{filename}", "w") as file:
                file.write(f"Feedswap for the ms file {cal_ms} was done!")
            logger.info("")
            logger.info("")
            logger.info("")
    except Exception as e:
        logger.exception("FEEDSWAP: Feedswap failed")