#!/usr/bin/env python3

import sys,os
from utils.utils import run_command
from utils.log import setup_logging
from casatasks import *

# Set up the logging for this script and CASA
log_filename = setup_logging()

# Get the logger for your script
logger = logging.getLogger('flag.py')

cal_ms = "/data/victoria/tests/flag/obs01_1662797070_sdp_l0-cal_copy.ms"

def run_flagging_task():
    logger.info("Running CASA task for flagging...")

    try:
        # Example of running a CASA task
        listobs(vis=cal_ms, listfile='listobs.txt', overwrite=True)

        # FLAGMANAGER FOR SAVING FLAGS
        flagmanager(vis=cal_ms,\
                        mode="save",versionname="obs01_flag_before",oldname="",comment="",\
                        merge="replace")

        flagdata(vis=cal_ms, \
                    mode="summary", spw="0")

        # FLAG AUTOCORRELATIONS
        flagdata(vis=cal_ms,\
                    mode="manual",autocorr=True,inpfile="",reason="any",tbuff=0.0,\
                    field="J1331+3030,J1939-6342,J1150-0023",antenna="",uvrange="",timerange="",\
                    correlation="",scan="",intent="",array="",observation="",feed="",clipminmax=[],\
                    datacolumn="DATA",clipoutside=True,channelavg=False,chanbin=1,timeavg=False,timebin="0s",\
                    clipzeros=False,quackinterval=0.0,quackmode="beg",quackincrement=False,tolerance=0.0,\
                    addantenna="",lowerlimit=0.0,upperlimit=90.0,ntime="scan",combinescans=False,timecutoff=4.0,\
                    freqcutoff=3.0,timefit="line",freqfit="poly",maxnpieces=7,flagdimension="freqtime",\
                    usewindowstats="none",halfwin=1,extendflags=True,winsize=3,timedev="",freqdev="",\
                    timedevscale=5.0,freqdevscale=5.0,spectralmax=1000000.0,spectralmin=0.0,antint_ref_antenna="",\
                    minchanfrac=0.6,verbose=False,extendpols=True,growtime=50.0,growfreq=50.0,growaround=False,\
                    flagneartime=False,flagnearfreq=False,minrel=0.0,maxrel=1.0,minabs=0,maxabs=-1,spwchan=False,\
                    spwcorr=False,basecnt=False,fieldcnt=False,name="Summary",action="apply",display="",\
                    flagbackup=False,savepars=False,cmdreason="",outfile="",overwrite=True,writeflags=True)

        #FLAG SHADOWED ANTENNAS
        flagdata(vis=cal_ms,\
                    mode="shadow",autocorr=False,inpfile="",reason="any",tbuff=0.0,spw="",\
                    field="J1331+3030,J1939-6342,J1150-0023",antenna="",uvrange="",timerange="",\
                    correlation="",scan="",intent="",array="",observation="",feed="",clipminmax=[],\
                    datacolumn="DATA",clipoutside=True,channelavg=False,chanbin=1,timeavg=False,timebin="0s",\
                    clipzeros=False,quackinterval=0.0,quackmode="beg",quackincrement=False,tolerance=0.0,\
                    addantenna="",lowerlimit=0.0,upperlimit=90.0,ntime="scan",combinescans=False,timecutoff=4.0,\
                    freqcutoff=3.0,timefit="line",freqfit="poly",maxnpieces=7,flagdimension="freqtime",\
                    usewindowstats="none",halfwin=1,extendflags=True,winsize=3,timedev="",freqdev="",\
                    timedevscale=5.0,freqdevscale=5.0,spectralmax=1000000.0,spectralmin=0.0,antint_ref_antenna="",\
                    minchanfrac=0.6,verbose=False,extendpols=True,growtime=50.0,growfreq=50.0,growaround=False,\
                    flagneartime=False,flagnearfreq=False,minrel=0.0,maxrel=1.0,minabs=0,maxabs=-1,spwchan=False,\
                    spwcorr=False,basecnt=False,fieldcnt=False,name="Summary",action="apply",display="",\
                    flagbackup=False,savepars=False,cmdreason="",outfile="",overwrite=True,writeflags=True)

        #FLAG BAD CHANNELS
        flagdata(vis=cal_ms,\
                    mode="manual",autocorr=False,inpfile="",reason="any",tbuff=0.0,\
                    spw="*:856~880MHz ,*:1658~1800MHz,*:1419.8~1421.3MHz",\
                    field="J1331+3030,J1939-6342,J1150-0023",antenna="",uvrange="",timerange="",\
                    correlation="",scan="",intent="",array="",observation="",feed="",clipminmax=[],\
                    datacolumn="DATA",clipoutside=True,channelavg=False,chanbin=1,timeavg=False,timebin="0s",\
                    clipzeros=False,quackinterval=0.0,quackmode="beg",quackincrement=False,tolerance=0.0,\
                    addantenna="",lowerlimit=0.0,upperlimit=90.0,ntime="scan",combinescans=False,timecutoff=4.0,\
                    freqcutoff=3.0,timefit="line",freqfit="poly",maxnpieces=7,flagdimension="freqtime",\
                    usewindowstats="none",halfwin=1,extendflags=True,winsize=3,timedev="",freqdev="",timedevscale=5.0,\
                    freqdevscale=5.0,spectralmax=1000000.0,spectralmin=0.0,antint_ref_antenna="",minchanfrac=0.6,\
                    verbose=False,extendpols=True,growtime=50.0,growfreq=50.0,growaround=False,flagneartime=False,\
                    flagnearfreq=False,minrel=0.0,maxrel=1.0,minabs=0,maxabs=-1,spwchan=False,spwcorr=False,\
                    basecnt=False,fieldcnt=False,name="Summary",action="apply",display="",flagbackup=False,\
                    savepars=False,cmdreason="",outfile="",overwrite=True,writeflags=True)

        #APPLY FLAG MASK (RFI MASKER)
        stdout, stderr = run_command(f"mask_ms.py --mask /stimela_mount/input/meerkat.rfimask.npy \
                                        --accumulation_mode or --memory 4096 --uvrange 0~1000\
                                        {cal_ms}")

        #AOFLAGGER AUTO-FLAGGING
        stdout, stderr = run_command(f"aoflagger -strategy firstpass_QUV.rfis\
                                        {cal_ms}")

        #FLAG SUMMARY
        flagdata(vis='/data/victoria/tests/flag/obs01_1662797070_sdp_l0-cal_copy.ms', mode='summary')

        run_command('rm casa*.log')
        logger.info("Flagging task completed successfully.")
    except Exception as e:
        logger.error(f"Error in flagging task: {str(e)}")

# Example usage of the function
run_flagging_task()

# listobs(vis=cal_ms, listfile='listobs.txt', overwrite=True)

# # FLAGMANAGER FOR SAVING FLAGS
# flagmanager(vis=cal_ms,\
#                 mode="save",versionname="obs01_flag_before",oldname="",comment="",\
#                 merge="replace")

# flagdata(vis=cal_ms, \
#             mode="summary", spw="0")

# # FLAG AUTOCORRELATIONS
# flagdata(vis=cal_ms,\
#             mode="manual",autocorr=True,inpfile="",reason="any",tbuff=0.0,\
#             field="J1331+3030,J1939-6342,J1150-0023",antenna="",uvrange="",timerange="",\
#             correlation="",scan="",intent="",array="",observation="",feed="",clipminmax=[],\
#             datacolumn="DATA",clipoutside=True,channelavg=False,chanbin=1,timeavg=False,timebin="0s",\
#             clipzeros=False,quackinterval=0.0,quackmode="beg",quackincrement=False,tolerance=0.0,\
#             addantenna="",lowerlimit=0.0,upperlimit=90.0,ntime="scan",combinescans=False,timecutoff=4.0,\
#             freqcutoff=3.0,timefit="line",freqfit="poly",maxnpieces=7,flagdimension="freqtime",\
#             usewindowstats="none",halfwin=1,extendflags=True,winsize=3,timedev="",freqdev="",\
#             timedevscale=5.0,freqdevscale=5.0,spectralmax=1000000.0,spectralmin=0.0,antint_ref_antenna="",\
#             minchanfrac=0.6,verbose=False,extendpols=True,growtime=50.0,growfreq=50.0,growaround=False,\
#             flagneartime=False,flagnearfreq=False,minrel=0.0,maxrel=1.0,minabs=0,maxabs=-1,spwchan=False,\
#             spwcorr=False,basecnt=False,fieldcnt=False,name="Summary",action="apply",display="",\
#             flagbackup=False,savepars=False,cmdreason="",outfile="",overwrite=True,writeflags=True)

# #FLAG SHADOWED ANTENNAS
# flagdata(vis=cal_ms,\
#             mode="shadow",autocorr=False,inpfile="",reason="any",tbuff=0.0,spw="",\
#             field="J1331+3030,J1939-6342,J1150-0023",antenna="",uvrange="",timerange="",\
#             correlation="",scan="",intent="",array="",observation="",feed="",clipminmax=[],\
#             datacolumn="DATA",clipoutside=True,channelavg=False,chanbin=1,timeavg=False,timebin="0s",\
#             clipzeros=False,quackinterval=0.0,quackmode="beg",quackincrement=False,tolerance=0.0,\
#             addantenna="",lowerlimit=0.0,upperlimit=90.0,ntime="scan",combinescans=False,timecutoff=4.0,\
#             freqcutoff=3.0,timefit="line",freqfit="poly",maxnpieces=7,flagdimension="freqtime",\
#             usewindowstats="none",halfwin=1,extendflags=True,winsize=3,timedev="",freqdev="",\
#             timedevscale=5.0,freqdevscale=5.0,spectralmax=1000000.0,spectralmin=0.0,antint_ref_antenna="",\
#             minchanfrac=0.6,verbose=False,extendpols=True,growtime=50.0,growfreq=50.0,growaround=False,\
#             flagneartime=False,flagnearfreq=False,minrel=0.0,maxrel=1.0,minabs=0,maxabs=-1,spwchan=False,\
#             spwcorr=False,basecnt=False,fieldcnt=False,name="Summary",action="apply",display="",\
#             flagbackup=False,savepars=False,cmdreason="",outfile="",overwrite=True,writeflags=True)

# #FLAG BAD CHANNELS
# flagdata(vis=cal_ms,\
#             mode="manual",autocorr=False,inpfile="",reason="any",tbuff=0.0,\
#             spw="*:856~880MHz ,*:1658~1800MHz,*:1419.8~1421.3MHz",\
#             field="J1331+3030,J1939-6342,J1150-0023",antenna="",uvrange="",timerange="",\
#             correlation="",scan="",intent="",array="",observation="",feed="",clipminmax=[],\
#             datacolumn="DATA",clipoutside=True,channelavg=False,chanbin=1,timeavg=False,timebin="0s",\
#             clipzeros=False,quackinterval=0.0,quackmode="beg",quackincrement=False,tolerance=0.0,\
#             addantenna="",lowerlimit=0.0,upperlimit=90.0,ntime="scan",combinescans=False,timecutoff=4.0,\
#             freqcutoff=3.0,timefit="line",freqfit="poly",maxnpieces=7,flagdimension="freqtime",\
#             usewindowstats="none",halfwin=1,extendflags=True,winsize=3,timedev="",freqdev="",timedevscale=5.0,\
#             freqdevscale=5.0,spectralmax=1000000.0,spectralmin=0.0,antint_ref_antenna="",minchanfrac=0.6,\
#             verbose=False,extendpols=True,growtime=50.0,growfreq=50.0,growaround=False,flagneartime=False,\
#             flagnearfreq=False,minrel=0.0,maxrel=1.0,minabs=0,maxabs=-1,spwchan=False,spwcorr=False,\
#             basecnt=False,fieldcnt=False,name="Summary",action="apply",display="",flagbackup=False,\
#             savepars=False,cmdreason="",outfile="",overwrite=True,writeflags=True)

# #APPLY FLAG MASK (RFI MASKER)
# stdout, stderr = run_command(f"mask_ms.py --mask /stimela_mount/input/meerkat.rfimask.npy \
#                                 --accumulation_mode or --memory 4096 --uvrange 0~1000\
#                                 {cal_ms}")

# #AOFLAGGER AUTO-FLAGGING
# stdout, stderr = run_command(f"aoflagger -strategy firstpass_QUV.rfis\
#                                 {cal_ms}")

# #FLAG SUMMARY
# flagdata(vis='/data/victoria/tests/flag/obs01_1662797070_sdp_l0-cal_copy.ms', mode='summary')

# run_command('rm casa*.log')