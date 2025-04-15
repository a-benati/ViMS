#!/usr/bin/env python3

import sys, os
from utils import utils, log
from casatasks import *

cal_ms = "/a.benati/lw/victoria/tests/flag/obs01_1662797070_sdp_l0-cal_copy.ms"

def log_flagsum(summary, logger):
    """
    Prints a readable version of the flagging summary given.
    
    Parameters:
        summary: name of the summary dictionary created by flagdata
        logger: logger instance of the pipeline
    """
    
    # 1. Total flagged summary
    if 'flagged' in summary and 'total' in summary:
        flagged = summary['flagged']
        total = summary['total']
        perc = (flagged / total) * 100 if total > 0 else 0
        logger.info("Total flagged: {}/{} ({:.2f}%)".format(flagged, total, perc))

    # 2. Per correlation
    if 'correlation' in summary:
        logger.info("Flags per Correlation:")
        for corr, stats in summary['correlation'].items():
            flagged = stats['flagged']
            total = stats['total']
            perc = (flagged / total) * 100 if total > 0 else 0
            logger.info("   Correlation {}: {}/{} flagged ({:.2f}%)".format(corr, flagged, total, perc))
        logger.info("")

    # 3. Per field
    if 'field' in summary:
        logger.info("Flags per Field:")
        for field, stats in summary['field'].items():
            flagged = stats['flagged']
            total = stats['total']
            perc = (flagged / total) * 100 if total > 0 else 0
            logger.info("   Field {}: {}/{} flagged ({:.2f}%)".format(field, flagged, total, perc))

def append_flagsum_google_doc(summary):
    """
    Prints a readable version of the flagging summary given.
    
    Parameters:
        summary: name of the summary dictionary created by flagdata
        logger: logger instance of the pipeline
    """
    log.append_to_google_doc("FLAGGING SUMMARY", "", warnings="", plot_link="")
    # 1. Total flagged summary
    if 'flagged' in summary and 'total' in summary:
        flagged = summary['flagged']
        total = summary['total']
        perc = (flagged / total) * 100 if total > 0 else 0
        log.append_to_google_doc("", "Total flagged: {}/{} ({:.2f}%)".format(flagged, total, perc), \
                                 warnings="", plot_link="")

    # 2. Per correlation
    if 'correlation' in summary:
        log.append_to_google_doc("", "Flags per Correlation:", \
                                 warnings="", plot_link="")
        for corr, stats in summary['correlation'].items():
            flagged = stats['flagged']
            total = stats['total']
            perc = (flagged / total) * 100 if total > 0 else 0
            log.append_to_google_doc("", "   Correlation {}: {}/{} flagged ({:.2f}%)".format(corr, flagged, total, perc), \
                                 warnings="", plot_link="")
        log.append_to_google_doc("", "", warnings="", plot_link="")

    # 3. Per field
    if 'field' in summary:
        log.append_to_google_doc("", "Flags per Field:", \
                                 warnings="", plot_link="")
        for field, stats in summary['field'].items():
            flagged = stats['flagged']
            total = stats['total']
            perc = (flagged / total) * 100 if total > 0 else 0
            log.append_to_google_doc("", "   Field {}: {}/{} flagged ({:.2f}%)".format(field, flagged, total, perc), \
                                 warnings="", plot_link="")
        log.append_to_google_doc("", "", warnings="", plot_link="")

def run(logger, obs_id):
    logger.info("")
    logger.info("")
    logger.info("##########################################################")
    logger.info("########################## FLAG ##########################")
    logger.info("##########################################################")
    logger.info("")
    logger.info("")

    try:
        log.append_to_google_doc("######################################################", "", warnings="", plot_link="")
        log.append_to_google_doc("######################## FLAG ########################", "", warnings="", plot_link="")
        log.append_to_google_doc("######################################################", "", warnings="", plot_link="")
        log.append_to_google_doc("FLAG", "Started", warnings="", plot_link="")
        # If there is a plot generated
        # plot_url = upload_plot_to_drive("path/to/plot.png")

        # FLAGMANAGER FOR SAVING FLAGS
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Saving flags...")
        # flagmanager(vis=cal_ms,\
        #                     mode="save",versionname="obs01_flag_before",oldname="",comment="",\
        #                     merge="replace")
        log.redirect_casa_log(logger)
        logger.info("FLAG: Saved flags\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        log.append_to_google_doc("FLAG", "Saved flags", warnings="", plot_link="")

        # RESTORE FLAGS
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Restoring flags...")
        # flagdata(vis=cal_ms,\
        #             mode="unflag",autocorr=True,inpfile="",reason="any",tbuff=0.0,\
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
        log.redirect_casa_log(logger)
        logger.info("FLAG: Restored flags\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        log.append_to_google_doc("FLAG", "Restored flags", warnings="", plot_link="")

        # PLOT MS WITHOUT FLAGS
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Plotting MS file without flags...")
        logger.info("-------------------------------------------------------------------------------------")
        plot_path = f"./OUTPUT/{obs_id}/PLOTS/"
        plot_name = f"{obs_id}{{_field}}_before_flags.png"
        stdout, stderr = utils.run_command(f"shadems -x FREQ -y amp --iter-field --dir {plot_path} \
                                           --png {plot_name} {cal_ms}")
        plot_link = log.upload_plot_to_drive(plot_path, plot_name)
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in ShadeMS: {stderr}")
        logger.info("-------------------------------------------------------------------------------------")
        logger.info("FLAG: Plotted MS file without flags\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        logger.info("\n\n\n\n\n")
        log.append_to_google_doc("FLAG", "Plotted MS file without flags", warnings="", plot_link=plot_link)

        # FLAG AUTOCORRELATIONS
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Flagging autocorrelations...")
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
        log.redirect_casa_log(logger)
        logger.info("FLAG: Flagged autocorrelations\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        log.append_to_google_doc("FLAG", "Flagged autocorrelations", warnings="", plot_link="")

        #FLAG SHADOWED ANTENNAS
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Flagging shadowed antennas...")
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
        log.redirect_casa_log(logger)
        logger.info("FLAG: Flagged shadowed antennas\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        log.append_to_google_doc("FLAG", "Flagged shadowed antennas", warnings="", plot_link="")

        #FLAG BAD CHANNELS
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Flagging bad channels...")
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
        log.redirect_casa_log(logger)
        logger.info("FLAG: Flagged bad channels\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        log.append_to_google_doc("FLAG", "Flagged bad channels", warnings="", plot_link="")

        #APPLY FLAG MASK (RFI MASKER)
        # stdout, stderr = utils.run_command(f"mask_ms.py --mask /ViMS/ViMS/utils/meerkat.rfimask.npy \
        #                                 --accumulation_mode or --memory 4096 --uvrange 0~1000\
        #                                 {cal_ms}")

        #AOFLAGGER AUTO-FLAGGING
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Flagging with AOFlagger (Annalisa's strategy) 1st time...")
        logger.info("-------------------------------------------------------------------------------------")
        stdout, stderr = utils.run_command(f"aoflagger -strategy /ViMS/ViMS/utils/default_thr2.lua\
                                        {cal_ms}")
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in AOFlagger: {stderr}")
        logger.info("-------------------------------------------------------------------------------------")
        logger.info("FLAG: Flagged with AOFlagger (Annalisa's strategy) 1st time\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        logger.info("\n\n\n\n\n")
        log.append_to_google_doc("FLAG", "Flagged with AOFlagger (Annalisa's strategy) 1st time", warnings="", plot_link="")

        logger.info("FLAG: Flagging with AOFlagger (Annalisa's strategy) 2nd time...")
        logger.info("-------------------------------------------------------------------------------------")
        stdout, stderr = utils.run_command(f"aoflagger -strategy /ViMS/ViMS/utils/default_thr2.lua\
                                        {cal_ms}")
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in AOFlagger: {stderr}")
        logger.info("FLAG: Flagged with AOFlagger (Annalisa's strategy) 2nd time\n\n\n\n\n")
        logger.info("-------------------------------------------------------------------------------------")
        logger.info("")
        logger.info("")
        logger.info("")
        log.append_to_google_doc("FLAG", "Flagged with AOFlagger (Annalisa's strategy) 2nd time", warnings="", plot_link="")

        #FLAG SUMMARY
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Printing flagging summary...")
        summary = flagdata(vis=cal_ms, mode='summary')
        log.redirect_casa_log(logger)
        log_flagsum(summary, logger)
        append_flagsum_google_doc(summary)
        logger.info("")
        logger.info("")
        logger.info("")

        # PLOT MS WITH ALL THE FLAGS
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Plotting MS file with all the flags...")
        logger.info("-------------------------------------------------------------------------------------")
        plot_path = f"./OUTPUT/{obs_id}/PLOTS/"
        plot_name = f"{obs_id}{{_field}}_after_flags.png"
        stdout, stderr = utils.run_command(f"shadems -x FREQ -y amp --iter-field --dir {plot_path} \
                                           --png {plot_name} {cal_ms}")
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in ShadeMS: {stderr}")
        logger.info("-------------------------------------------------------------------------------------")
        logger.info("FLAG: Plotted MS file with all the flags\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        logger.info("\n\n\n\n\n")
        log.append_to_google_doc("FLAG", "Plotted MS file with all the flags", warnings="", plot_link="")

        logger.info("Flag step completed successfully!")
        logger.info("######################################################")
        logger.info("###################### END FLAG ######################")
        logger.info("######################################################")
        logger.info("")
        logger.info("")
        log.append_to_google_doc("Flag step completed successfully!", "", warnings="", plot_link="")
        log.append_to_google_doc("######################################################", "", warnings="", plot_link="")
        log.append_to_google_doc("###################### END FLAG ######################", "", warnings="", plot_link="")
        log.append_to_google_doc("######################################################", "", warnings="", plot_link="")
        log.append_to_google_doc("", "", warnings="", plot_link="")
        log.append_to_google_doc("", "", warnings="", plot_link="")
    except Exception as e:
        logger.error(f"Error in FLAG step: {str(e)}")