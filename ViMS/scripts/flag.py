#!/usr/bin/env python3

import os, glob, re
from utils import utils, log
from casatasks import *

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

def get_flag_perc(summary):
    if 'flagged' in summary and 'total' in summary:
        flagged = summary['flagged']
        total = summary['total']
        perc = (flagged / total) * 100 if total > 0 else 0
        return f"{perc:.1f}%"
    
def save_flags(logger, obs_id, ms, when='before'):
    """
    Saves the flags of the MS file before and after flagging.
    
    Parameters:
        logger: logger instance of the pipeline
        obs_id: observation ID
        ms: measurement set file
        when: when to save the flags, either 'before' or 'after' flagging
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Saving flags...")
        flagmanager(vis=ms,\
                            mode="save",versionname=f"{obs_id}_flag_{when}",oldname="",comment="",\
                            merge="replace")
        log.redirect_casa_log(logger)
        logger.info("FLAG: Saved flags\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
    except Exception as e:
        logger.exception("Error while saving flags")

def restore_flags(logger, ms):
    """
    Restores the flags of the MS file.
    
    Parameters:
        logger: logger instance of the pipeline
        ms: measurement set file
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Restoring flags...")
        flagdata(vis=ms,\
                    mode="unflag",autocorr=True,inpfile="",reason="any",tbuff=0.0,\
                    field="",antenna="",uvrange="",timerange="",\
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
        logger.info("FLAG: Restored flags\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
    except Exception:
        logger.exception("Error while restoring flags")

def plot_flags(logger, obs_id, ms, path, when='before'):
    """
    Plots the MS file before and after flagging.
    
    Parameters:
        logger: logger instance of the pipeline
        obs_id: observation ID
        ms: measurement set file
        path: path to the output directory
        when: when to plot the MS file, either 'before' or 'after' flagging
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info(f"FLAG: Plotting MS file {when} flags...")
        logger.info("-------------------------------------------------------------------------------------")
        plot_path = path + "/PLOTS/"
        plot_name = f"{obs_id}{{_field}}_{when}_flags.png"
        cmd = f'/opt/shadems-env/bin/shadems -x FREQ -y amp --iter-field --dir "{plot_path}" --png "{plot_name}" {ms}'
        stdout, stderr = utils.run_command(cmd, logger)
        plot_pattern = os.path.join(plot_path, f"{obs_id}*_{when}_flags.png")
        plot_files = glob.glob(plot_pattern)
        plot_links = []
        for plot in plot_files:
            plot_links.append(log.upload_plot_to_drive(plot))
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in ShadeMS: {stderr}")
        logger.info("-------------------------------------------------------------------------------------")
        logger.info(f"FLAG: Plotted MS file {when} flags\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        logger.info("\n\n\n\n\n")
    except Exception as e:
        logger.exception(f"Error while plotting MS file {when} flags")
    
def flag_autocorrelations(logger, ms):
    """
    Function to flag autocorrelations on the measurement set.
    
    Parameters:
        logger: logger instance of the pipeline
        ms: measurement set file to be flagged
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Flagging autocorrelations...")
        flagdata(vis=ms,\
                    mode="manual",autocorr=True,inpfile="",reason="any",tbuff=0.0,\
                    field="",antenna="",uvrange="",timerange="",\
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
    except Exception as e:
        logger.exception("Error while flagging autocorrelations")

def flag_shadowed_antenna(logger, ms):
    """
    Function to flag shadowed antennas on the measurement set.
    
    Parameters:
        logger: logger instance of the pipeline
        ms: measurement set file to be flagged
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Flagging shadowed antennas...")
        flagdata(vis=ms,\
                    mode="shadow",autocorr=False,inpfile="",reason="any",tbuff=0.0,spw="",\
                    field="",antenna="",uvrange="",timerange="",\
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
    except Exception as e:
        logger.exception("Error while flagging shadowed antennas")

def flag_bad_channels(logger, ms):
    """
    Function to flag bad channels on the measurement set.
    
    Parameters:
        logger: logger instance of the pipeline
        ms: measurement set file to be flagged
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Flagging bad channels...")
        flagdata(vis=ms,\
                    mode="manual",autocorr=False,inpfile="",reason="any",tbuff=0.0,\
                    spw="*:856~880MHz ,*:1658~1800MHz,*:1419.8~1421.3MHz",\
                    field="",antenna="",uvrange="",timerange="",\
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
    except Exception as e:
        logger.exception("Error while flagging bad channels")

def flag_rfi_mask(logger, ms): # NB NOT IMPLEMENTED!
    """
    Function to apply RFI mask on the measurement set.
    
    Parameters:
        logger: logger instance of the pipeline
        ms: measurement set file to be flagged
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Applying RFI mask...")
        cmd = f"mask_ms.py --mask /ViMS/ViMS/utils/meerkat.rfimask.npy \
                --accumulation_mode or --memory 4096 --uvrange 0~1000 {ms}"
        stdout, stderr = utils.run_command(cmd, logger)
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in RFI mask: {stderr}")
        logger.info("FLAG: Applied RFI mask\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
    except Exception as e:
        logger.exception("Error while applying RFI mask")

def flag_aoflagger(logger, ms):
    """
    Function to flag the measurement set using AOFlagger.
    
    Parameters:
        logger: logger instance of the pipeline
        obs_id: observation ID
        ms: calibrated measurement set file
        path: path to the output directory
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Flagging with AOFlagger...")
        logger.info("-------------------------------------------------------------------------------------")
        cmd = f"aoflagger -strategy /ViMS/ViMS/utils/default_thr2.lua {ms}"
        stdout, stderr = utils.run_command(cmd, logger)
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in AOFlagger: {stderr}")
        logger.info("-------------------------------------------------------------------------------------")
        logger.info("FLAG: Flagged with AOFlagger\n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        logger.info("\n\n\n\n\n")
    except Exception as e:
        logger.exception("Error while flagging with AOFlagger")

def flag_tricolour(logger, ms, target):
    """
    Function to flag the measurement set using TriColour.
    
    Parameters:
        logger: logger instance of the pipeline
        ms: measurement set file
        target: target field to flag
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Flagging with Tricolour...")
        logger.info("-------------------------------------------------------------------------------------")
        cmd = f"tricolour --config /ViMS/ViMS/utils/mk_rfi_flagging_target_fields_firstpass.yaml\
                --flagging-strategy polarisation --data-column DATA --field-names {target}\
                --window-backend numpy {ms}"
        stdout, stderr = utils.run_command(cmd, logger)
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in Tricolour: {stderr}")
        logger.info("-------------------------------------------------------------------------------------")
        logger.info("FLAG: Flagged with Tricolour \n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        logger.info("\n\n\n\n\n")
    except Exception as e:
        logger.exception("Error while flagging with Tricolour")

def flag_tricolour_cal(logger, ms):
    """
    Function to flag the measurement set using TriColour.
    
    Parameters:
        logger: logger instance of the pipeline
        ms: measurement set file
        target: target field to flag
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Flagging with Tricolour...")
        logger.info("-------------------------------------------------------------------------------------")
        cmd = f"tricolour --config /angelina/meerkat_virgo/ViMS/ViMS/utils/mk_rfi_flagging_calibrator_fields_firstpass.yaml\
                --flagging-strategy total_power --data-column DATA --field-names 0,1,2\
                --window-backend numpy {ms}"
        stdout, stderr = utils.run_command(cmd, logger)
        logger.info(stdout)
        if stderr:
            logger.error(f"Error in Tricolour: {stderr}")
        logger.info("-------------------------------------------------------------------------------------")
        logger.info("FLAG: Flagged with Tricolour \n\n\n\n\n")
        logger.info("")
        logger.info("")
        logger.info("")
        logger.info("\n\n\n\n\n")
    except Exception as e:
        logger.exception("Error while flagging with Tricolour")

def flag_summary(logger, ms, obs_id):
    """
    Function to print a summary of the flags applied to the measurement set.
    
    Parameters:
        logger: logger instance of the pipeline
        ms: measurement set file
        obs_id: observation ID
    """
    try:
        logger.info("\n\n\n\n\n")
        logger.info("FLAG: Printing flagging summary...")
        summary = flagdata(vis=ms, mode='summary')
        log.redirect_casa_log(logger)
        log_flagsum(summary, logger)
        log.update_cell_in_google_doc(obs_id, 'Flag Perc', get_flag_perc(summary))
        logger.info("")
        logger.info("")
        logger.info("")
    except Exception as e:
        logger.exception("Error while printing flagging summary")

def run(logger, obs_id, ms, path, toflag='cal'):
    """
    Function to run the flagging step of the ViMS pipeline.

    Parameters:
        logger: logger instance of the pipeline
        obs_id: observation ID
        ms: measurement set file
        path: path to the output directory
        targets: list of target fields to flag
        toflag: type of flagging to perform, either 'cal' or 'target'
    """
    logger.info("")
    logger.info("")
    logger.info("##########################################################")
    logger.info("########################## FLAG ##########################")
    logger.info("##########################################################")
    logger.info("")
    logger.info("")

    # log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("######################## FLAG ########################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")

    save_flags(logger, obs_id, ms, 'before')
    restore_flags(logger, ms)
    # plot_flags(logger, obs_id, ms, path, 'before')
    flag_autocorrelations(logger, ms)
    flag_shadowed_antenna(logger, ms)
    flag_bad_channels(logger, ms)

    if toflag == 'cal':
        if obs_id =='obs01' or obs_id == 'obs02':
            logger.info('Using stricter flagging for daytime observations')
            flag_tricolour_cal(logger, ms)

        else:
            logger.info('Using standard flagging for nighttime observations')
            flag_aoflagger(logger, ms) # 1st time
            flag_aoflagger(logger, ms) # 2nd time
    
    elif toflag == 'target':
        match = re.search(r'sdp_l0-([^.]+)\.ms$', ms)
        target = match.group(1)
        flag_tricolour(logger, ms, target)

    else:
        logger.error(f"Unknown flagging type: {toflag}. Please use 'cal' or 'target'.")
        return
    
    save_flags(logger, obs_id, ms, 'after')
    flag_summary(logger, ms, obs_id)
    plot_flags(logger, obs_id, ms, path, 'after')

    logger.info("Flag step completed successfully!")
    logger.info("")
    logger.info("")
    logger.info("######################################################")
    logger.info("###################### END FLAG ######################")
    logger.info("######################################################")
    logger.info("")
    logger.info("")
    # log.append_to_google_doc("Flag step completed successfully!", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("###################### END FLAG ######################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("######################################################", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")
    # log.append_to_google_doc("", "", warnings="", plot_link="", doc_name="ViMS Pipeline Plots")