# Cross- and polarisation calibration script for MeerKAT L-band data, 1 linearly pol calibrator + 1 unpolarised calibrator
#based on Ben Hugo's calibration script recipe.casa.py and christopher Hales EVLA memo 201


#!/usr/bin/env python3
import numpy as np
import os,sys
from casatasks import *
from casatools import table
#from casaplotms import plotms
import argparse



parser = argparse.ArgumentParser(description='Do crosscal on MeerKAT for 1 linearly pol calibrator + 1 unpolarised calibrator')
parser.add_argument('calms', type=str, help='Name of measurement set to calibrate')
parser.add_argument('--ant', type=str, help='Reference antenna', default='m000')
parser.add_argument('--obs', type=str, help='Observation number', required=True)
parser.add_argument('--path', type=str, help='Path to Observation folder', required=True)
parser.add_argument('--fields', type=int, nargs=4, help='calibrator IDs in the order fluxcal,bpcal,gcal,polcal', default=[2,2,0,1])
parser.add_argument('--do_plot', action='store_true', help=' create the diagnostic plots (default: False)')
parser.add_argument('--self_xcal', action='store_true', help='Do selfcal on xcal (default: False)')
args = parser.parse_args()

#definition for readable flag summary
def print_flagsum(summary):
    """
    Prints a readable version of the flagging summary given.
    
    Parameters:
        summary: name of the summary dictionary created by flagdata
    """
    
    # 1. Total flagged summary
    if 'flagged' in summary and 'total' in summary:
        flagged = summary['flagged']
        total = summary['total']
        perc = (flagged / total) * 100 if total > 0 else 0
        print("Total flagged: {}/{} ({:.2f}%)".format(flagged, total, perc))

    # 2. Per correlation
    if 'correlation' in summary:
        print("Flags per Correlation:")
        for corr, stats in summary['correlation'].items():
            flagged = stats['flagged']
            total = stats['total']
            perc = (flagged / total) * 100 if total > 0 else 0
            print("   Correlation {}: {}/{} flagged ({:.2f}%)".format(corr, flagged, total, perc))
        print()

    # 3. Per field
    if 'field' in summary:
        print("Flags per Field:")
        for field, stats in summary['field'].items():
            flagged = stats['flagged']
            total = stats['total']
            perc = (flagged / total) * 100 if total > 0 else 0
            print("   Field {}: {}/{} flagged ({:.2f}%)".format(field, flagged, total, perc))
        print()




calms = args.path +'/msdir/'+ args.calms

obs = args.obs
path = args.path

# Name your gain tables
ktab = path+'/output/caltables/'+obs+'_calib.kcal'
gtab_p = path+'/output/caltables/'+obs+'_calib.gcal_p'
gtab_a = path+'/output/caltables/'+obs+'_calib.gcal_a'
btab = path+'/output/caltables/'+obs+'_calib.bandpass'
ktab2 = path+'/output/caltables/'+obs+'_calib.kcal2'
gtab_p2 = path+'/output/caltables/'+obs+'_calib.gcal_p2'
gtab_a2 = path+'/output/caltables/'+obs+'_calib.gcal_a2'
btab2 = path+'/output/caltables/'+obs+'_calib.bandpass2'
ktab_sec = path+'/output/caltables/'+obs+'_calib.kcal.sec'
gtab_sec_p = path+'/output/caltables/'+obs+'_calib.gcal_p.sec'
Ttab_sec = path+'/output/caltables/'+obs+'_calib.T.sec'
ktab_pol = path+'/output/caltables/'+obs+'_calib.kcal.pol'
gtab_pol_p = path+'/output/caltables/'+obs+'_calib.gcal_p.pol'
Ttab_pol = path+'/output/caltables/'+obs+'_calib.T.pol'

# Name your polcal tables
ptab_df = path+'/output/caltables/'+obs+'_calib.df'
ptab_df2 = path+'/output/caltables/'+obs+'_calib.df2'
kxtab = path+'/output/caltables/'+obs+'_calib.kcrosscal'
ptab_xf = path+'/output/caltables/'+obs+'_calib.xf'
ptab_xfcorr = path+'/output/caltables/'+obs+'_calib.xf.ambcorr'


###################################################################
#get names of the fields from table
fields = args.fields

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
inital_flag = any(entry['name'] == obs+'_initial_flag' for entry in flag_versions.values())


if inital_flag:
    flagmanager(vis=calms, mode='restore', versionname=obs+'_initial_flag', merge='replace')
    print("Found '"+obs+"_initial_flag'. Restoring it.")

else:
    flagmanager(vis=calms, mode='save', versionname=obs+'_initial_flag', merge='replace')
    print("No 'initial_flag' found. Save current flagging state.")

print()
print('Clearing calibrations')
clearcal(vis=calms)
print()

###################################################################
# Set model for flux and polarisation calibrator


for cal in set(fcal.split(',')+bpcal.split(',')+xcal.split(',')):

    if cal == 'J1939-6342':
        print('setting model for flux calibrator J1939-6342')
        setjy(vis = calms, field = "{0}".format(fcal), spw = "", selectdata = False, timerange = "", scan = "", \
            standard = 'Stevens-Reynolds 2016', scalebychan = True, useephemdir = False, usescratch = True)


    elif cal =='J1331+3030':
        print('setting model for polarisation calibrator J1331+3030')

        
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
      print("Unknown calibrator, insert model in the script please ", cal)
      sys.exit()

###################################################################
# crosscalibration of the primary calibrator

ref_ant = args.ant

print()
print('Starting crosscalibration of the primary following the calibration scheme KGBFKGB')
print('using the following reference antenna: ', ref_ant)

# Delay calibration
gaincal(vis = calms, caltable = ktab, selectdata = True,\
    solint = "inf", field = bpcal, combine = "scan",uvrange='',\
    refant = ref_ant, solnorm = False, gaintype = "K",\
    minsnr=3,parang = False)

if args.do_plot ==True:
    #plotms(vis=ktab, xaxis='time', coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_Kcal.png',showgui=False,overwrite=True)
    os.system(f'ragavi-gains --table {ktab} --field 2 --htmlname {path}/output/diagnostic_plots/crosscal/{obs}_Kcal --plotname {path}/output/diagnostic_plots/crosscal/{obs}_Kcal.png')


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

if args.do_plot ==True:
    #plotms(vis=gtab_a, xaxis='time', yaxis='amplitude', coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_Gcal_amp.png',showgui=False,overwrite=True)
    #plotms(vis=gtab_p, xaxis='time', yaxis='phase', coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_Gcal_phase.png',showgui=False,overwrite=True)
    os.system(f'ragavi-gains --table {gtab_a} --field 2 -o {path}/output/diagnostic_plots/crosscal/{obs}_Gcal_amp -p {path}/output/diagnostic_plots/crosscal/{obs}_Gcal_amp.png')
    os.system(f'ragavi-gains --table {gtab_p} --field 2 -o {path}/output/diagnostic_plots/crosscal/{obs}_Gcal_phase -p {path}/output/diagnostic_plots/crosscal/{obs}_Gcal_phase.png')

# bandpass cal on bandpass calibrator
bandpass(vis = calms, caltable = btab, selectdata = True,\
         solint = "inf", field = bpcal, combine = "scan", uvrange='',\
         refant = ref_ant, solnorm = False, bandtype = "B",\
    gaintable = [ktab,gtab_p,gtab_a], gainfield = ['','',''],\
    interp = ['','',''], parang = False)

if args.do_plot ==True:
    #plotms(vis=btab, xaxis='chan',yaxis='amp',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'Bpcal_amp.png',showgui=False,overwrite=True)
    #plotms(vis=btab, xaxis='chan',yaxis='phase',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'Bpcal_phase.png',showgui=False,overwrite=True)
    os.system(f'ragavi-gains --table {btab} --field 2 -o {path}/output/diagnostic_plots/crosscal/{obs}_Bpcal -p {path}/output/diagnostic_plots/crosscal/{obs}_Bpcal.png')
    #os.system(f'shadems {btab} -x FREQ -y phase -c ANTENNA1 --dir {path}/output/diagnostic_plots/crosscal --png {obs}Bpcal_phase.png')

applycal(vis=calms, field=bpcal, gaintable=[ktab,gtab_p,gtab_a,btab], applymode='calflag', flagbackup=False)

if args.do_plot ==True:
    #plotms(vis=calms, xaxis='phase', yaxis='amplitude', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XX,YY', field=bpcal,avgscan=True, avgtime='99999999', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+str(bpcal)+'_crosscal_XXYY.png',showgui=False,overwrite=True)
    #plotms(vis=calms, xaxis='frequency', yaxis='amplitude', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XY,YX', field=bpcal,avgscan=True, avgtime='99999999', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+str(bpcal)+'_crosscal_XYYX.png',showgui=False,overwrite=True)
    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {bpcal} --dir {path}/output/diagnostic_plots/crosscal --png {obs}_{bpcal}_crosscal_XXYY.png')
    os.system(f'shadems {calms} -x FREQ -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/output/diagnostic_plots/crosscal --png {obs}_{bpcal}_crosscal_XYYX.png')

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
print_flagsum(crosscal_flag)

# Refined delay calibration
gaincal(vis = calms, caltable = ktab2, selectdata = True,\
    solint = "inf", field = bpcal, combine = "scan",uvrange='',\
    refant = ref_ant, solnorm = False, gaintype = "K",\
    minsnr=3,parang = False, gaintable=[btab, gtab_p, gtab_a])

if args.do_plot ==True:
    #plotms(vis=ktab2, xaxis='time', coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_Kcal2.png',showgui=False,overwrite=True,)
    os.system(f'ragavi-gains --table {ktab2} --field 2 -o {path}/output/diagnostic_plots/crosscal/{obs}_Kcal2 -p {path}/output/diagnostic_plots/crosscal/{obs}_Kcal2.png')

# refined phase cal on bandpass calibrator
gaincal(vis = calms, caltable = gtab_p2, selectdata = True,\
    solint = "60s", field = bpcal, combine = "",\
    refant = ref_ant, gaintype = "G", calmode = "p",uvrange='',refantmode='strict',\
    gaintable = [btab, ktab2], gainfield = ['',''], interp = ['',''],parang = False)

gaincal(vis = calms, caltable = gtab_a2, selectdata = True,\
    solint = "inf", field = bpcal, combine = "",\
    refant = ref_ant, gaintype = "G", calmode = "a",uvrange='',\
    gaintable = [btab, ktab2, gtab_p2], gainfield = ['','',''], interp = ['','',''],parang = False)

if args.do_plot ==True:
    #plotms(vis=gtab_a2, xaxis='time', yaxis='amplitude', coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_Gcal_amp2.png',showgui=False,overwrite=True)
    #plotms(vis=gtab_p2, xaxis='time', yaxis='phase', coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_Gcal_phase2.png',showgui=False,overwrite=True)
    os.system(f'ragavi-gains --table {gtab_a2} --field 2 -o {path}/output/diagnostic_plots/crosscal/{obs}_Gcal_amp2 -p {path}/output/diagnostic_plots/crosscal/{obs}_Gcal_amp2.png')
    os.system(f'ragavi-gains --table {gtab_p2} --field 2 -o {path}/output/diagnostic_plots/crosscal/{obs}_Gcal_phase2 -p {path}/output/diagnostic_plots/crosscal/{obs}_Gcal_phase2.png')

# refined bandpass cal on bandpass calibrator
bandpass(vis = calms, caltable = btab2, selectdata = True,\
         solint = "inf", field = bpcal, combine = "scan", uvrange='',\
         refant = ref_ant, solnorm = False, bandtype = "B",\
    gaintable = [ktab2,gtab_p2,gtab_a2], gainfield = ['','',''],\
    interp = ['','',''], parang = False)

if args.do_plot ==True:
    #plotms(vis=btab2, xaxis='chan',yaxis='amp',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_Bpcal_amp2.png',showgui=False,overwrite=True, showlegend=True)
    #plotms(vis=btab2, xaxis='chan',yaxis='phase',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_Bpcal_phase2.png',showgui=False,overwrite=True, showlegend=True)
    os.system(f'ragavi-gains --table {btab2} --field 2 -o {path}/output/diagnostic_plots/crosscal/{obs}_Bpcal2 -p {path}/output/diagnostic_plots/crosscal/{obs}_Bpcal2.png')
    #os.system(f'shadems {btab2} -x FREQ -y phase -c ANTENNA1 --dir {path}/output/diagnostic_plots/crosscal --png {obs}_Bpcal_phase2.png')

applycal(vis=calms, field=bpcal, gaintable=[ktab2,gtab_p2,gtab_a2,btab2], applymode='calflag', flagbackup=False)

if args.do_plot ==True:
    #plotms(vis=calms, xaxis='phase', yaxis='amplitude', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XX,YY', field=bpcal,avgscan=True, avgtime='99999999', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_'+str(bpcal)+'_crosscal_XXYY_flagged.png',showgui=False,overwrite=True)
    #plotms(vis=calms, xaxis='frequency', yaxis='amplitude', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XY,YX', field=bpcal,avgscan=True, avgtime='99999999', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_'+str(bpcal)+'_crosscal_XYYX_flagged.png',showgui=False,overwrite=True)
    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {bpcal} --dir {path}/output/diagnostic_plots/crosscal --png {obs}_{bpcal}_crosscal_XXYY_flagged.png')
    os.system(f'shadems {calms} -x FREQ -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/output/diagnostic_plots/crosscal --png {obs}_{bpcal}_crosscal_XYYX_flagged.png')

#####################################################################
#Do two iterations of leakage calibration

# flag corrceted data in XYYX and redo dfcal
flagdata(vis=calms, mode="rflag", datacolumn="corrected", field=bpcal, quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False, correlation='XY,YX')
flagdata(vis=calms, mode='extend', datacolumn="corrected", field=bpcal, growtime=80, growfreq=80, flagbackup=False, growaround=True, flagnearfreq=True, correlation='XY,YX')

if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_before_df' for entry in flag_versions.values()):
    flagmanager(vis=calms, mode='delete', versionname=obs+'_flag_before_df', merge='replace')
    print("Found 'flag_before_df'. Deleting it.")
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
    print("Found 'flag_df'. Deleting it.")
flagmanager(vis=ptab_df, mode='save', versionname=obs+'_flag_df', merge='replace')

if args.do_plot ==True:
    #plotms(vis=ptab_df, xaxis='chan',yaxis='amplitude',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_Dfcal_amp_flagged.png',showgui=False,overwrite=True)
    #plotms(vis=ptab_df, xaxis='chan',yaxis='phase',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_Dfcal_phase_flagged.png',showgui=False,overwrite=True)
    os.system(f'ragavi-gains --table {ptab_df} --field 2 -o {path}/output/diagnostic_plots/polcal/{obs}_Dfcal_flagged -p {path}/output/diagnostic_plots/polcal/{obs}_Dfcal_flagged.png')
    #os.system(f'shadems {ptab_df} -x FREQ -y phase -c ANTENNA1 --dir {path}/output/diagnostic_plots/polcal --png {obs}_Dfcal_phase_flagged.png')


# Apply Df to bpcal
applycal(vis=calms,field=fcal,gaintable=[ktab2,gtab_p2,gtab_a2,btab2,ptab_df],parang=False, flagbackup=False)

#flag corrceted data in XYYX and redo dfcal
flagdata(vis=calms, mode="rflag", datacolumn="corrected", field=bpcal, quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False, correlation='XY,YX')
flagdata(vis=calms, mode='extend', datacolumn="corrected", field=bpcal, growtime=80, growfreq=80, flagbackup=False, growaround=True, flagnearfreq=True, correlation='XY,YX')

if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_df_sec_iter' for entry in flag_versions.values()):
    flagmanager(vis=calms, mode='delete', versionname=obs+'_flag_df_sec_iter', merge='replace')
    print("Found 'flag_df_sec_iter'. Deleting it.")
flagmanager(vis=calms, mode='save', versionname=obs+'_flag_df_sec_iter', merge='replace')


polcal(vis = calms, caltable = ptab_df2, selectdata = True,\
           solint = 'inf', field = bpcal, combine = '',\
           refant = ref_ant, poltype = 'Dflls', preavg= 200.0,\
           gaintable = [ktab2, gtab_p2,gtab_a2,btab2],\
           gainfield = ['', '', '',''],\
           smodel=[14.8,0,0,0],\
           interp = ['', '', '', ''])

if args.do_plot ==True:
    #plotms(vis=ptab_df2, xaxis='chan',yaxis='amplitude',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_Dfcal_amp_preflag2.png',showgui=False,overwrite=True)
    #plotms(vis=ptab_df2, xaxis='chan',yaxis='phase',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_Dfcal_phase_preflag2.png',showgui=False,overwrite=True)
    os.system(f'ragavi-gains --table {ptab_df2} --field 2 -o {path}/output/diagnostic_plots/polcal/{obs}_Dfcal_preflag2 -p {path}/output/diagnostic_plots/polcal/{obs}_Dfcal_preflag2.png')
    #os.system(f'shadems {ptab_df2} -x FREQ -y phase -c ANTENNA1 --dir {path}/output/diagnostic_plots/polcal --png {obs}_Dfcal_phase_preflag2.png')

flagdata(vis=ptab_df2, mode='tfcrop',datacolumn="CPARAM", quackinterval=0.0,ntime="60s",combinescans=True,timecutoff=5.0, freqcutoff=3.0, usewindowstats="both", flagbackup=False)
df_flagversions2 = flagmanager(vis=ptab_df2, mode='list')
if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_df2' for entry in df_flagversions2.values()):
    flagmanager(vis=ptab_df2, mode='delete', versionname=obs+'_flag_df2', merge='replace')
    print("Found 'flag_df2'. Deleting it.")
flagmanager(vis=ptab_df2, mode='save', versionname=obs+'_flag_df2', merge='replace')

if args.do_plot ==True:
    #plotms(vis=ptab_df2, xaxis='chan',yaxis='amplitude',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_Dfcal_amp_flagged2.png',showgui=False,overwrite=True)
    #plotms(vis=ptab_df2, xaxis='chan',yaxis='phase',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_Dfcal_phase_flagged2.png',showgui=False,overwrite=True)
    os.system(f'ragavi-gains --table {ptab_df2} --field 2 -o {path}/output/diagnostic_plots/polcal/{obs}_Dfcal_flagged2 -p {path}/output/diagnostic_plots/polcal/{obs}_Dfcal_flagged2.png')
    #os.system(f'shadems {ptab_df2} -x FREQ -y phase -c ANTENNA1 --dir {path}/output/diagnostic_plots/polcal --png {obs}_Dfcal_phase_flagged2.png')

applycal(vis=calms,field=gcal,gaintable=[ktab2,gtab_p2,gtab_a2,btab2,ptab_df2],parang=False, flagbackup=False)

flagdata(vis=calms, mode="rflag", field=gcal, datacolumn="corrected", quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False)
flagdata(vis=calms, mode='extend', field=gcal, datacolumn='corrected', growtime=80, growfreq=80, flagbackup=False, growaround=True, flagnearfreq=True)

if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_after_df' for entry in flag_versions.values()):
    flagmanager(vis=calms, mode='delete', versionname=obs+'_flag_after_df', merge='replace')
    print("Found 'flag_after_df'. Deleting it.")
flagmanager(vis=calms, mode='save', versionname=obs+'_flag_after_df', merge='replace')
df_cal_flag = flagdata(vis=calms, mode='summary')
print()
print('Flagging summary after Df calibration:')
print_flagsum(df_cal_flag)

# Check that amplitude of leakage cal is gone down (few %) after calibration
if args.do_plot ==True:
    #plotms(vis=calms,xaxis='freq',yaxis='amplitude',correlation='XY,YX',field=bpcal,avgscan=True,ydatacolumn='data',avgtime='9999999',plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_'+str(bpcal)+'Df-DATA.png',showgui=False,overwrite=True)
    #plotms(vis=calms,xaxis='freq',yaxis='amplitude',correlation='XY,YX',field=bpcal,avgscan=True,ydatacolumn='corrected',avgtime='9999999',plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_'+str(bpcal)+'-Df-CORRECTED.png',showgui=False,overwrite=True)
    #plotms(vis=calms, xaxis='phase', yaxis='amplitude', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XX,YY', field=gcal,avgscan=True, avgtime='99999999', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+'_'+str(gcal)+'_crosscal_XXYY.png',showgui=False,overwrite=True)
    os.system(f'shadems {calms} -x FREQ -y DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/output/diagnostic_plots/crosscal --png {obs}_{bpcal}_Df-DATA.png')
    os.system(f'shadems {calms} -x FREQ -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {bpcal} --dir {path}/output/diagnostic_plots/crosscal --png {obs}_{bpcal}_Df-CORRECTED.png')
    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {gcal} --dir {path}/output/diagnostic_plots/crosscal --png {obs}_{gcal}_crosscal_XXYY.png')

#####################################################################################################
# Crosscalibration of the secondary calibrator
print()
print('Starting crosscalibration of the secondary calibrator')

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
if args.do_plot ==True:
    applycal(vis=calms, field=gcal, gaintable=[ktab_sec, gtab_sec_p,gtab_a2,btab2,Ttab_sec,ptab_df2], parang=False, flagbackup=False)
    #plotms(vis=calms, xaxis='uvdist', yaxis='amplitude', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='antenna1', correlation='XX,YY', field=gcal,avgscan=True, avgtime='99999999', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+str(gcal)+'_amp_XXYY.png',showgui=False,overwrite=True)
    #plotms(vis=calms, xaxis='uvdist', yaxis='phase', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XX,YY', field=gcal,avgscan=True, avgtime='99999999', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+str(gcal)+'_phase_XXYY.png',showgui=False,overwrite=True)
    #plotms(vis=calms, xaxis='phase', yaxis='amplitude', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XX,YY', field=gcal,avgscan=True, avgtime='99999999', plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+str(gcal)+'_phaserefine_XXYY.png',showgui=False,overwrite=True)
    #plotms(vis=calms,xaxis='freq',yaxis='amplitude',correlation='XY,YX',field=gcal,avgscan=True,ydatacolumn='corrected',avgtime='9999999',plotfile=path+'/output/diagnostic_plots/crosscal/'+str(obs)+str(gcal)+'-Df-CORRECTED.png',showgui=False,overwrite=True)
    os.system(f'shadems {calms} -x UV -y CORRECTED_DATA:amp -c ANTENNA1 --corr XX,YY --field {gcal} --dir {path}/output/diagnostic_plots/crosscal --png {obs}_{gcal}_amp_XXYY.png')
    os.system(f'shadems {calms} -x UV -y CORRECTED_DATA:phase -c ANTENNA1 --corr XX,YY --field {gcal} --dir {path}/output/diagnostic_plots/crosscal --png {obs}_{gcal}_phase_XXYY.png')
    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY  --field {gcal} --dir {path}/output/diagnostic_plots/crosscal --png {obs}_{gcal}_phaserefine_XXYY.png')


#apply calibration up to  now to xcal: XY and YX will vary with time due to pang
applycal(vis=calms,field=xcal,gaintable=[ktab,gtab_p2,gtab_a2,btab2,ptab_df2],parang=False, flagbackup=False)

flagdata(vis=calms, mode="rflag", field=xcal, datacolumn="corrected", quackinterval=0.0, timecutoff=4.0, freqcutoff=3.0, extendpols=False, flagbackup=False, outfile="",overwrite=True, extendflags=False)
flagdata(vis=calms, mode='extend', field=xcal, datacolumn='corrected', growtime=80, growfreq=80, flagbackup=False, growaround=True, flagnearfreq=True)
if any(isinstance(entry, dict) and entry.get('name') == obs+'_flag_before_xf' for entry in flag_versions.values()):
    flagmanager(vis=calms, mode='delete', versionname=obs+'_flag_before_xf', merge='replace')
    print("Found 'flag_before_xf'. Deleting it.")
flagmanager(vis=calms, mode='save', versionname=obs+'_flag_before_xf', merge='replace')


if args.do_plot ==True:
     #plotms(vis=calms, xaxis='phase', yaxis='amplitude', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XX,YY', field=xcal,avgscan=True, avgtime='99999999', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_'+str(xcal)+'_precalXf_XXYY.png',showgui=False,overwrite=True)
     #plotms(vis=calms, xaxis='time', yaxis='amplitude', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XY,YX', field=xcal,avgscan=True, avgchannel='99999999', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_'+str(xcal)+'_precalXf_XYYX.png',showgui=False,overwrite=True)
    os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {xcal} --dir {path}/output/diagnostic_plots/polcal --png {obs}_{xcal}_precalXf_XXYY.png')
    os.system(f'shadems {calms} -x TIME -y CORRECTED_DATA:amp -c CORR --corr XY,YX --field {xcal} --dir {path}/output/diagnostic_plots/polcal --png {obs}_{xcal}_precalXf_XYYX.png')

#############################################################################################################
#calibration of the polarisation calibrator
print()
print('Starting crosscalibration of the polarisation calibrator')

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
if args.self_xcal==True:
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

if args.do_plot ==True:
     applycal(vis=calms,field=xcal,gaintable=[ktab_pol, gtab_pol_p, gtab_a2, btab2, Ttab_sec, ptab_df2],parang=False, flagbackup=False)
     #plotms(vis=calms, xaxis='phase', yaxis='amplitude', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XX,YY', field=xcal,avgscan=True, avgtime='99999999', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_'+str(xcal)+'_precalXf_XXYY_refinephase.png',showgui=False,overwrite=True)
     #plotms(vis=calms, xaxis='imag', yaxis='real', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XY,YX', field=xcal,avgscan=True, avgchannel='99999999', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_'+str(xcal)+'_precalXf_XYYX_real_im.png',showgui=False,overwrite=True)
     os.system(f'shadems {calms} -x CORRECTED_DATA:phase -y CORRECTED_DATA:amp -c CORR --corr XX,YY --field {xcal} --dir {path}/output/diagnostic_plots/polcal --png {obs}_{xcal}_precalXf_XXYY_refinephase.png')
     os.system(f'shadems {calms} -x CORRECTED_DATA:imag -y CORRECTED_DATA:real -c CORR --corr XY,YX --field {xcal} --dir {path}/output/diagnostic_plots/polcal --png {obs}_{xcal}_precalXf_XYYX_real_im.png')


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

 
if args.do_plot ==True:
    #plotms(vis=ptab_xf, xaxis='chan',yaxis='amplitude',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_Xfcal_amp.png',showgui=False,overwrite=True)
    #plotms(vis=ptab_xf, xaxis='chan',yaxis='phase',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_Xfcal_phase.png',showgui=False,overwrite=True)
    os.system(f'ragavi-gains --table {ptab_xf} --field 1 -o {path}/output/diagnostic_plots/polcal/{obs}_Xfcal -p {path}/output/diagnostic_plots/polcal/{obs}_Xfcal.png')
    #os.system(f'shadems {ptab_xf} -x FREQ -y phase -c ANTENNA1 --dir {path}/output/diagnostic_plots/polcal --png {obs}_Xfcal_phase.png')

print()
print('Correcting for phase ambiguity')
exec(open('/localwork/angelina/meerkat_virgo/ViMS/ViMS/scripts/xyamb_corr.py').read())
S=xyamb(xytab=ptab_xf ,xyout=ptab_xfcorr)

if args.do_plot ==True:
    #plotms(vis=ptab_xfcorr, xaxis='chan',yaxis='phase',coloraxis='antenna1', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_Xfcal_phase_ambcorr.png',showgui=False,overwrite=True)
    os.system(f'ragavi-gains --table {ptab_xfcorr} --field 1 -o {path}/output/diagnostic_plots/polcal/{obs}_Xfcal_ambcorr -p {path}/output/diagnostic_plots/polcal/{obs}_Xfcal_ambcorr.png')


applycal(vis=calms, field=xcal,gaintable=[ktab_pol, gtab_pol_p, kxtab, ptab_xfcorr, gtab_a2, btab2, Ttab_pol, ptab_df2],parang=True, flagbackup=False)
applycal(vis=calms, field=gcal,gaintable=[ktab_sec, gtab_sec_p, kxtab, ptab_xfcorr, gtab_a2, btab2, Ttab_sec, ptab_df2], parang=True, flagbackup=False)
applycal(vis=calms, field=fcal,gaintable=[ktab2, gtab_p2, kxtab, ptab_xfcorr, gtab_a2, btab2, ptab_df2], parang=True, flagbackup=False)

final_flag = flagdata(vis=calms, mode='summary')
print()
print('Flagging summary after cross and pol calibration:')
print_flagsum(final_flag)

# Check: plot imaginary versis real and compare to previous plot

if args.do_plot==True:
    #plotms(vis=calms, xaxis='imag', yaxis='real', xdatacolumn='corrected', ydatacolumn='corrected', coloraxis='corr', correlation='XY,YX', field=xcal,avgscan=True, avgchannel='99999999', plotfile=path+'/output/diagnostic_plots/polcal/'+str(obs)+'_'+str(xcal)+'aftercalXf_XYYX_real_im.png',showgui=False,overwrite=True)
    os.system(f'shadems {calms} -x CORRECTED_DATA:imag -y CORRECTED_DATA:real -c CORR --corr XY,YX --field {xcal} --dir {path}/output/diagnostic_plots/polcal --png {obs}_{xcal}aftercalXf_XYYX_real_im.png')
    os.system(f'ragavi-vis --ms {calms} -x frequency -y phase -dc CORRECTED tbin 12000 -ca antenna1 --corr XY,YX --field {xcal} -o {path}/output/diagnostic_plots/polcal/{obs}_{xcal}aftercalXf_XYYX_phase.png')



