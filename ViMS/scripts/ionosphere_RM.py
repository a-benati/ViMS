from __future__ import annotations

import pyrap.tables as tab
from pylab import *
import RMextract.getRM as gt
import argparse
import numpy as np
from casatools import table

rc("lines",lw=3)
rc('font', family='serif', size=16)

parser = argparse.ArgumentParser(description='Caclulate mean RM of the ionosphere')
parser.add_argument('--path', type=str, help='Path to observation folder')
parser.add_argument('--obs', type=str, help='Prefix of the output files')
parser.add_argument('--ms', type=str, help='Name of measurement set')
parser.add_argument('--refant', type=str, help='Reference antenna', default='m000')
parser.add_argument('--freq_list', type=str, help='Name of the frequency list file')
args = parser.parse_args()


msdir = args.path + '/msdir/' + args.ms
ionex_dir = '/localwork/angelina/meerkat_virgo/IONEXdata/'
freq_arr = np.loadtxt(args.path +'/STOKES_CUBES/' + args.freq_list)
pointing = [3.539257790414, 0.53248520675] #direction of 3C286
field_id = 1

tec = gt.getRM(MS=msdir, ionexPath=ionex_dir, server='ftp://gssc.esa.int/gnss/products/ionex/',earth_rot=0.5,ha_limit=1*np.pi, radec=pointing, prefix='UQRG', out_file=args.path + '/STOKES_CUBES/' + args.obs + '_ioncorr.txt')

times_tot = np.squeeze(tec['times'])
flags = tec['flags'][args.refant]
maskeddata=np.ma.array(tec['RM'][args.refant],mask=np.logical_not(flags))

RM = np.squeeze(maskeddata)

tb = table()
tb.open(msdir)

times_pol = tb.query(f'FIELD_ID == {field_id}').getcol('TIME')
tb.close()

a = times_pol[0]
b = times_pol[-1]

time_pol = []
rm_pol = []

for ind, x in enumerate(times_tot):
    if x <= b and x >= a:
        time_pol.append(x)
        y = RM[ind]
        rm_pol.append(y)

RM_avg = np.mean(rm_pol)
print('mean ionospehric RM:',RM_avg)