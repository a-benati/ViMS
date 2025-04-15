from __future__ import annotations

from pathlib import Path

from spinifex import h5parm_tools
from spinifex.vis_tools import ms_tools
import argparse


parser = argparse.ArgumentParser(description='Caclulate mean RM of the ionosphere')
parser.add_argument('--path', type=str, help='Path to observation folder')
parser.add_argument('--obs', type=str, help='Prefix of the output files')
parser.add_argument('--ms', type=str, help='Name of measurement set')
parser.add_argument('--refant', type=str, help='Reference antenna', default='m000')
args = parser.parse_args()


msdir = args.path + '/msdir/' + args.ms
ms_path = Path(msdir)
ionex_dir = '/localwork/angelina/meerkat_virgo/IONEXdata/'



# not useful for combined calibrator ms file
#ms_metadata = ms_tools.get_metadata_from_ms(ms_path)
#rm = ms_tools.get_rm_from_ms(ms_path, use_stations=ms_metadata.station_names,prefix='uqr', output_directory=ionex_dir)

