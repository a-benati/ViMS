#!/usr/bin/env python3

import numpy as np
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