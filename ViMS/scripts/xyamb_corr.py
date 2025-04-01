

def xyamb(xytab, qu, xyout=''):
    import time
    #from casatools import table as tb
    import numpy as np
    """
    Resolve the 180-degree cross-hand phase ambiguity in a CASA calibration table.

    Parameters:
    xytab : str
        Path to the input calibration table.
    qu : tuple of float
        Expected (Q, U) values of the calibrator.
    xyout : str, optional
        Path to the output calibration table. If not specified, the input table is modified in place.

    Returns:
    list
        The corrected Stokes parameters [I, Q, U, V].
    """
    if not isinstance(qu, tuple) or len(qu) != 2:
        raise ValueError("qu must be a tuple: (Q, U)")

    if xyout == '':
        xyout = xytab
    if xyout != xytab:
        tb.open(xytab)
        tb.copy(xyout)
        tb.clearlocks()
        tb.close()
    
    qu_exp = complex(qu[0], qu[1])
    print('Expected QU = '+str(qu) )

    #tb=table()
    tb.open(xyout, nomodify=False)

    spw_ids = np.unique(tb.getcol('SPECTRAL_WINDOW_ID'))

    for spw in spw_ids:
        st = tb.query('SPECTRAL_WINDOW_ID=='+str(spw))
        if st.nrows() > 0:
            c = st.getcol('CPARAM')
            fl = st.getcol('FLAG')

            num_channels = c.shape[1]
            flipped_channels=0
            for ch in range(num_channels):
                valid_data = c[0,ch,:][~fl[0,ch,:]]
                if valid_data.size > 0:
                    xyph0 = np.angle(np.mean(valid_data), True)

                    # Calculate the phase difference
                    phase_diff =  np.abs(((xyph0 - np.angle(qu_exp,True)) % 360))

                    if phase_diff > 90.0:
                        flipped_channels += 1
                        c[0, ch, :] *= -1.0
                        st.putcol('CPARAM', c)
            
            print('Flipped '+str(flipped_channels)+' channels in SPW '+str(spw))
            
            st.close()
            time.sleep(1)
    
    tb.clearlocks()
    tb.flush()
    tb.close()
    time.sleep(1)

    # Return the expected Stokes parameters
    stokes = [1.0, qu[0], qu[1], 0.0]
    return stokes