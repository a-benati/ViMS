

def xyamb(xytab, xyout=''):
    import time
    #from casatools import table as tb
    import numpy as np
    """
    Resolve the 180-degree cross-hand phase ambiguity in a CASA calibration table.
    CAlculates the mean phase and shifts every point deviating more then 90 degrees from the mean phase by 180 degrees.

    Parameters:
    xytab : str
        Path to the input calibration table.
    xyout : str, optional
        Path to the output calibration table. If not specified, the input table is modified in place.
    """

    if xyout == '':
        xyout = xytab
    if xyout != xytab:
        tb.open(xytab)
        tb.copy(xyout)
        tb.clearlocks()
        tb.close()
    

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
            avg_phase = np.angle(np.mean(c[0, :, :][~fl[0,:,:]]), True)
            print('Average phase = '+str(avg_phase))
            for ch in range(num_channels):
                valid_data = c[0,ch,:][~fl[0,ch,:]]
                if valid_data.size > 0:
                    xyph0 = np.angle(np.mean(valid_data), True)

                    # Calculate the phase difference
                    phase_diff =  np.abs(((xyph0 - avg_phase)))

                    if phase_diff >= 100.0:
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
