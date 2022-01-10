"""
Area and amplitude spectrum for photon recorded in different pile-up fraction and dpe fraction.
Lanqing Yuan, Jan 10 2022
"""

import numpy as np
import straxen
import pandas as pd
import sys


########################
# Amplitude acceptance #
########################


def get_avg_spe_amp(top, trunc_bound=[-10,400],
                    spe_amps_path='/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'):
    """From LED calibration result get the average SPE amplitude spectrum for either array.

    Args:
        top (bool): whether or not we specify top array. 
        trunc_bound (list, optional): ADC range that we keep the spectrums. Please put None if you don't want any truncation. Defaults to [-10,400].
        spe_amps_path (str, optional): the LED calibrated SPE amplitude spectrum to load. Defaults to '/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'.

    Returns:
        spe_amps_indices(1darray): coordinate of amplitude spectrum in unit of ADC.
        avg_spe_amp(1darray): probability density of spe in a certaint array to have height in certain ADC. 
    """
    spe_amps = np.load(spe_amps_path)['y']
    binning_config = spe_amps[spe_amps['channel']==0]['binning'][0]
    spe_amps_indices = np.arange(binning_config[0]+1,binning_config[1],binning_config[2], dtype=np.int)
    spe_amps = spe_amps['noise_subtracted_5bin'] # considered the best by Giovanni; temporarily hard-coded.

    # normalization
    spe_amps_norm = np.zeros(np.shape(spe_amps))
    for i in range(494):
        spe_amps_norm[i] = spe_amps[i]/spe_amps[i].sum()
        if np.isnan(spe_amps_norm[i]).any():
            spe_amps_norm[i] = 0

    spe_amps_top = np.mean(spe_amps_norm[:253], axis=0)
    spe_amps_bot = np.mean(spe_amps_norm[253:], axis=0)
    spe_amps_top = spe_amps_top/spe_amps_top.sum()
    spe_amps_bot = spe_amps_bot/spe_amps_bot.sum()

    if trunc_bound != None:
        adc_range = np.arange(trunc_bound[0], trunc_bound[1])
        spe_amps_top = spe_amps_top[np.where(spe_amps_indices==trunc_bound[0])[0][0]:np.where(spe_amps_indices==trunc_bound[1])[0][0]]
        spe_amps_bot = spe_amps_bot[np.where(spe_amps_indices==trunc_bound[0])[0][0]:np.where(spe_amps_indices==trunc_bound[1])[0][0]]
        spe_amps_indices = adc_range

    if top:
        return spe_amps_indices, spe_amps_top
    else:
        return spe_amps_indices, spe_amps_bot


def get_avg_dpe_amp(top, shift = 99, trunc_bound=[-10,400],
                    spe_amps_path='/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'):
    """From LED calibration result and based on self convolution, get the average DPE amplitude spectrum for 
    either array. Note that we are assuming the DPE mechanism that makes DPE equivalent to just two independent SPEs.

    Args:
        top (bool): whether or not we specify top array. 
        shift (int, optional): shift in number of indicies in self convolution to make sure DPE has mean exactly as twice of SPE. Defaults to 99.
        trunc_bound (list, optional): ADC range that we keep the spectrums. Please put None if you don't want any truncation. Defaults to [-10,400].
        spe_amps_path (str, optional): the LED calibrated SPE amplitude spectrum to load. Defaults to '/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'.

    Returns:
        dpe_amps_indices(1darray): coordinate of amplitude spectrum in unit of ADC.
        avg_dpe_amp(1darray): probability density of spe in a certaint array to have this height.
    """
    spe_amps_indices, spe_amps_top = get_avg_spe_amp(True, trunc_bound=None, spe_amps_path=spe_amps_path)
    spe_amps_indices, spe_amps_bot = get_avg_spe_amp(False, trunc_bound=None, spe_amps_path=spe_amps_path)

    dpe_amps_top = np.convolve(spe_amps_top, spe_amps_top, 'full')
    dpe_amps_top = dpe_amps_top/dpe_amps_top.sum() # normalization
    dpe_amps_bot = np.convolve(spe_amps_bot, spe_amps_bot, 'full')
    dpe_amps_bot = dpe_amps_bot/dpe_amps_bot.sum() # normalization

    # shift the spectrum coordinates to obtain mean(DPE) = 2*mean(SPE)
    shift = 99
    dpe_amps_indices = np.linspace(spe_amps_indices[0]-shift, 
                                    spe_amps_indices[-1]+len(spe_amps_indices)-shift-1,
                                    len(spe_amps_indices)*2-1, dtype=np.int)
    assert (abs(np.sum(dpe_amps_indices*dpe_amps_top) - 2*np.sum(spe_amps_indices*spe_amps_top))<1
            and abs(np.sum(dpe_amps_indices*dpe_amps_top) == 2*np.sum(spe_amps_indices*spe_amps_top))<1
            ), 'The input shift is wrong! It does not allow mean DPE equal to twice mean SPE.'

    
    if trunc_bound != None:
        adc_range = np.arange(trunc_bound[0], trunc_bound[1])
        dpe_amps_top = dpe_amps_top[np.where(dpe_amps_indices==trunc_bound[0])[0][0]:np.where(dpe_amps_indices==trunc_bound[1])[0][0]]
        dpe_amps_bot = dpe_amps_bot[np.where(dpe_amps_indices==trunc_bound[0])[0][0]:np.where(dpe_amps_indices==trunc_bound[1])[0][0]]
        dpe_amps_indices = adc_range

    if top:
        return dpe_amps_indices, dpe_amps_top
    else:
        return dpe_amps_indices, dpe_amps_bot


def get_avg_sphd_amp(top, shift = 99, trunc_bound=[-10,400], dpes=np.linspace(0.18,0.24,100),
                     spe_amps_path='/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'):
    """From LED calibration result, get the average single photon detected amplitude spectrum without
    any pile-up. The spectrum is given by a weighted sum of spe and dpe amplitude spectrum.

    Args:
        top (bool): whether or not we specify top array. 
        shift (int, optional): shift in number of indicies in self convolution to make sure DPE has mean exactly as twice of SPE. Defaults to 99.
        trunc_bound (list, optional): ADC range that we keep the spectrums. Defaults to [-10,400].
        dpes (1darray or float, optional): DPE fraction. Defaults to np.linspace(0.18,0.24,100).
        spe_amps_path (str, optional): the LED calibrated SPE amplitude spectrum to load. Defaults to '/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'.

    Returns:
        sphd_amps_indices(1darray): coordinate of amplitude spectrum in unit of ADC.
        avg_sphd_amp(2darray): axis0=dpe, axis1=adc. Probability density of sphd in a certain array to have this height.

    """
    spe_amps_indices, spe_amps_top = get_avg_spe_amp(True, trunc_bound=trunc_bound, spe_amps_path=spe_amps_path)
    spe_amps_indices, spe_amps_bot = get_avg_spe_amp(False, trunc_bound=trunc_bound, spe_amps_path=spe_amps_path)
    dpe_amps_indices, dpe_amps_top = get_avg_dpe_amp(True, shift=shift, trunc_bound=trunc_bound, spe_amps_path=spe_amps_path)
    dpe_amps_indices, dpe_amps_bot = get_avg_dpe_amp(False, shift=shift, trunc_bound=trunc_bound, spe_amps_path=spe_amps_path)

    if type(dpes) == float:
        dpes = np.array(dpes)
    sphd_amps_indices = np.arange(trunc_bound[0], trunc_bound[1])
    sphd_amps_top = np.zeros((len(dpes), trunc_bound[1]-trunc_bound[0]))
    sphd_amps_bot = np.zeros((len(dpes), trunc_bound[1]-trunc_bound[0]))    

    for i,dpe in enumerate(dpes):
        sphd_amps_top[i] = spe_amps_top*(1-dpe) + dpe_amps_top*dpe
        sphd_amps_top[i] = sphd_amps_top[i]/sphd_amps_top[i].sum() # normallization for security
        sphd_amps_bot[i] = spe_amps_bot*(1-dpe) + dpe_amps_bot*dpe
        sphd_amps_bot[i] = sphd_amps_bot[i]/sphd_amps_bot[i].sum() # normallization for security

    if top:
        return sphd_amps_indices, sphd_amps_top
    else:
        return sphd_amps_indices, sphd_amps_bot


def get_avg_dphd_amp(top, shift_dphd=10, shift = 99, trunc_bound=[-10,400], dpes=np.linspace(0.18,0.24,100),
                     spe_amps_path='/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'):
    """From LED calibration result, get the average perfectly piled-up double photon detected amplitude spectrum
    averaged in either array. The spectrum is given by self-convolution of sphd amplitude spectrum.

    Args:
        top (bool): whether or not we specify top array. 
        shift_dphd (int, optional): hift in number of indicies in self convolution to make sure DPhD has mean exactly as twice of SPhD. Defaults to 10.
        shift (int, optional): shift in number of indicies in self convolution to make sure DPE has mean exactly as twice of SPE. Defaults to 99.
        trunc_bound (list, optional): ADC range that we keep the spectrums. Defaults to [-10,400].
        dpes (1darray or float, optional): DPE fraction. Defaults to np.linspace(0.18,0.24,100).
        spe_amps_path (str, optional): the LED calibrated SPE amplitude spectrum to load. Defaults to '/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'.

    Returns:
        dphd_amps_indices(1darray): coordinate of amplitude spectrum in unit of ADC.
        avg_dphd_amp(2darray): axis0=dpe, axis1=adc. Probability density of sphd in a certain array to have this height.
    """
    sphd_amps_indices, sphd_amps_top = get_avg_sphd_amp(top=True,shift = shift,trunc_bound=trunc_bound, dpes=dpes,
                                                       spe_amps_path=spe_amps_path)
    sphd_amps_indices, sphd_amps_bot = get_avg_sphd_amp(top=False,shift = shift,trunc_bound=trunc_bound, dpes=dpes,
                                                       spe_amps_path=spe_amps_path)
    if type(dpes) == float:
        dpes = np.array(dpes)
    dphd_amps_top = np.zeros((len(dpes), 2*len(sphd_amps_indices)-1))
    dphd_amps_bot = np.zeros((len(dpes), 2*len(sphd_amps_indices)-1))

    for i,dpe in enumerate(dpes):
        dphd_amps_top[i] = np.convolve(sphd_amps_top[i], sphd_amps_top[i], 'full')
        dphd_amps_top[i] = dphd_amps_top[i]/dphd_amps_top[i].sum() # normallization for security
        dphd_amps_bot[i] = np.convolve(sphd_amps_bot[i], sphd_amps_bot[i], 'full')
        dphd_amps_bot[i] = dphd_amps_bot[i]/dphd_amps_bot[i].sum() # normallization for security
    
    dphd_amps_indices = np.arange(sphd_amps_indices[0]-shift_dphd,sphd_amps_indices[-1]+len(sphd_amps_indices)-shift_dphd, 1)
    if top:
        return dphd_amps_indices, dphd_amps_top
    else:
        return dphd_amps_indices, dphd_amps_bot


def get_avg_phd_acc(top, adc_threshold=15, pile_probs=np.linspace(0,0.4,100), 
                    shift = 99, trunc_bound=[-10,400], dpes=np.linspace(0.18,0.24,100),
                    spe_amps_path='/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'):
    """From LED calibration result, get the average phd acceptance for case of different dpe fraction and pile-up 
    fraction. When pile-up happens, we think the phd acceptance to be 1. On the other hand, we will use the acceptance
    from SPhD amplitude spectrum. 
    Args:
        top (bool): whether or not we specify top array. 
        adc_threshold (int): below this amplitude in unit of ADC the photon will be dropped by DAQ.
        pile_probs (1darray or float, optional): probability of photon pile-up happens. Defaults to np.linspace(0,0.4,100).
        shift (int, optional): shift in number of indicies in self convolution to make sure DPE has mean exactly as twice of SPE. Defaults to 99.
        trunc_bound (list, optional): ADC range that we keep the spectrums. Defaults to [-10,400].
        dpes (1darray or float, optional): DPE fraction. Defaults to np.linspace(0.18,0.24,100).
        spe_amps_path (str, optional): the LED calibrated SPE amplitude spectrum to load. Defaults to '/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'.

    Returns:
        dpes (1darray): DPE fraction.
        pile_probs (1darray): probability of photon pile-up happens.
        avg_phd_acc (2darray): axis0=dpes, axis1=pile_probs. Average acceptance of a PhD in a certain array with certain p_dpe and pile-up fraction.
    """
    sphd_amps_indices, sphd_amps_top = get_avg_sphd_amp(top=True,shift = shift,trunc_bound=trunc_bound, dpes=dpes,
                                                       spe_amps_path=spe_amps_path)
    sphd_amps_indices, sphd_amps_bot = get_avg_sphd_amp(top=False,shift = shift,trunc_bound=trunc_bound, dpes=dpes,
                                                       spe_amps_path=spe_amps_path)
    
    top_sphd_daq_loss = np.sum(sphd_amps_top[:,:np.where(sphd_amps_indices==adc_threshold)[0][0]+1], axis=1)
    bot_sphd_daq_loss = np.sum(sphd_amps_bot[:,:np.where(sphd_amps_indices==adc_threshold)[0][0]+1], axis=1)

    if type(dpes) == float:
        dpes = np.array(dpes)
    if type(pile_probs) == float:
        pile_probs = np.array(pile_probs)
    avg_phd_acc_top = np.zeros((len(dpes), len(pile_probs)))
    avg_phd_acc_bot = np.zeros((len(dpes), len(pile_probs)))

    for i,dpe in enumerate(dpes):
        for j,pile_prob in enumerate(pile_probs):
            avg_phd_acc_top[i,j] = 1 - top_sphd_daq_loss[i]/(1-pile_prob)
            avg_phd_acc_bot[i,j] = 1 - bot_sphd_daq_loss[i]/(1-pile_prob)
    
    if top:
        return dpes, pile_probs, avg_phd_acc_top
    else:
        return dpes, pile_probs, avg_phd_acc_bot


##################
# Area spectrums #
##################


