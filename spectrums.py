"""
Area and amplitude spectrum for photon recorded in different pile-up fraction and dpe fraction.
Lanqing Yuan, Jan 10 2022
"""

import numpy as np
import straxen
import pandas as pd
import sys
from tqdm import tqdm


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

    if type(dpes) != np.ndarray:
        dpes = np.array([dpes])
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
    if type(dpes) != np.ndarray:
        dpes = np.array([dpes])
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


def get_avg_phd_acc(top, adc_threshold=15, pile_probs=np.linspace(0,0.4,100), tag_probs=np.linspace(0,0.8,100), 
                    shift = 99, trunc_bound=[-10,400], dpes=np.linspace(0.18,0.24,100),
                    spe_amps_path='/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'):
    """From LED calibration result, get the average phd acceptance for case of different dpe fraction and pile-up 
    fraction. When pile-up happens, we think the phd acceptance to be 1. On the other hand, we will use the acceptance
    from SPhD amplitude spectrum. 
    Args:
        top (bool): whether or not we specify top array. 
        adc_threshold (int): below this amplitude in unit of ADC the photon will be dropped by DAQ.
        pile_probs (1darray or float, optional): probability of photon pile-up happens. Defaults to np.linspace(0,0.4,100).
        tag_probs (1darray or float, optional): probability of photon tag-along happens. Defaults to np.linspace(0,0.8,100).
        shift (int, optional): shift in number of indicies in self convolution to make sure DPE has mean exactly as twice of SPE. Defaults to 99.
        trunc_bound (list, optional): ADC range that we keep the spectrums. Defaults to [-10,400].
        dpes (1darray or float, optional): DPE fraction. Defaults to np.linspace(0.18,0.24,100).
        spe_amps_path (str, optional): the LED calibrated SPE amplitude spectrum to load. Defaults to '/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'.

    Returns:
        dpes (1darray): DPE fraction.
        pile_probs (1darray): probability of photon pile-up happens.
        tag__probs (1darray): probability of photon tag-along happens.
        avg_phd_acc (3darray): axis0=dpes, axis1=pile_probs, axis2=tag_probs. Average acceptance of a PhD in a certain array with certain p_dpe and pile-up fraction.
    """
    sphd_amps_indices, sphd_amps = get_avg_sphd_amp(top=top,shift = shift,trunc_bound=trunc_bound, dpes=dpes,
                                                       spe_amps_path=spe_amps_path)
    
    sphd_daq_loss = np.sum(sphd_amps[:,:np.where(sphd_amps_indices==adc_threshold)[0][0]+1], axis=1)

    if type(dpes) != np.ndarray:
        dpes = np.array([dpes])
    if type(pile_probs) != np.ndarray:
        pile_probs = np.array([pile_probs])
    if type(tag_probs) != np.ndarray:
        tag_probs = np.array([tag_probs])
    avg_phd_acc = np.zeros((len(dpes), len(pile_probs), len(tag_probs)))

    for i,dpe in enumerate(dpes):
        for j,pile_prob in enumerate(pile_probs):
            for k,tag_prob in enumerate(tag_probs):
                avg_phd_acc[i,j,k] = pile_prob + (1-pile_prob) * (1-sphd_daq_loss[i] + sphd_daq_loss[i]*tag_prob)

    return dpes, pile_probs, tag_probs, avg_phd_acc


##################
# Area spectrums #
##################


def get_avg_spe_area(top, trunc_bound=[-1,4.99], spe_areas_path='/dali/lgrandi/giovo/XENONnT/Utility/SPEshape/20210713/old/alt_2_default/df_spe_shape_20210713_alt.csv'):
    """From LED calibration result get the average SPE area spectrum for either array.

    Args:
        top (bool): whether or not we specify top array. 
        trunc_bound (list, optional): area range in unit of PE that we keep the spectrums. Please put None if you don't want any truncation. Defaults to [-1,4.99].
        spe_areas_path (str, optional): the LED calibrated SPE areaspectrum to load. Defaults to '/dali/lgrandi/giovo/XENONnT/Utility/SPEshape/20210713/old/alt_2_default/df_spe_shape_20210713_alt.csv'

    Returns:
        spe_areas_indices(1darray): coordinate of area spectrum in unit of PE.
        avg_spe_areas(1darray): probability density of spe in a certaint array to have area in certain PE. 
    """
    spe_areas = pd.read_csv(spe_areas_path, index_col=0)
    spe_areas_indices = np.array(spe_areas.index)
    spe_areas_indices = np.around(spe_areas_indices,decimals=2)
    spe_areas = np.array(spe_areas).transpose() # now spe_areas has shape (494, 600)

    # normalization
    spe_areas_norm = np.zeros(np.shape(spe_areas))
    for i in range(494):
        spe_areas_norm[i] = spe_areas[i]/spe_areas[i].sum()
        if np.isnan(spe_areas_norm[i]).any():
            spe_areas_norm[i] = 0
            
    spe_areas_top = np.mean(spe_areas_norm[:253], axis=0)
    spe_areas_bot = np.mean(spe_areas_norm[253:], axis=0)
    spe_areas_top = spe_areas_top/spe_areas_top.sum()
    spe_areas_bot = spe_areas_bot/spe_areas_bot.sum()

    if trunc_bound != None:
        area_range = np.arange(trunc_bound[0], trunc_bound[1],0.01)
        spe_areas_top = spe_areas_top[np.where(spe_areas_indices==trunc_bound[0])[0][0]:np.where(spe_areas_indices==trunc_bound[1])[0][0]]
        spe_areas_bot = spe_areas_bot[np.where(spe_areas_indices==trunc_bound[0])[0][0]:np.where(spe_areas_indices==trunc_bound[1])[0][0]]
        spe_areas_indices = area_range

    if top:
        return spe_areas_indices, spe_areas_top
    else:
        return spe_areas_indices, spe_areas_bot


def get_avg_dpe_area(top, shift = 100, trunc_bound=[-1,4.99], 
                     spe_areas_path='/dali/lgrandi/giovo/XENONnT/Utility/SPEshape/20210713/old/alt_2_default/df_spe_shape_20210713_alt.csv'):
    """From self-convolutded LED calibration result get the average SPE area spectrum for either array. Note that we are assuming the DPE 
    mechanism that makes DPE equivalent to just two independent SPEs.

    Args:
        top (bool): whether or not we specify top array. 
        shift (int, optional): shift in number of indicies in self convolution to make sure DPE has mean exactly as twice of SPE. Defaults to 100.
        trunc_bound (list, optional): area range in unit of PE that we keep the spectrums. Please put None if you don't want any truncation. Defaults to [-1,4.99].
        spe_areas_path (str, optional): the LED calibrated SPE areaspectrum to load. Defaults to '/dali/lgrandi/giovo/XENONnT/Utility/SPEshape/20210713/old/alt_2_default/df_spe_shape_20210713_alt.csv'

    Returns:
        dpe_areas_indices(1darray): coordinate of area spectrum in unit of PE.
        avg_dpe_areas(1darray): probability density of dpe in a certaint array to have area in certain PE. 
    """
    spe_areas_indices, spe_areas_top = get_avg_spe_area(True, trunc_bound=None, spe_areas_path=spe_areas_path)
    spe_areas_indices, spe_areas_bot = get_avg_spe_area(False, trunc_bound=None, spe_areas_path=spe_areas_path)
    
    dpe_areas_indices = np.linspace(spe_areas_indices[0]-shift*0.01, 
                                    spe_areas_indices[-1]+(len(spe_areas_indices)-shift-1)*0.01,
                                    len(spe_areas_indices)*2-1)
    # normalization
    dpe_areas_top = np.convolve(spe_areas_top, spe_areas_top, 'full')
    dpe_areas_top = dpe_areas_top/dpe_areas_top.sum()
    dpe_areas_bot = np.convolve(spe_areas_bot, spe_areas_bot, 'full')
    dpe_areas_bot = dpe_areas_bot/dpe_areas_bot.sum()
    assert (abs(np.sum(dpe_areas_indices*dpe_areas_top) - 2*np.sum(spe_areas_indices*spe_areas_top))<0.01
            and abs(np.sum(dpe_areas_indices*dpe_areas_top) == 2*np.sum(spe_areas_indices*spe_areas_top))<0.01
            ), 'The input shift is wrong! It does not allow mean DPE equal to twice mean SPE.'

    if trunc_bound != None:
        area_range = np.arange(trunc_bound[0], trunc_bound[1],0.01)
        dpe_areas_top = dpe_areas_top[np.where(dpe_areas_indices==trunc_bound[0])[0][0]:np.where(dpe_areas_indices==trunc_bound[1])[0][0]]
        dpe_areas_bot = dpe_areas_bot[np.where(dpe_areas_indices==trunc_bound[0])[0][0]:np.where(dpe_areas_indices==trunc_bound[1])[0][0]]
        dpe_areas_indices = area_range
    
    if top:
        return dpe_areas_indices, dpe_areas_top
    else:
        return dpe_areas_indices, dpe_areas_bot


def get_avg_sphd_area(top, shift = 100, trunc_bound=[-1,4.99], dpes=np.linspace(0.18,0.24,100),
                      spe_areas_path='/dali/lgrandi/giovo/XENONnT/Utility/SPEshape/20210713/old/alt_2_default/df_spe_shape_20210713_alt.csv'):
    """From LED calibration result get the average SPhD area spectrum as a weighted sum of SPE and DPE spectrum with dpe fraction
    as weights. This will serve as the spectrum of a PhD piled up with another.

    Args:
        top (bool): whether or not we specify top array. 
        shift (int, optional): shift in number of indicies in self convolution to make sure DPE has mean exactly as twice of SPE. Defaults to 100.
        trunc_bound (list, optional): area range in unit of PE that we keep the spectrums. Please put None if you don't want any truncation. Defaults to [-1,4.99].
        dpes (1darray or float, optional): DPE fraction. Defaults to np.linspace(0.18,0.24,100).
        spe_areas_path (str, optional): the LED calibrated SPE areaspectrum to load. Defaults to '/dali/lgrandi/giovo/XENONnT/Utility/SPEshape/20210713/old/alt_2_default/df_spe_shape_20210713_alt.csv'
    
    Returns:
        sphd_areas_indices(1darray): coordinate of area spectrum in unit of PE.
        avg_sphd_areas(1darray): probability density of sphd in a certaint array to have area in certain PE. 
    """
    spe_areas_indices, spe_areas_top = get_avg_spe_area(True, trunc_bound=trunc_bound, spe_areas_path=spe_areas_path)
    spe_areas_indices, spe_areas_bot = get_avg_spe_area(False, trunc_bound=trunc_bound, spe_areas_path=spe_areas_path)
    dpe_areas_indices, dpe_areas_top = get_avg_dpe_area(True, shift=shift, trunc_bound=trunc_bound, spe_areas_path=spe_areas_path)
    dpe_areas_indices, dpe_areas_bot = get_avg_dpe_area(False, shift=shift, trunc_bound=trunc_bound, spe_areas_path=spe_areas_path)

    sphd_areas_indices = np.arange(trunc_bound[0], trunc_bound[1], 0.01)
    sphd_areas_top = np.zeros((len(dpes), len(sphd_areas_indices)))
    sphd_areas_bot = np.zeros((len(dpes), len(sphd_areas_indices)))    
    
    if type(dpes) != np.ndarray:
        dpes = np.array([dpes])
    for i,dpe in enumerate(dpes):
        sphd_areas_top[i] = spe_areas_top*(1-dpe) + dpe_areas_top*dpe
        sphd_areas_top[i] = sphd_areas_top[i]/sphd_areas_top[i].sum() # normallization for security
        sphd_areas_bot[i] = spe_areas_bot*(1-dpe) + dpe_areas_bot*dpe
        sphd_areas_bot[i] = sphd_areas_bot[i]/sphd_areas_bot[i].sum() # normallization for security

    if top:
        return sphd_areas_indices, sphd_areas_top
    else:
        return sphd_areas_indices, sphd_areas_bot


def get_avg_sphr_area(top, sphr_areas_path='/project2/lgrandi/yuanlq/shared/s1_modeling_maps/middle_steps/'):
    """From argon S1 get single photon recorded spectrum without pile-up.

    Args:
        top (bool): whether or not we specify top array. 
        sphr_areas_path (str, optional): the path to ar37 S1 based SPhR area spectrum to load.  Defaults to '/home/yuanlq/xenon/combpile/maps/'.

    Returns:
        sphr_areas_indices(1darray): coordinate of area spectrum in unit of PE.
        avg_sphr_areas(1darray): probability density of sphr without pile-up in a certaint array to have area in certain PE. 
    """
    sphr_areas_top = np.load(sphr_areas_path+'sphr_areas_top.npy')
    sphr_areas_bot = np.load(sphr_areas_path+'sphr_areas_bot.npy')
    sphr_areas_indices = np.load(sphr_areas_path+'sphr_areas_indices.npy')

    if top:
        return sphr_areas_indices, sphr_areas_top
    else:
        return sphr_areas_indices, sphr_areas_bot


def get_avg_phr_area(top, shift = 100, trunc_bound_pe=[-1,4.99], trunc_bound_adc=[-10,400], adc_threshold=15, 
                     dpes=np.linspace(0.18,0.24,100), pile_probs=np.linspace(0,0.4,100), tag_probs=np.linspace(0,0.8,100), 
                     spe_amps_path='/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz',
                     spe_areas_path='/dali/lgrandi/giovo/XENONnT/Utility/SPEshape/20210713/old/alt_2_default/df_spe_shape_20210713_alt.csv',
                     sphr_areas_path='/project2/lgrandi/yuanlq/shared/s1_modeling_maps/middle_steps/'):
    """From LED calibration and ar37 S1 data get the area spectrum of PhR in different pile-up fraction and dpes.

    Args:
        top (bool): whether or not we specify top array. 
        shift (int, optional): shift in number of indicies in self convolution to make sure DPE has mean exactly as twice of SPE. Defaults to 100.
        trunc_bound_pe (list, optional): area range in unit of PE that we keep the spectrums. Please put None if you don't want any truncation. Defaults to [-1,4.99].
        trunc_bound_adc (list, optional): ADC range that we keep the spectrums. Please put None if you don't want any truncation. Defaults to [-10,400].
        adc_threshold (int): below this amplitude in unit of ADC the photon will be dropped by DAQ.
        dpes (1darray or float, optional): DPE fraction. Defaults to np.linspace(0.18,0.24,100).
        pile_probs (1darray or float, optional): probability of photon pile-up happens. Defaults to np.linspace(0,0.4,100).
        tag_probs (1darray or float, optional): probability of photon tag-along happens. Defaults to np.linspace(0,0.8,100).
        spe_amps_path (str, optional): the LED calibrated SPE amplitude spectrum to load. Defaults to '/dali/lgrandi/led_calibration/SPE_acceptance/20210713/spe_025420_025418.npz'.
        spe_areas_path (str, optional): the LED calibrated SPE area spectrum to load. Defaults to '/dali/lgrandi/giovo/XENONnT/Utility/SPEshape/20210713/old/alt_2_default/df_spe_shape_20210713_alt.csv'
        sphr_areas_path (str, optional): the path to ar37 S1 based SPhR area spectrum to load. Defaults to '/home/yuanlq/xenon/combpile/maps/'.

    Returns:
        dpes (1darray): DPE fraction. 
        pile_probs (1darray): probability of photon pile-up happens.
        tag_probs (1darray or float, optional): probability of photon tag-along happens. 
        avg_phr_areas(4darray): axis0=dpes, axis1=pile_probs, axis, axis3=areas. Normalized are spectrums for 
    """
    # 1darray
    sphr_areas_indices, sphr_areas = get_avg_sphr_area(top,sphr_areas_path=sphr_areas_path)
    # 2darray axis0=dpes
    sphd_areas_indices, sphd_areas = get_avg_sphd_area(top,shift=shift, trunc_bound=trunc_bound_pe,dpes=dpes,spe_areas_path=spe_areas_path)
    assert sphd_areas_indices.all() == sphr_areas_indices.all(), 'The x coordinate of the SPhR and SPhD area spectrum must be the same.'

    sphd_amps_indices, sphd_amps = get_avg_sphd_amp(top=top,shift = shift,trunc_bound=trunc_bound_adc, dpes=dpes,
                                                    spe_amps_path=spe_amps_path)
    
    sphd_daq_loss = np.sum(sphd_amps[:,:np.where(sphd_amps_indices==adc_threshold)[0][0]+1], axis=1)

    if type(dpes) != np.ndarray:
        dpes = np.array([dpes])
    if type(pile_probs) != np.ndarray:
        pile_probs = np.array([pile_probs])
    if type(tag_probs) != np.ndarray:
        tag_probs = np.array([tag_probs])
    avg_phr_areas = np.zeros((len(dpes), len(pile_probs), len(tag_probs), len(sphd_areas_indices)))
    for i, dpe in tqdm(enumerate(dpes)):
        for j, pile_prob in enumerate(pile_probs):
            for k, tag_prob in enumerate(tag_probs):
                # spectrum when not piled up
                not_piled = (1-sphd_daq_loss[i]) * sphr_areas + sphd_daq_loss[i] * tag_prob * sphd_areas[i]
                not_piled = not_piled/not_piled.sum() # normalization
                avg_phr_areas[i,j,k,:] = pile_prob * sphd_areas[i] + (1-pile_prob)*not_piled
    
    return dpes, pile_probs, tag_probs, avg_phr_areas
