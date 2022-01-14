"""
Core functions for photon pile-up combinatorics model.
Lanqing Yuan, Jan 08 2022
"""

from scipy import special
import numpy as np
import scipy.integrate as integrate
import straxen
from tqdm import tqdm
from scipy.stats import binom
import math


#####################
# Utility functions #
#####################


def emg_pdf(xvals, tau=80, mu=-10, sigma=23):
    """Toy S1 pulse shape template based on exponentially modified Gaussian. 

    Args:
        xvals (1darray): time stamps in unit of ns.
        tau (int, optional): exponential decay constant for pulse shape. Defaults to 80 ns.
        mu (int, optional): mean parameter for gaussian, which trivially shift pulse. Defaults to -10 ns.
        sigma (int, optional): width of gaussian smearing. Defaults to 23 ns.

    Returns:
        (1darray): EMG pulse shape template for S1.
    """
    A = 1./(2.*tau )
    A *= np.exp( - 1.0 * ( xvals - mu -sigma**2/(2*tau))/tau )    

    mu_K = xvals - mu - sigma**2/tau
    func = special.erfc( -mu_K / (np.sqrt(2)*sigma))

    return (A *func) # normalization?


#######################
# Combinatorics model #
#######################


def P_ma_n(n, m_array, aft=0.5, top=True):
    """Probability of n phds to be found m_array in a certain array.

    Args:
        n (int): number of phds throughtout the detector.
        m_array (int): number of phds in the specified array.
        aft (float, optional): area fraction top. Defaults to 0.5
        top (bool, optional): whether or not we specify top array. Defaults to True.

    Returns:
        (float): P(m_array|n)
    """
    if top:
        p_top = math.comb(n, m_array) * aft**m_array * (1-aft)**(n-m_array)
        return p_top
    else:
        p_bot = math.comb(n, m_array) * (1-aft)**m_array * aft**(n-m_array)
        return p_bot


def E_ma_n(n, aft=0.3, top=True):
    """Expectation of n phds to be found m_array in a certain array.

    Args:
        n (int): number of phds throughtout the detector.
        aft (float, optional): area fraction top. Defaults to 0.3.
        top (bool, optional): whether or not we specify top array. Defaults to True.

    Returns:
        (float): E[m_array|n]
    """
    expectation = 0
    for m_array in range(n+1):
        expectation += P_ma_n(n, m_array, aft=aft, top=top)*m_array
        
    return expectation


def P_m_ma(m_array, m, top=True, n_top_ch=250, n_bot_ch=234):
    """Probability of m_array phds in a certain array to be found m in a certain channel.
    Assumed evenly distributed S1 pattern. For use case uneven S1 pattern, change either n_top_ch 
    or n_bot_ch to 1/occupancy of the channel of interest in a certain array (not the whole detector). 

    Args:
        m_array (int): number of phds in the specified array.
        m (int): number of phds in the channel of interest.
        top (bool, optional): whether or not we specify top array. Defaults to True.
        n_top_ch (int, optional): number of top array channels. Assign this to 1/occupancy in case of uneven S1 pattern. Defaults to 250.
        n_bot_ch (int, optional): number of bottom array channels. Assign this to 1/occupancy in case of uneven S1 pattern. Defaults to 234.

    Returns:
        (float): P(m|m_array)
    """
    if top: # top array
        p_m_ma = math.comb(m_array, m) * (1/n_top_ch)**m * (1-1/n_top_ch)**(m_array-m)
    else: # bottom array
        p_m_ma = math.comb(m_array, m) * (1/n_bot_ch)**m * (1-1/n_bot_ch)**(m_array-m)
    return p_m_ma


def E_m_n(n, top=True, aft=0.3, n_top_ch=250, n_bot_ch=234, max_m = 35):
    """Expected phd observed m in an specific channel where at least one phd is seen, given n phds in total. 

    Args:
        n (int): number of phds throughtout the detector.
        top (bool, optional): whether or not we specify top array. Defaults to True.
        aft (float, optional): area fraction top. Defaults to 0.3
        n_top_ch (int, optional): number of top array channels. Assign this to 1/occupancy in case of uneven S1 pattern. Defaults to 250.
        n_bot_ch (int, optional): number of bottom array channels. Assign this to 1/occupancy in case of uneven S1 pattern. Defaults to 234.
        max_m (int, optional): Truncation value for m-loop. We will not loop over m beyond this value for computation's sake. Defaults to 35.

    Returns:
        (float): E[m|n, m>0]
    """
    expectation = 0
    p_condition = 0
    if top:
        for m_top in range(0, n+1):
            p_ma_n = P_ma_n(n, m_top, aft=aft, top=True)
            for m in range(1, min(m_top+1, max_m)): 
                p_m_ma = P_m_ma(m_top, m, top=True, n_top_ch=n_top_ch, n_bot_ch=n_bot_ch)
                expectation += p_ma_n * p_m_ma * m
                p_condition += p_ma_n * p_m_ma

    else:
        for m_bot in range(0, n+1):
            p_ma_n = P_ma_n(n, m_bot, aft=aft, top=False)
            for m in range(1, min(m_bot+1, max_m)):
                p_m_ma = P_m_ma(m_bot, m, top=False, n_top_ch=n_top_ch, n_bot_ch=n_bot_ch)
                expectation += p_ma_n * p_m_ma * m
                p_condition += p_ma_n * p_m_ma
    
    return expectation/p_condition


def P_m_n(n, m, top=True, aft=0.3, n_top_ch=250, n_bot_ch=234):
    """Given n phds throughout the detector, the probability of seeing m photons in a specific channel.

    Args:
        n (int): number of phds throughtout the detector.
        m (int): number of phds in the channel of interest.
        top (bool, optional): whether or not we specify top array. Defaults to True.
        aft (float, optional): area fraction top. Defaults to 0.3
        n_top_ch (int, optional): number of top array channels. Assign this to 1/occupancy in case of uneven S1 pattern. Defaults to 250.
        n_bot_ch (int, optional): number of bottom array channels. Assign this to 1/occupancy in case of uneven S1 pattern. Defaults to 234.

    Returns:
        [type]: P(m|n)
    """
    prob = 0
    if top:
        for m_top in range(n+1):
            p_ma_n = P_ma_n(n, m_top, aft=aft, top=True)
            p_m_ma = P_m_ma(m_top, m, top=True, n_top_ch=n_top_ch, n_bot_ch=n_bot_ch)
            prob += p_ma_n * p_m_ma

    else:
        for m_bot in range(n+1):
            p_ma_n = P_ma_n(n, m_bot, aft=aft, top=False)
            p_m_ma = P_m_ma(m_bot, m, top=False, n_top_ch=n_top_ch, n_bot_ch=n_bot_ch) 
            prob += p_ma_n * p_m_ma
    
    return prob

###########
# Pile-up #
###########

def P_pile_m(m, tau=80, mu=-10, sigma=23, dt=10):
    """Probability of pile-up happens on a specfic photon with m-1 neighbors in the same channel. Assumed
    all photons detected are from the same physical S1.

    Args:
        m (int): number of phds in the channel of interest.
        tau (int, optional): exponential decay constant for pulse shape. Defaults to 80 ns.
        mu (int, optional): mean parameter for gaussian, which trivially shift pulse. Defaults to -10 ns.
        sigma (int, optional): width of gaussian smearing. Defaults to 23 ns.
        dt (int, optional): within this time distance two photons will be considered as piled-up. In principle, it should be very close to SPE width. Defaults to 10 ns.

    Returns:
        (float): probability of pile-up happens between m phds in same channel.
    """
    assert type(dt) == int, 'dt must be an integer!'
    assert m > 0, 'm must be larger than 0 to define pile-up!'

    ts = np.arange(-dt-200, dt+900) # time stamps in unit of ns
    arrival_time_spec = emg_pdf(ts,tau=tau, mu=mu, sigma=sigma)
    
    # Compute integral in the no pile-up region as a funciton of time of photon of interest
    int_no_pile_region = np.zeros(len(ts)-2*dt)
    for i in range(len(ts)-2*dt):
        int_no_pile_region[i] += arrival_time_spec[:i].sum() # left to pile-up window
        int_no_pile_region[i] += arrival_time_spec[2*dt+i:].sum() # right to pile-up window

    return 1-np.sum(arrival_time_spec[dt:-dt] * int_no_pile_region**(m-1))

    # approximation way to calculate p_pile_m2:
    # -200 to 900 ns for the pulse shape time range
    # F,_ = integrate.quad(lambda t: emg_pdf(t,tau=tau, mu=mu, sigma=sigma)**2, -200, 900)
    # (F = 0.004645486388382721)
    # return 2*F*dt


def P_pile_n(n, top, occupancies, degeneracies,
             tau=80, mu=-10, sigma=23, dt=10, aft=0.3, max_m = 50):
    """Given n photons throught the detector, what is the probability of seeing a photon to be overlapped.
    Note that this photon doesn't have to be from a specific channel. We are averaging over all channels in 
    a specific array. The S1 pattern is default to be not uniform.

    Args:
        n (int): number of phds throughtout the detector.
        top (bool): whether or not we specify top array. 
        occupancies (1darray): probability of one specific channel sees one photon when 1 phd is overseved in certain array.
        degeneracies (1darray): number of channels in a certain array see has the corresponding occupancy.
        tau (int, optional): exponential decay constant for pulse shape. Defaults to 80 ns.
        mu (int, optional): mean parameter for gaussian, which trivially shift pulse. Defaults to -10 ns.
        sigma (int, optional): width of gaussian smearing. Defaults to 23 ns.
        dt (int, optional): within this time distance two photons will be considered as piled-up. In principle, it should be very close to SPE width. Defaults to 10 ns.
        aft (float, optional): area fraction top. Defaults to 0.3
        max_m (int, optional): Truncation value for m-loop. We will not loop over m beyond this value for computation's sake. Defaults to 35.

    Returns:
        (float): probability of an average-model channel in certain array to be piled-up with other photons
    """
    # clean up pattern
    occupancies = occupancies[degeneracies!=0]
    degeneracies = degeneracies[degeneracies!=0]
    norm = np.sum(degeneracies * occupancies)

    no_pile_prob_n = 0
    # probability of pile-up given m photons in one channel
    p_pile_ms = np.zeros(max_m)
    for m in range(max_m):
        p_pile_m = P_pile_m(m=m, tau=tau, mu=mu, sigma=sigma, dt=dt)
        p_pile_ms[m] = p_pile_m

    for i in range(len(occupancies)):
        occupancy = occupancies[i]
        degeneracy = degeneracies[i]
        # Condition: at least one photon is seen in the channel of interest
        p_condition = 1 - P_m_n(n=n, m=0, top=top, aft=aft, n_top_ch=1/occupancy,n_bot_ch=1/occupancy)
        
        # the probability of a photon in one channel with certain occupancy seeing no pile-up
        p_clean_occ = 0 
        for m in range(1, min(max_m, n+1)):
            p_clean_m = 1 - p_pile_ms[m]
            p_m_n = P_m_n(n=n, m=m, top=top, aft=aft, n_top_ch=1/occupancy, n_bot_ch=1/occupancy)
            p_clean_occ += p_m_n * p_clean_m
        
        no_pile_prob_n += degeneracy * occupancy * p_clean_occ / p_condition / norm

    return 1-no_pile_prob_n 


#############
# Tag-along #
#############
def P_tag_m(m, tau=80, mu=-10, sigma=23, dt=10, pre_window=30, post_window=200):
    """Probability of tag-along happens on a specfic photon with m-1 neighbors in the same channel. Assumed
    all photons detected are from the same physical S1.

    Args:
        m (int): number of phds in the channel of interest.
        tau (int, optional): exponential decay constant for pulse shape. Defaults to 80 ns.
        mu (int, optional): mean parameter for gaussian, which trivially shift pulse. Defaults to -10 ns.
        sigma (int, optional): width of gaussian smearing. Defaults to 23 ns.
        dt (int, optional): within this time distance two photons will be considered as piled-up. In principle, it should be very close to SPE width. Defaults to 10 ns.
        pre_window (int, optional): this number of ns prior to trigger will be recorded in hit waveforms. Default to 30 ns.
        post_window (int, optional): this number of ns post to trigger will be recorded in hit waveforms. Default to 200 ns.

    Returns:
        (float): probability of tag-along happens between m phds in same channel.
    """
    assert type(dt) == int, 'dt must be an integer!'
    assert m > 0, 'm must be larger than 0 to define pile-up!'
    assert min(pre_window, post_window) >= dt, 'hit extension window must be larger than pile-up window!'

    ts = np.arange(-post_window-200, pre_window+900) # time stamps in unit of ns
    arrival_time_spec = emg_pdf(ts,tau=tau, mu=mu, sigma=sigma)
    
    # Compute integral in the no pile-up region as a funciton of time of photon of interest
    int_no_tag_region = np.zeros(len(ts)-pre_window-post_window)
    for i in range(len(ts)-pre_window-post_window):
        int_no_tag_region[i] += arrival_time_spec[:i].sum() # left to tag-along window
        int_no_tag_region[i] += arrival_time_spec[i+post_window-dt:i+post_window+2*dt].sum() # left to pile-up window
        int_no_tag_region[i] += arrival_time_spec[pre_window+post_window+i:].sum() # right to tag-along window

    return 1-np.sum(arrival_time_spec[post_window:-pre_window] * int_no_tag_region**(m-1))


def P_tag_n(n, top, occupancies, degeneracies,
             tau=80, mu=-10, sigma=23, dt=10, pre_window=30, post_window=200, aft=0.3, max_m = 50):
    """Given n photons throught the detector, what is the probability of seeing a photon to be in tag-along window of 
    some other photons. Note that this photon doesn't have to be from a specific channel. We are averaging over all 
    channels in a specific array. The S1 pattern is default to be not uniform.

    Args:
        n (int): number of phds throughtout the detector.
        top (bool): whether or not we specify top array. 
        occupancies (1darray): probability of one specific channel sees one photon when 1 phd is overseved in certain array.
        degeneracies (1darray): number of channels in a certain array see has the corresponding occupancy.
        tau (int, optional): exponential decay constant for pulse shape. Defaults to 80 ns.
        mu (int, optional): mean parameter for gaussian, which trivially shift pulse. Defaults to -10 ns.
        sigma (int, optional): width of gaussian smearing. Defaults to 23 ns.
        dt (int, optional): within this time distance two photons will be considered as piled-up. In principle, it should be very close to SPE width. Defaults to 10 ns.
        pre_window (int, optional): this number of ns prior to trigger will be recorded in hit waveforms. Default to 30 ns.
        post_window (int, optional): this number of ns post to trigger will be recorded in hit waveforms. Default to 200 ns.
        aft (float, optional): area fraction top. Defaults to 0.3
        max_m (int, optional): Truncation value for m-loop. We will not loop over m beyond this value for computation's sake. Defaults to 35.

    Returns:
        (float): probability of an average-model channel in certain array to tag along other photons.
    """
    # clean up pattern
    occupancies = occupancies[degeneracies!=0]
    degeneracies = degeneracies[degeneracies!=0]
    norm = np.sum(degeneracies * occupancies)

    no_tag_prob_n = 0
    # probability of tag-along given m photons in one channel
    p_tag_ms = np.zeros(max_m)
    for m in range(max_m):
        p_tag_m = P_tag_m(m=m, tau=tau, mu=mu, sigma=sigma, dt=dt, pre_window=pre_window, post_window=post_window)
        p_tag_ms[m] = p_tag_m

    for i in range(len(occupancies)):
        occupancy = occupancies[i]
        degeneracy = degeneracies[i]
        # Condition: at least one photon is seen in the channel of interest
        p_condition = 1 - P_m_n(n=n, m=0, top=top, aft=aft, n_top_ch=1/occupancy,n_bot_ch=1/occupancy)
        
        # the probability of a photon in one channel with certain occupancy is not tagging along
        p_clean_occ = 0 
        for m in range(1, min(max_m, n+1)):
            p_clean_m = 1 - p_tag_ms[m]
            p_m_n = P_m_n(n=n, m=m, top=top, aft=aft, n_top_ch=1/occupancy, n_bot_ch=1/occupancy)
            p_clean_occ += p_m_n * p_clean_m
        
        no_tag_prob_n += degeneracy * occupancy * p_clean_occ / p_condition / norm

    return 1-no_tag_prob_n 