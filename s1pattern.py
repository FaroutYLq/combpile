"""
Functions for pre-processing s1 pattern map before feeding combinatorics model.
Lanqing Yuan, Jan 08 2022
"""


import numpy as np
import straxen
from tqdm import tqdm


###################
# get pattern map #
###################


def make_map(map_file, fmt=None, method='WeightedNearestNeighbors'):
    """Fetch and make an instance of InterpolatingMap based on map_file
    Alternatively map_file can be a list of ["constant dummy", constant: int, shape: list]
    return an instance of DummyMap"""

    if isinstance(map_file, list):
        assert map_file[0] == 'constant dummy', ('Alternative file input can only be '
                                                 '("constant dummy", constant: int, shape: list')
        return DummyMap(map_file[1], map_file[2])

    elif isinstance(map_file, str):
        if fmt is None:
            fmt = parse_extension(map_file)

        map_data = straxen.get_resource(map_file, fmt=fmt)
        return straxen.InterpolatingMap(map_data, method=method)

    else:
        raise TypeError("Can't handle map_file except a string or a list")


def parse_extension(name):
    """Get the extention from a file name. If zipped or tarred, can contain a dot"""
    split_name = name.split('.')
    if len(split_name) == 2:
        fmt = split_name[-1]
    elif len(split_name) > 2 and 'gz' in name:
        fmt = '.'.join(split_name[-2:])
    else:
        fmt = split_name[-1]
    return fmt


##############################
# coarse grained pattern map #
##############################


def inside_tpc_xy_indicies(s1_pattern_map, fv_radius=60.73, 
                           xs = np.linspace(-64.18666666666667, 64.18666666666667, 30)):
    """Find the indicies of good x and y, where inside the TPC FV radius s1 pattern map is defined.

    Args:
        s1_pattern_map (4darray): the original s1 pattern map.
        fv_radius (float, optional): Radius of fiducial volume. Defaults to 60.73 cm.
        xs (1darray, optional): values for x,y coodinates corresponding to indicies. Defaults to np.linspace(-64.18666666666667, 64.18666666666667, 30).

    Returns:
        (2darray): axis0=different pairs, axis1=x or y indicies. Pairs of good indicies inside the fiducial volume radius.
    """
    len_x, len_y = np.shape(s1_pattern_map.data['map'])[:2]
    test_z_ind = 0
    first = True
    for i in range(len_x):
        for j in range(len_y):
            s1pattern = s1_pattern_map.data['map'][i,j,test_z_ind,:]
            check = s1pattern.sum()
            if check != 0 and xs[i]**2+xs[j]**2<fv_radius**2: # in Fiducial Volume radius
                if first:
                    good_xy_indicies = np.array([i, j])
                    first = False
                else:
                    good_xy_indicies = np.vstack((good_xy_indicies, np.array([i,j]))) 
    return good_xy_indicies


def coarse_grain_pattern(s1_pattern_map, good_xy_indicies, z_range, top=True,
                         zs = np.linspace(-153.4992099, 6.526389900000001, 100),
                         occ_bins = np.linspace(0,0.025,50)):
    """Find the average channel occupancy distribution for a cylinder of volume with specified height range.

    Args:
        s1_pattern_map (4darray): the original s1 pattern map.
        good_xy_indicies (2darray): axis0=different pairs, axis1=x or y indicies. Pairs of good indicies inside the fiducial volume radius.
        z_range (tuple): (minimum height, maximum height) in unit of cm, for the volume to be averaged.
        top (bool, optional): whether or not we specify top array. Defaults to True.
        zs (1darray, optional): the coodinates in unit of cm for the height range corresponding to the s1 pattern map you feed. Defaults to np.linspace(-153.4992099, 6.526389900000001, 100).
        occ_bins (1darray, optional): bins where the histogram for occupancy in built. Defaults to np.linspace(0,0.025,50).

    Returns:
        [type]: [description]
    """
    s1_pattern = s1_pattern_map.data['map']

    z_left_index = np.digitize(z_range[0],zs)
    z_right_index = np.digitize(z_range[1],zs)+1
    len_z = z_right_index - z_left_index
    n_voxels = len_z*len(good_xy_indicies)
    
    counts_avg = np.zeros(len(occ_bins)-1)
    
    for z_ind in range(z_left_index, z_right_index):
        for xy_pair in good_xy_indicies:
            if top:
                pattern = s1_pattern[xy_pair[0],xy_pair[1],z_ind,:253]
            else:
                pattern = s1_pattern[xy_pair[0],xy_pair[1],z_ind,253:]
            pattern = pattern/pattern.sum() # normalization
            
            counts,_ = np.histogram(pattern, bins=occ_bins)
            counts = counts/counts.sum()
            counts_avg += counts/n_voxels # Assuming each voxel contribute equally
            
    return counts_avg
            

def get_pattern(z_range, top, method='WeightedNearestNeighbors', fv_radius=60.73, 
                map_file="/home/yuanlq/software/private_nt_aux_files/sim_files/XENONnT_s1_xyz_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl", 
                xs = np.linspace(-64.18666666666667, 64.18666666666667, 30), 
                zs = np.linspace(-153.4992099, 6.526389900000001, 100),
                occ_bins = np.linspace(0,0.025,50), n_top_ch=250, n_bot_ch=234):
    """Find the average occupancy distribution for all channels in a certain array. The average is done over all voxels in a certain z range.

    Args:
        z_range (tuple): (minimum height, maximum height) in unit of cm, for the volume to be averaged.
        top (bool): whether or not we specify top array. 
        method (str, optional): interpolation method for s1 pattern map. Defaults to 'WeightedNearestNeighbors'.
        fv_radius (float, optional): Radius of fiducial volume. Defaults to 60.73 cm.
        map_file (str, optional): path to s1 pattern map. . Defaults to "/home/yuanlq/software/private_nt_aux_files/sim_files/XENONnT_s1_xyz_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl".
        xs (1darray, optional): values for x,y coodinates corresponding to indicies. Defaults to np.linspace(-64.18666666666667, 64.18666666666667, 30).
        zs (1darray, optional): the coodinates in unit of cm for the height range corresponding to the s1 pattern map you feed. Defaults to np.linspace(-153.4992099, 6.526389900000001, 100).
        occ_bins (1darray, optional): bins where the histogram for occupancy in built. Defaults to np.linspace(0,0.025,50).
        n_top_ch (int, optional): number of top array channels. Assign this to 1/occupancy in case of uneven S1 pattern. Defaults to 250.
        n_bot_ch (int, optional): number of bottom array channels. Assign this to 1/occupancy in case of uneven S1 pattern. Defaults to 234.

    Returns:
        occupancies (1darray): probability of one specific channel sees one photon when 1 phd is overseved in certain array.
        degeneracies (1darray): number of channels in a certain array see has the corresponding occupancy.
    """
    s1_pattern_map = make_map(map_file=map_file, fmt=None, method=method)
    good_xy_indicies = inside_tpc_xy_indicies(s1_pattern_map=s1_pattern_map, fv_radius=fv_radius, xs = xs)
    occupancies = (occ_bins[1:] + occ_bins[:-1])/2 # occupancy values as the middle of occupancy bins
    counts_avg = coarse_grain_pattern(s1_pattern_map=s1_pattern_map, 
                                      good_xy_indicies=good_xy_indicies, 
                                      z_range=z_range, top=top, zs=zs, occ_bins=occ_bins)
    if top:
        degeneracies = np.around(counts_avg * n_top_ch)
    else:
        degeneracies = np.around(counts_avg * n_bot_ch)

    return occupancies, degeneracies