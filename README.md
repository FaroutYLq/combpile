# combpile

Lanqing Yuan

Toy combinatorics model for photon pile-up and tag-along and how can we use it in S1 modeling. Notes [here](https://xe1t-wiki.lngs.infn.it/doku.php?id=lanqing:photon_pile_up_in_s1#energy_and_position_dependent_single_photon_recorded_area_spectrum).

## Results
For accessibility, results are stored in `/project2/lgrandi/yuanlq/s1_modeling_maps/results`.

| File | type | Description |
| ---- | ---- | ----------- |
| `dpes.npy`| 1darray | DPE fraction coordinates |
| `s1_nphds.npy` | 1darray | S1 number of photon detected coordinates |
| `z_slice_median.npy` | 1darray | Median value for  Z clusters |
| `phr_area_spec_x.npy `| 1darray | Area coordinate in unit of PE for photon recorded area spectrum | 
| `phd_acc_top.npy` | 3darray | Single photon acceptance in top array. axis0=z, axis1=nphd, axis2=pdpe |
| `phd_acc_bot.npy` | 3darray | Single photon acceptance in bottom array. axis0=z, axis1=nphd, axis2=pdpe |
| `phr_area_spec_top.npy` | 3darray | Single photon area spectrum in top array. axis0=z, axis1=nphd, axis2=pdpe, axis3=area|
| `phr_area_spec_bot.npy` | 3darray | Single photon area spectrum in bottom array. axis0=z, axis1=nphd, axis2=pdpe, axis3=area |