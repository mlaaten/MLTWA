# MLTWA

Rudimentary python-code for the application of Multiple Lapse Time Window Analysis (MLTWA) for the calculation of seismic attenuation.

With mltwa.py the data is imported and inverted, the output is done event-wise as an npz file. Lines 14 to 62 must be adjusted accordingly.

plot_grd.py is an example script for plotting the object function of an event over different frequencies and shows how the mltwa.py export can be further processed.


## References
Initial method: 
Fehler M., Hoshiba M., Sato H. and Obara K., 1992. Separation of scattering and intrinsic attenuation for the Kanto-Tokai region, Japan, using measurements of S-wave energy versus hypocentral distance, Geophys. J Int., 108, 787–800. doi: 10.1111/j.1365-246X.1992.tb03470.x

Code was used in:
van Laaten M., Eulenfeld T. and Wegler U., 2021, Comparison of Multiple Lapse Time Window Analysis and Qopen to determine intrinsic and scattering attenuation, Geophys. J Int., 228, 913–926. doi: 10.1093/gji/ggab390.
