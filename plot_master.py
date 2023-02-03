# master script to plot all figures for paper

# import libraries
import matplotlib as mpl
import matplotlib.pyplot as mpl
import matplotlib.colors as colors
import xarray as xr
import numpy as np
import cmocean

# set figure parameters
std_font = 20

### --- FIGURE 1 --- ###
### ------ (a) Incoming shortwave radiation
### ------ (b) Pine Island Bay bathymetry and ice shelf thickness

# read in netcdf file:
fig_1 = xr.open_dataset('figure_1.nc')

qsw_atm   = fig_1.qsw_atm.values
bathy     = fig_1.bathy.values
ice_draft = fig_1.ice_draft.values

# Set ice shelf mask for whole domain using condition ice_draft > 0

# Plot incoming shortwave radiation, with ice shelf mask

# Plot ice shelf topography with colourmap

# Plot bathymetry with contours

### ---------------- ###

### --- FIGURE 2 --- ###
### ------ (a) Shortwave flux at 10m depth
### ------ (b) Longwave flux at surface
### ------ (c) Sensible heat flux at surface
### ------ (d) Latent heat flux at surface

# read in netcdf file
fig_2 = xr.open_dataset('figure_2.nc')

qsw_10m   = fig_2.qsw_10m.values
qlw       = fig_2.qlw.values
qsen      = fig_2.qsen.values
qlat      = fig_2.qlat.values

# Plot anomaly in 10m shortwave flux

# Plot anomaly in surface longwave flux

# Plot anomaly in sensible heat flux

# Plot anomaly in latent heat flux

### ---------------- ###

### --- FIGURE 3 --- ###
### ------ (a) Ocean-atmosphere heat flux
### ------ (b) Ocean-ice heat flux

# read in netcdf file
fig_3 = xr.open_dataset('figure_3.nc')

qoce      = fig_3.qoce.values
qice      = fig_3.qice.values

# Plot anomaly in ocean-atmosphere heat flux

# Plot anomaly in ocean-ice heat flux

### ---------------- ###

### --- FIGURE 4 --- ###
### ------ (a) Sea ice cover time series, external
### ------ (b) Sea ice cover time series,within polynya

# read in netcdf file
fig_4 = xr.open_dataset('figure_4.nc')

ice_ext_blue  = fig_4.ice_ext_blue.values
ice_ext_green = fig_4.ice_ext_green.values
ice_pib_blue  = fig_4.ice_pib_blue.values
ice_pib_green = fig_4.ice_pib_green.values
ice_obs       = fig_4.ice_obs.values

# Plot time series away from polynya(s)

# Plot time series within polynya

### ---------------- ###

### --- FIGURE 5 --- ### 
### ------ (a) Chlorophyll concentrations in Pine Island Polynya
### ------ (b) Euphotic depths in Pine Island Polynya

# read in netcdf file
fig_5 = xr.open_dataset('figure_5.nc')

chl_mod   = fig_5.chl_mod.values
chl_obs   = fig_5.chl_obs.values
zeu_mod   = fig_5.zeu_mod.values
zeu_obs   = fig_5.zeu_obs.values

# Plot modelled chlorophyll distribution

# Overlay chlorophyll observations

# Plot modelled euphotic depths

# Overlay euphotic depth observations

### ---------------- ###

### --- FIGURE 6 --- ###
### ------ (a) Heat budget 0-10m
### ------ (b) Heat budget 10-30m
### ------ (c) Heat budget 30-70m
### ------ (d) Heat budget 70-150m
### ------ (e) Heat budget > 150m

# read in netcdf file
fig_6 = xr.open_dataset('figure_6.nc')

solr_blue  = fig_6.solr_blue.values
solr_green = fig_6.solr_green.values
vert_blue  = fig_6.vert_blue.values
vert_green = fig_6.vert_green.values
hori_blue  = fig_6.hori_blue.values
hori_green = fig_6.hori_green.values

# Plot time series for 0-10m

# Plot time series for 10-30m

# Plot time series for 30-70m

# Plot time series for 70-150m

# Plot time series for > 150m

### ---------------- ###

### --- FIGURE 7 --- ###
### ------ (a) Hovmoller temperature plot with isotherms
### ------ (b) Hovmoller salinity plot with isohalines

# read in netcdf file
fig_7 = xr.open_dataset('figure_7.nc')

temp = fig_7.temp.values
isot = fig_7.isot.values
salt = fig_7.salt.values
isoh = fig_7.isoh.values

# Plot hovmoller for temperature

# Overlay isotherm

# Plot hovmoller for salt

# Overlay isohaline

### ---------------- ###

### --- FIGURE 8 --- ###
### ------ (a) Spatial distribution of melt rate anomaly
### ------ (b) Pine Island Ice Shelf melt rate time series

# read in netcdf file
fig_8 = xr.open_dataset('figure_8.nc')

melt_blue  = fig_8.melt_blue.values
melt_green = fig_8.melt_green.values

# Calculate melt rate anomaly

# Plot distributions of anomaly

# Integrate horizontally

# Plot time series of melt rate in each expt

### ---------------- ###
