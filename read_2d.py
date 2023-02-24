# Script to read in two-dimensional diagnostics
# Change units, do calculations and output netcdf files

# load modules
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
import MITgcmutils as mit
import MITgcmutils.mds as mds
import os
import calendar
from calendar import monthrange

# years
y_range = 1
# months
m_range = y_range*12

# grid
nlat = 384
nlon = 600

# Create array of month lengths

m_iter0=2997936
m_length = np.zeros((y_range,12))
day_length=144
for yr in range(0,y_range):
    yr_act = 2012 + yr
    print(yr_act)
    for m in range(0,12):
        m_length[yr,m] = monthrange(yr_act,m+1)[1]*day_length

m_length = np.ravel(m_length)

### --- read in domain information: bathymetry and ice shelf --- ###

### --- read in grid coordinates --- ###

model_lat = np.zeros((nlat,nlon))
model_lon = np.zeros((nlat,nlon))

### --- read in sea ice diagnostic --- ###
print('ice')
# BLUE
os.chdir('hol_blue/seaIce')
m_iter = m_iter0
ice_b  = np.zeros((m_range,nlat,nlon))
ice_thick_b = np.zeros((m_range,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('seaIceDiag',m_iter)
    # fractional sea ice cover - BLUE
    ice_b[m,:,:] = diag[0,:,:]
    # sea ice thickness - BLUE
    ice_thick_b[m,:,:] = diag[1,:,:]
#os.chdir()

# GREEN
os.chdir('../../hol_green/seaIce')
m_iter = m_iter0
ice_g  = np.zeros((m_range,nlat,nlon))
ice_thick_g = np.zeros((m_range,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('seaIceDiag',m_iter)
    # fractional sea ice cover - GREEN
    ice_g[m,:,:] = diag[0,:,:]
    # sea ice thickness - GREEN
    ice_thick_g[m,:,:] = diag[1,:,:]
os.chdir('../..')

# create 15% sea ice masks

mask_b = np.zeros((m_range,nlat,nlon))
mask_b[ ice_b < 0.15 ] = 1
mask_g = np.zeros((m_range,nlat,nlon))
mask_g[ ice_g < 0.15 ] = 1

# take average over several years
ice_b = np.reshape(ice_b,(-1,12,nlat,nlon))
ice_b = np.nanmean(ice_b,0)
ice_g = np.reshape(ice_g,(-1,12,nlat,nlon))
ice_g = np.nanmean(ice_g,0)
print(np.shape(ice_g))

ice_thick_b = np.reshape(ice_thick_b,(-1,12,nlat,nlon))
ice_thick_b = np.nanmedian(ice_thick_b,0)
ice_thick_g = np.reshape(ice_thick_g,(-1,12,nlat,nlon))
ice_thick_g = np.nanmedian(ice_thick_g,0)
print(np.shape(ice_thick_g))

# write out netcdf file for figure 4:

try: ncfile.close()
except: pass
ncfile = Dataset('./figure_4.nc', mode='w')
print(ncfile)

lat_dim = ncfile.createDimension('latD', 384)     # latitude axis
lon_dim = ncfile.createDimension('lonD', 600)    # longitude axis
month_dim = ncfile.createDimension('monthD',12) # time axis
for dim in ncfile.dimensions.items():
    print(dim)

lat = ncfile.createVariable('lat', np.float32, ('latD','lonD'))
lat.units = 'degrees_south'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('latD','lonD'))
lon.units = 'degrees_west'
lon.long_name = 'longitude'

ice_blue = ncfile.createVariable('ice_blue',np.float64,('monthD','latD','lonD'))
ice_blue.units = ''
ice_blue.long_name = 'Ice cover in BLUE experiment'
print(ice_blue)
ice_green = ncfile.createVariable('ice_green',np.float64,('monthD','latD','lonD'))
ice_green.units = ''
ice_green.long_name = 'Ice cover in GREEN experiment'
print(ice_green)
ice_thick_blue = ncfile.createVariable('ice_thick_blue',np.float64,('monthD','latD','lonD'))
ice_thick_blue.units = 'm'
ice_thick_blue.long_name = 'Ice thickness in BLUE experiment'
print(ice_thick_blue)
ice_thick_green = ncfile.createVariable('ice_thick_green',np.float64,('monthD','latD','lonD'))
ice_thick_green.units = 'm'
ice_thick_green.long_name = 'Ice thickness in GREEN experiment'
print(ice_thick_green)
 
lat[:,:]     = model_lat
lon[:,:]     = model_lon
ice_blue[:,:,:]  = ice_b
ice_green[:,:,:] = ice_g
ice_thick_blue[:,:,:]  = ice_thick_b
ice_thick_green[:,:,:] = ice_thick_g

print(ncfile)
ncfile.close(); print('Dataset is closed')

### --- read in surface diagnostic --- ###
print('surf')
# BLUE
os.chdir('hol_blue/surf')
m_iter = m_iter0
melt_b = np.zeros((m_range,nlat,nlon))
qtot_b = np.zeros((nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('surfDiag',m_iter)
    # ice shelf meltwater flux - BLUE
    melt_b[m,:,:] = diag[1,:,:]
    # surface heat flux - BLUE
    qtot_b = qtot_b + diag[2,:,:]
#os.chdir()

# GREEN
os.chdir('../../hol_green/surf')
m_iter = m_iter0
melt_g = np.zeros((m_range,nlat,nlon))
qtot_g = np.zeros((nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('surfDiag',m_iter)
    # ice shelf meltwater flux - GREEN
    melt_g[m,:,:] = diag[1,:,:] 
    # surface heat flux - GREEN
    qtot_g = qtot_g + diag[2,:,:]
#os.chdir()

### --- read in surface heat balance diagnostic --- ###
print('surfHeat')
# BLUE
os.chdir('../../hol_blue/surfHeat')
m_iter = m_iter0
qsen_b = np.zeros((nlat,nlon))
qlat_b = np.zeros((nlat,nlon))
qlw_b  = np.zeros((nlat,nlon))
si_qsw_b = np.zeros((m_range,nlat,nlon))
qoce_b = np.zeros((nlat,nlon))
qisf_b = np.zeros((m_range,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('surfHeatDiag',m_iter)
    # sensible heat flux - BLUE
    qsen_b = qsen_b + np.multiply(diag[0,:,:],mask_b[m,:,:])
    # latent heat flux - BLUE
    qlat_b = qlat_b + np.multiply(diag[1,:,:],mask_b[m,:,:])
    # longwave heat flux - BLUE
    qlw_b  = qlw_b  + np.multiply(diag[2,:,:],mask_b[m,:,:])
    # under-ice shortwave - BLUE
    si_qsw_b[m,:,:] = diag[6,:,:]
    # open water heat flux - BLUE
    qoce_b = qoce_b + diag[7,:,:]
    # ice shelf heat flux - BLUE
    qisf_b[m,:,:] = diag[12,:,:]
#os.chdir()

# GREEN
os.chdir('../../hol_green/surfHeat')
m_iter = m_iter0
qsen_g = np.zeros((nlat,nlon))
qlat_g = np.zeros((nlat,nlon))
qlw_g  = np.zeros((nlat,nlon))
si_qsw_g = np.zeros((m_range,nlat,nlon))
qoce_g = np.zeros((nlat,nlon))
qisf_g = np.zeros((m_range,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('surfHeatDiag',m_iter)
    # sensible heat flux - GREEN
    qsen_g = qsen_g + np.multiply(diag[0,:,:],mask_g[m,:,:])
    # latent heat flux - GREEN
    qlat_g = qlat_g + np.multiply(diag[1,:,:],mask_g[m,:,:])
    # longwave heat flux - GREEN
    qlw_g  = qlw_g  + np.multiply(diag[2,:,:],mask_g[m,:,:])
    # under-ice shortwave - GREEN
    si_qsw_g[m,:,:] = diag[6,:,:]
    # open water heat flux - GREEN
    qoce_g = qoce_g + diag[7,:,:]
    # ice shelf heat flux - GREEN
    qisf_g[m,:,:] = diag[12,:,:]
os.chdir('../..')

# reshape ice shelf outputs to get annual averages

print('maximum melt rates')
print(np.nanmax(melt_b))
print(np.nanmax(melt_g))

# meltwater fluxes
melt_b = np.reshape(melt_b,(-1,12,nlat,nlon))
melt_b = np.nanmean(melt_b,1)
melt_g = np.reshape(melt_g,(-1,12,nlat,nlon))
melt_g = np.nanmean(melt_g,1)

print(np.shape(melt_g))
# heat fluxes
qisf_b = np.reshape(qisf_b,(-1,12,nlat,nlon))
qisf_b = np.nanmean(qisf_b,1)
qisf_g = np.reshape(qisf_g,(-1,12,nlat,nlon))
qisf_g = np.nanmean(qisf_g,1)

# write out netcdf file for figure 8

try: ncfile.close()
except: pass
ncfile = Dataset('./figure_8.nc', mode='w')
print(ncfile)

lat_dim = ncfile.createDimension('latD', 384)     # latitude axis
lon_dim = ncfile.createDimension('lonD', 600)    # longitude axis
#year_dim = ncfile.createDimension('yearD', 2) # time axis
for dim in ncfile.dimensions.items():
    print(dim)

lat = ncfile.createVariable('lat', np.float32, ('latD','lonD'))
lat.units = 'degrees_south'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('latD','lonD'))
lon.units = 'degrees_west'
lon.long_name = 'longitude'

melt_blue = ncfile.createVariable('melt_blue',np.float64,('latD','lonD'))
melt_blue.units = 'kg/m³/s'
melt_blue.long_name = 'Melt rate in BLUE experiment'
print(ice_blue)
melt_green = ncfile.createVariable('melt_green',np.float64,('latD','lonD'))
melt_green.units = 'kg/m³/s'
melt_green.long_name = 'Melt rate in GREEN experiment'
print(ice_green)
qisf_blue = ncfile.createVariable('qisf_blue',np.float64,('latD','lonD'))
qisf_blue.units = 'W/m²'
qisf_blue.long_name = 'Ice shelf heat flux in BLUE experiment'
print(ice_thick_blue)
qisf_green = ncfile.createVariable('qisf_green',np.float64,('latD','lonD'))
qisf_green.units = 'W/m²'
qisf_green.long_name = 'Ice shelf heat flux in GREEN experiment'
print(ice_thick_green)

lat[:,:]     = model_lat
lon[:,:]     = model_lon
melt_blue  = melt_b
melt_green = melt_g
qisf_blue  = qisf_b
qisf_green = qisf_g

print(ncfile)
ncfile.close(); print('Dataset is closed')

### --- read in shortwave diagnostics --- ###

print('light')

# BLUE
os.chdir('hol_blue/light')
m_iter = m_iter0
qsw_10m_b = np.zeros((m_range,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('lightDiag',m_iter)
    # penetrating shortwave - BLUE
    qsw_10m_b[m,:,:] = np.nansum(diag[:,1,:,:],0)

# multipy by incoming shortwave
qsw_10m_b = np.multiply(qsw_10m_b,si_qsw_b)

# GREEN
os.chdir('../../hol_green/light')
m_iter = m_iter0
qsw_10m_g = np.zeros((m_range,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('lightDiag',m_iter)
    # penetrating shortwave - GREEN
    qsw_10m_g[m,:,:] = np.nansum(diag[:,1,:,:],0)
os.chdir('../..')

# multipy by incoming shortwave
qsw_10m_g = np.multiply(qsw_10m_g,si_qsw_g)

# write out netcdf files for figures 2 & 3

try: ncfile.close()
except: pass
ncfile = Dataset('./figure_3.nc', mode='w')
print(ncfile)

lat_dim = ncfile.createDimension('latD', 384)     # latitude axis
lon_dim = ncfile.createDimension('lonD', 600)    # longitude axis
for dim in ncfile.dimensions.items():
    print(dim)

lat = ncfile.createVariable('lat', np.float32, ('latD','lonD'))
lat.units = 'degrees_south'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('latD','lonD'))
lon.units = 'degrees_west'
lon.long_name = 'longitude'

qoce = ncfile.createVariable('qoce',np.float64,('latD','lonD'))
qoce.units = 'W/m²'
qoce.long_name = 'Air-sea heat flux'
print(qoce)
qice = ncfile.createVariable('qice',np.float64,('latD','lonD'))
qice.units = 'W/m²'
qice.long_name = 'Ice-sea heat flux'
print(qice)
qsen = ncfile.createVariable('qsen',np.float64,('latD','lonD'))
qsen.units = 'W/m²'
qsen.long_name = 'Sensible heat flux'
print(qsen)
qlat = ncfile.createVariable('qlat',np.float64,('latD','lonD'))
qlat.units = 'W/m²'
qlat.long_name = 'Latent heat flux'
print(qlat)
qlw = ncfile.createVariable('qlw',np.float64,('latD','lonD'))
qlw.units = 'W/m²'
qlw.long_name = 'Longwave heat flux'
print(qlw)
qsw_10m = ncfile.createVariable('qsw_10m',np.float64,('latD','lonD'))
qsw_10m.units = 'W/m²'
qsw_10m.long_name = 'Shortwave flux at 10m depth'
print(qsw_10m)

lat[:,:]     = model_lat
lon[:,:]     = model_lon
qoce[:,:]    = (qoce_g - qoce_b)/m_range
qice[:,:]    = (-qtot_g - qoce_g)/m_range - (-qtot_b - qoce_b)/m_range
qsen[:,:]    = (qsen_g - qsen_b)/m_range
qlat[:,:]    = (qlat_g - qlat_b)/m_range
qlw[:,:]     = (qlw_g  - qlw_b)/m_range
qsw_10m[:,:] = np.nanmean(qsw_10m_g - qsw_10m_b,0)

print(ncfile)
ncfile.close(); print('Dataset is closed')

### --- read in bio-optical diagnostic --- ###

# GREEN
#os.chdir('../../hol_green/lightSurf')
#m_iter = m_iter0
#qsw_10m_g = np.zeros((m_range,nlat,nlon))
#for m in range(0,m_range):
#    m_iter = m_iter + m_length[m]
#    diag = mds.rdmds('lightDiag',m_iter)
    # penetrating shortwave - GREEN
#    qsw_10m_g[m,:,:] = np.nansum(diag[:,1,:,:],0)
#os.chdir('../..')

