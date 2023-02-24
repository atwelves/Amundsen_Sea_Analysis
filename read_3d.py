# 23.02.23

# Script to read in three-dimensional diagnostics
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
import matplotlib as mpl
import matplotlib.pyplot as plt

# years
y_range = 5
# months
m_range = y_range*12

# grid
nlat = 20
nlon = 20
ndep = 50

# Pine Island Bay
pib_y_min = 30
pib_y_max = 50
pib_x_min = 350
pib_x_max = 370

# Create array of month lengths

m_iter0=2787552
m_length = np.zeros((y_range,12))
day_length=144
for yr in range(0,y_range):
    yr_act = 2008 + yr
    print(yr_act)
    for m in range(0,12):
        m_length[yr,m] = monthrange(yr_act,m+1)[1]*day_length

m_length = np.ravel(m_length)

# calculate conversion factor for W/m³ to deg.C/day 

# specific heat capacity of water
cp  = 4000
# density of water
rho = 1000
# seconds in day
day = 86400
conv_fac = day/(rho*cp)

### --- read in domain information: bathymetry and ice shelf --- ###

### --- read in grid coordinates --- ###

model_lat = np.zeros((nlat,nlon))
model_lon = np.zeros((nlat,nlon))

drf = mds.rdmds('DRF')

drf = np.repeat(drf,nlat,1)
drf = np.repeat(drf,nlon,2)
drf = np.repeat(drf[np.newaxis,:,:,:],m_range,0)
print(np.shape(drf))

rac = mds.rdmds('RAC')

rac = np.repeat(rac[np.newaxis,pib_y_min:pib_y_max,pib_x_min:pib_x_max],ndep,0)
rac = np.repeat(rac[np.newaxis,:,:,:],m_range,0)
print(np.shape(rac))

cell_vol = np.multiply(drf,rac)

### --- read in shortwave diagnostics --- ###

print('light')

# BLUE
os.chdir('hol_blue/light')
m_iter = m_iter0
swfrac_b = np.zeros((m_range,ndep,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('lightDiag',m_iter)
    # shortwave fractions - BLUE
    swfrac_b[m,:,:,:] = np.nansum(diag[:,:,pib_y_min:pib_y_max,pib_x_min:pib_x_max],0)

# Find fraction absorbed at each level
swfrac_b[:,0:ndep-1,:,:] = np.diff(swfrac_b,axis=1)
# Convert from W/m² to W/m³
swfrac_b = np.divide(swfrac_b,drf)

# GREEN
os.chdir('../../hol_green/light')
m_iter = m_iter0
swfrac_g = np.zeros((m_range,ndep,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('lightDiag',m_iter)
    # shortwave fractions - GREEN
    swfrac_g[m,:,:,:] = np.nansum(diag[:,:,pib_y_min:pib_y_max,pib_x_min:pib_x_max],0)

# Find fraction absorbed at each level
swfrac_g[:,0:ndep-1,:,:] = np.diff(swfrac_g,axis=1)
# Convert from W/m² to W/m³
swfrac_g = np.divide(swfrac_g,drf)

### --- read in vertical heat flux diagnostics --- ###

print('budget')

# BLUE
os.chdir('../../hol_blue/heat')
m_iter = m_iter0
flx_b = np.zeros((m_range,ndep+1,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('heatDiag',m_iter)
    # vertical heat fluxes - BLUE
    flx_b[m,0:ndep,:,:] = np.nansum(diag[:,:,pib_y_min:pib_y_max,pib_x_min:pib_x_max],0)

# GREEN
os.chdir('../../hol_green/heat')
m_iter = m_iter0
flx_g = np.zeros((m_range,ndep+1,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('heatDiag',m_iter)
    # vertical heat fluxes - GREEN
    flx_g[m,0:ndep,:,:] = np.nansum(diag[:,:,pib_y_min:pib_y_max,pib_x_min:pib_x_max],0)

# Find balance at each level
flux_b = np.diff(flx_b,axis=1)
flux_g = np.diff(flx_g,axis=1)

### --- read in temperature and salinity diagnostics --- ###

print('total')

# BLUE
os.chdir('../../hol_blue/phys')
m_iter = m_iter0
tend_b = np.zeros((m_range,ndep,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('physDiag',m_iter)
    # vertical heat fluxes - BLUE
    tend_b[m,:,:,:] = diag[5,:,pib_y_min:pib_y_max,pib_x_min:pib_x_max]

# GREEN
os.chdir('../../hol_green/phys')
m_iter = m_iter0
tend_g = np.zeros((m_range,ndep,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('physDiag',m_iter)
    # vertical heat fluxes - GREEN
    tend_g[m,:,:,:] = diag[5,:,pib_y_min:pib_y_max,pib_x_min:pib_x_max]

# eliminate terms under ice shelf and land
for k in range(0,ndep):
    swfrac_b[:,k,:,:][tend_b[:,0,:,:]==0] = np.nan
    swfrac_g[:,k,:,:][tend_b[:,0,:,:]==0] = np.nan
    flx_b[:,k,:,:][tend_b[:,0,:,:]==0] = np.nan
    flx_g[:,k,:,:][tend_b[:,0,:,:]==0] = np.nan
    tend_b[:,k,:,:][tend_b[:,0,:,:]==0] = np.nan
    tend_g[:,k,:,:][tend_b[:,0,:,:]==0] = np.nan

# remove bathymetry

for k in range (1,ndep):
    swfrac_b[:,k,:,:][tend_b[:,k,:,:]==0] = np.nan
    swfrac_g[:,k,:,:][tend_b[:,k,:,:]==0] = np.nan
    flx_b[:,k,:,:][tend_b[:,k,:,:]==0] = np.nan
    flx_g[:,k,:,:][tend_b[:,k,:,:]==0] = np.nan
    tend_b[:,k,:,:][tend_b[:,k,:,:]==0] = np.nan
    tend_g[:,k,:,:][tend_b[:,k,:,:]==0] = np.nan

### --- read in surface diagnostics --- ###

print('surf')

# BLUE
os.chdir('../../hol_blue/surf')
m_iter = m_iter0
tflx_b = np.zeros((m_range,nlat,nlon))
qsw_b = np.zeros((m_range,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('surfDiag',m_iter)
    # surface heat flux - BLUE
    tflx_b[m,:,:] = diag[2,pib_y_min:pib_y_max,pib_x_min:pib_x_max]
    # surface shortwave - BLUE
    qsw_b[m,:,:]  = diag[5,pib_y_min:pib_y_max,pib_x_min:pib_x_max]

# reshape
qsw_b = -np.repeat(qsw_b[:,np.newaxis,:,:],ndep,1)

# GREEN
os.chdir('../../hol_green/surf')
m_iter = m_iter0
tflx_g = np.zeros((m_range,nlat,nlon))
qsw_g = np.zeros((m_range,nlat,nlon))
for m in range(0,m_range):
    m_iter = m_iter + m_length[m]
    diag = mds.rdmds('surfDiag',m_iter)
    # surface heat flux - GREEN
    tflx_g[m,:,:] = diag[2,pib_y_min:pib_y_max,pib_x_min:pib_x_max]
    # surface shortwave - GREEN
    qsw_g[m,:,:]  = diag[5,pib_y_min:pib_y_max,pib_x_min:pib_x_max]

os.chdir('../..')

# reshape
qsw_g = -np.repeat(qsw_g[:,np.newaxis,:,:],ndep,1)

# write out netcdf file for figure 6

try: ncfile.close()
except: pass
ncfile = Dataset('./figure_6.nc', mode='w')
print(ncfile)

depth_dim = ncfile.createDimension('depthD', ndep)     # latitude axis
month_dim = ncfile.createDimension('monthD', m_range)    # longitude axis
for dim in ncfile.dimensions.items():
    print(dim)

#depth = ncfile.createVariable('depth', np.float32, ('depth'))
#depth.units = 'm'
#depth.long_name = 'depth'
#month = ncfile.createVariable('month', np.float32, ('monthD'))
#month.units = ''
#month.long_name = 'month'

solr_blue = ncfile.createVariable('solr_blue',np.float64,('monthD','depthD'))
solr_blue.units = 'deg.C/day'
solr_blue.long_name = 'shortwave heating in BLUE experiment'
print(solr_blue)
solr_green = ncfile.createVariable('solr_green',np.float64,('monthD','depthD'))
solr_green.units = 'deg.C/day'
solr_green.long_name = 'shortwave heating in GREEN experiment'
print(solr_green)
vert_blue = ncfile.createVariable('vert_blue',np.float64,('monthD','depthD'))
vert_blue.units = 'deg.C/day'
vert_blue.long_name = 'vertical heating in BLUE experiment'
print(vert_blue)
vert_green = ncfile.createVariable('vert_green',np.float64,('monthD','depthD'))
vert_green.units = 'deg.C/day'
vert_green.long_name = 'vertical heating in GREEN experiment'
print(vert_green)
totl_blue = ncfile.createVariable('totl_blue',np.float64,('monthD','depthD'))
totl_blue.units = 'deg.C/day'
totl_blue.long_name = 'total heating in BLUE experiment'
print(vert_blue)
totl_green = ncfile.createVariable('totl_green',np.float64,('monthD','depthD'))
totl_green.units = 'deg.C/day'
totl_green.long_name = 'total heating in GREEN experiment'
print(vert_green)

#depth[:,:]   = model_lat
#month[:,:]   = 
solr_blue[:,:]  = np.nanmean(np.nanmean(np.multiply(swfrac_b,qsw_b),3),2)*conv_fac
solr_green[:,:] = np.nanmean(np.nanmean(np.multiply(swfrac_g,qsw_g),3),2)*conv_fac
vert_blue[:,:]  = np.nanmean(np.nanmean(np.divide(flux_b,cell_vol),3),2)*day
vert_green[:,:] = np.nanmean(np.nanmean(np.divide(flux_g,cell_vol),3),2)*day
totl_blue[:,:]  = np.nanmean(np.nanmean(tend_b,3),2)
totl_green[:,:] = np.nanmean(np.nanmean(tend_g,3),2)

print(ncfile)
ncfile.close(); print('Dataset is closed')

data = xr.open_dataset('figure_6.nc')
solr_blue = data.solr_blue.values
solr_green = data.solr_green.values
vert_blue = data.vert_blue.values
vert_green = data.vert_green.values
totl_blue = data.totl_blue.values
totl_green = data.totl_green.values

x = np.linspace(1,m_range,m_range)
fig = plt.figure
pbar = plt.bar(x,np.nanmean(totl_green[:,11:15],1)-np.nanmean(totl_blue[:,11:15],1))
plin = plt.plot(x,np.nanmean(vert_green[:,11:15],1)-np.nanmean(vert_blue[:,11:15],1),'g')
plin = plt.plot(x,np.nanmean(totl_green[:,11:15],1)-np.nanmean(vert_green[:,11:15],1)-np.nanmean(totl_blue[:,11:15],1)+np.nanmean(vert_blue[:,11:15],1),'b')
plt.show()

