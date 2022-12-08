# Amundsen_Sea_Analysis
Python scripts to analyse output from Amundsen Sea model

Series of scripts used to plot outputs from two versions of the Amundsen Sea model developed by Naughten et al. (2022), 
with addition of biogeochemistry package BLING

First version ('BLUE') has phytoplankton growth rate set to zero.  
Second version ('GREEN') has small and large phytoplankton growth rates set as in Nilsen et al. (2020)

Comparison of GREEN outputs with BLUE outputs shows impact of chlorophyll on physical model, via changes to shortwave attenuation in the upper ocean.

Two-way coupling between MITgcm and BLING implemented for the first time via a new subroutine "bling_swfrac", called from "do_oceanic_physics", 
which at each time step updates the attenuation profile used to calculate shortwave heating in "apply_forcing".

Coupling also used in thermodynamic calculations within sea_ice_growth

For more details see the linked repository 

