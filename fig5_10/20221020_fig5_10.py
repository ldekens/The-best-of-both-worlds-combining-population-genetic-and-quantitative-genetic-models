#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:10:25 2020

@author: dekens
"""


import slim_tools_fig5_10 as slim_tools



######### Simulation title ############
date = '20221020'
title = "fig5_10"


Nalleles = 200 ## number of alleles for the quantitative background

### Number of generations for IBS
Ngen = 10100 ## post burn in
Nburnin = 100
Nsim = Ngen - Nburnin


### Number of generation to skip to initialize deterministic comparisons
NSTART_deterministic = [1400, 0, 400] #for fig 5, where Nalleles =200, a=0.1
#NSTART_deterministic = [800, 0, 320] #for fig 10, where Nalleles =50, a=0.2
time_factor = 4 ### Time factor between IBS and deterministic comparisons


######### Working parameters ############
G = [0.1, 0.5, 1.]
a, r, theta, eta, K, Kslim, Mut, logmut, m, dt= 0.1, 1., 1., 0.5, 1., 1e4, False, -2, 0.8, 0.1
## a is sigma_LE, possibility to run with mutations (default is False)

######### Replicate runs
Nreplicate = 20
path_slim_file = date + '_'+ title
count = 0
for g in G:
    slim_tools.run_replicate_serie_no_plot(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, theta, dt)
    Nstart_deterministic = NSTART_deterministic[count]
    count = count +1
    workdir_deterministic = date + '_'+ title + '_deterministic_%i'%Nalleles+'/small_variance' + '_r=%4.2f'%r + '_g1=%4.2f'%g + '_g2=%4.2f'%g + '_m1=%4.2f'%m + '_m2=%4.2f'%m+ '_dt=%4.3f'%dt+ '_time_factor=%i'%time_factor +'_Nstart=%i'%Nstart_deterministic+'_Nz=211' 
    slim_tools.run_plot_upon_result(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, dt, time_factor, workdir_deterministic, Nstart_deterministic)
    