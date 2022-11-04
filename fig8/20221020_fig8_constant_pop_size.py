#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:10:25 2020

@author: dekens
"""


import slim_tools_constant_size as slim_tools



######### Simulation title ############
date = "20221020"
title = "fig8_constant_pop_size"
Nalleles = 200
Ngen = 20100 ## post burn in
Nburnin = 100
Nsim = Ngen - Nburnin
#NSTART_deterministic = [800, 0, 320] #Nalleles =50, a=0.2
NSTART_deterministic = [1400, 0, 400] #Nalleles =200, a=0.1
######### Working parameters ############
G = [1., 0.1, 0.5]
a, r, theta, eta, K, Kslim, Mut, logmut, m, dt= 0.1, 1., 1., 0.5, 1., 1e4, False, -2, 0.8, 0.1

######### Replicate runs
Nreplicate = 10
path_slim_file = date + "_" + title
time_factor = 4
count = 0
for g in G:
    #slim_tools.run_replicate_serie_no_plot(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, theta, dt)
    Nstart_deterministic = NSTART_deterministic[count]
    count = count +1
    workdir_deterministic = '20220927_major_locus_inf_bg_slim_constant_size_deterministic_continuous_symetrical_splitting_implicit_%i'%Nalleles+'/small_variance' + '_r=%4.2f'%r + '_g1=%4.2f'%g + '_g2=%4.2f'%g + '_m1=%4.2f'%m + '_m2=%4.2f'%m+ '_dt=%4.3f'%dt+ '_time_factor=%i'%time_factor +'_Nstart=%i'%Nstart_deterministic+'_Nz=211' 
    slim_tools.run_plot_upon_result(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, dt, time_factor, workdir_deterministic, Nstart_deterministic)
    
