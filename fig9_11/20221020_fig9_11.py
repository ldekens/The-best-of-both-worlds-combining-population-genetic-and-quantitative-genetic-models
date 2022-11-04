#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:10:25 2020

@author: dekens
"""


import slim_tools_fig9_11 as slim_tools




######### Simulation title ############

date = "20221020"
title = "fig9_11"
Nalleles = 50
Ngen = 10100 ## post burn in
Nburnin = 100
Nsim = Ngen - Nburnin

######### Working parameters ############
G = [0.1, 0.5, 1.]
a, r, theta, eta, K, Kslim, Mut, logmut, m, dt= 0.2, 1., 1., 0.5, 1., 1e4, False, -2, 0.8, 0.1

######### Replicate runs
Nreplicate = 20

path_slim_file = date + "_" + title
time_factor = 1
for g in G:
    slim_tools.run_replicate_serie_no_plot(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, theta, dt)
    Nstart_deterministic = 0
    workdir_deterministic = '20220222_major_locus_inf_bg_slim_overlap_deterministic_continuous_asymetrical_splitting_implicit_%i'%Nalleles +'/small_variance' + '_r=%4.2f'%r + '_g1=%4.2f'%g + '_g2=%4.2f'%g + '_m1=%4.2f'%m + '_m2=%4.2f'%m+ '_dt=%4.3f'%dt+ '_time_factor=%i'%time_factor +'_Nstart=%i'%Nstart_deterministic    
    #slim_tools.run_plot_upon_result(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, dt, time_factor, workdir_deterministic, Nstart_deterministic)
    
