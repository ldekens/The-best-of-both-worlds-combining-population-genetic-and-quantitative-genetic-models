#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:10:25 2020

@author: dekens
"""


import slim_tools_fig5_10 as slim_tools




######### Simulation title ############

date = "20221020"
title = "control_fig5_10"
Nalleles = 10
Ngen = 10100 ## post burn in
Nburnin = 100
Nsim = Ngen - Nburnin

######### Working parameters ############
G = [0.1, 0.5, 1.]

a, r, theta, eta, K, Kslim, Mut, logmut, m, dt= 0.1, 1., 1., 0.5, 1., 1e4, False, -2, 0.8, 0.1

######### Replicate runs
Nreplicate = 20
path_slim_file = date + "_" + title


for g in G:
    #slim_tools.run_replicate_serie_no_plot(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, theta, dt)
    slim_tools.run_plot_upon_result_control(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, dt, time_factor)    
