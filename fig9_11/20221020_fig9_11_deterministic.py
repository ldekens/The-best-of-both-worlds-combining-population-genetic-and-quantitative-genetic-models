#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:10:25 2020

@author: dekens
"""


import tools_fig9_11_deterministic as tools


######### Simulation title ############

Nalleles = 50
date = "20221020"
title = "fig9_11_deterministic_%i"%Nalleles
path = date + "_" + title
Nstart = 0
Nburnin = 100
Ngen = 10100
Nsim = Ngen - Nburnin - Nstart
time_factor = 1

######### Working parameters ############
Nreplicates = 20
G = [0.1, 0.5, 1.]


####### Details of IBS files
path_slim_files = "20220222_major_locus_inf_bg_slim_replicate_20_overlap_asymetrical_%i"%Nalleles
#list_files = ['n1a_200.txt', 'n1A_200.txt', 'n1a_200.txt', 'n2A_200.txt', 'N1a_200.txt', 'N1A_200.txt', 'N2a_200.txt', 'N2A_200.txt', 'mean_segvar_1.txt', 'mean_segvar_2.txt']
list_files = ['N1a.txt', 'N1A.txt', 'N2a.txt', 'N2A.txt', 'mean_1a.txt', 'mean_1A.txt', 'mean_2a.txt', 'mean_2A.txt', 'variance_1a.txt', 'variance_1A.txt', 'variance_2a.txt', 'variance_2A.txt', 'mean_segvar_1.txt', 'mean_segvar_2.txt']

tools.create_directory(path, False)


for g in G:
    parameters = .2, 1., 1., 1e4, 1., 1/2, g, g, .8, .8, .1, time_factor #a, r, K, Kslim, theta, eta, g1, g2, m1, m2, dt, time_factor
    tools.run_maj_locus_inf_bg(parameters, (time_factor*Nsim), Ngen, (time_factor*Nstart), path, path_slim_files, list_files, Nreplicates)
    #tools.plot_upon_result(parameters, Nsim, Ngen, Nstart, path, path_slim_files, list_files, Nreplicates)
