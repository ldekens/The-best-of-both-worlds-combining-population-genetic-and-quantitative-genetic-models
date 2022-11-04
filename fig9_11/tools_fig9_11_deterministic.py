#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:11:39 2022

@author: dekens
"""
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib as ml
from matplotlib import cm
import multiprocessing
from itertools import repeat
import scipy.signal as scsign
from scipy import sparse as sp
import scipy.sparse.linalg as scisplin
import time
import scipy.stats as scst
import pandas as pan

ml.rcParams['mathtext.fontset'] = 'stix'
ml.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams.update({
    "text.usetex": True})
    
viridis = cm.get_cmap('viridis', 300)

def plot_p1_p2(p1, p2, path_directory, name_variable, Nsim, xlabel, ylabel, logscale, time_factor):

    fig = plt.figure(figsize = (10, 6))
    if logscale:
        plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.set_ylabel(ylabel, fontsize = 25)
    ax.set_xlabel(xlabel, fontsize = 25)
    ax.plot(range(Nsim)[::time_factor], p1[::time_factor], color='darkblue', linewidth = 2,  label = 'Deme 1')
    ax.plot(range(Nsim)[::time_factor], p2[::time_factor], color='goldenrod', linewidth = 2, label = 'Deme 2')
    #plt.xticks(ticks = range(Nsim)[::time_factor])
    plt.legend(fontsize = 25)
    plt.savefig(path_directory+'/'+name_variable+'.png')
    plt.show()
    plt.close()
    
def plot_maj_locus_inf_bg(n1a, n1A, n2a, n2A, path_directory, name_variable, Nsim, xlabel, ylabel, logscale):

    fig = plt.figure(figsize = (10, 6))
    if logscale:
        plt.yscale('log')
    ax = fig.add_subplot(111)
    ax.set_ylabel(ylabel, fontsize = 25)
    ax.set_xlabel(xlabel, fontsize = 25)
    ax.plot(range(Nsim), n1a, color='darkblue', linewidth = 2,  label = 'Deme 1 - a')
    ax.plot(range(Nsim), n1A, color='RoyalBLue', linewidth = 2,  label = 'Deme 1 - A')
    ax.plot(range(Nsim), n2a, color='goldenrod', linewidth = 2, label = 'Deme 2 - a')
    ax.plot(range(Nsim), n2A, color='peru', linewidth = 2, label = 'Deme 2- A')

    plt.legend(fontsize = 25)
    plt.savefig(path_directory+'/'+name_variable+'.png')
    plt.show()
    plt.close()
    
def create_directory(workdir, remove):
    if (os.path.exists(workdir))&remove:
        shutil.rmtree(workdir)
    try:
    # Create target Directory
        os.mkdir(workdir)
    except FileExistsError:
        print("Directory " , workdir ,  " already exists")
    return

####### Creates grids used in the reproduction operator (double convolution - see Appendix G)
def grid_double_conv(zmax, Nz):
    z, dz = np.linspace(-zmax, zmax, Nz, retstep=True)
    zz, dzz = np.linspace(-zmax*2, zmax*2, 2*Nz-1, retstep=True)
    z2, dz2 = np.linspace(-zmax*2, zmax*2, 4*Nz-3, retstep=True)
    
    return(Nz, z, dz, zz, dzz, z2, dz2)

####### Creates a discritized Gaussian distribution with mean m and variance s**2 on the grid z
def Gauss(m, s, z):
    Nz = np.size(z)
    G = np.zeros(Nz)
    for k in range(Nz):
        G[k] = 1/( np.sqrt(2*np.pi)*s )* np.exp( - (z[k]-m)**2 / (2*s**2) )
    return(G)
    
####### Encodes the reproduction operator - double convolution
def reproduction_conv(n1, n2, N, epsilon, zmax, Nz):
    Nz, z, dz, zGauss, dzGauss, zaux, dzaux = grid_double_conv(zmax, Nz)

    Gsex = Gauss(0, epsilon/np.sqrt(2), 0.5*zGauss)
    Bconv_aux = scsign.convolve(scsign.convolve(n1, Gsex*dz)*dz, n2)
    if (N>0):
        Bconv = np.interp(z, zaux, Bconv_aux)/N
    else:
        Bconv = np.zeros(np.size(z))
    return(Bconv)

####### Compute the first four moments of a given a discrete distribution n on grid z and step dz
def moments(n, z, dz):
    N = sum(n)*dz
    m = sum(z*n)*dz/N
    v = sum((z - m)**2*n)*dz/N
    s = sum((z - m)**3/np.sqrt(v)**3*n)*dz/N
    return(N, m , v, s)
    

####### Function that implements the discrete time iterations
def update(n1a, n1A, n2a, n2A, parameters, z, dz, Nz, M_selection_A, M_selection_a, M_migration, Id):
    epsilon, r, kappa, Kslim, theta, eta, g1, g2, m1, m2, dt, time_factor = parameters
    dt=dt/time_factor
    zmax = z[-1]
    N1, N2 = sum(n1a*dz) + sum(n1A*dz), sum(n2a*dz) + sum(n2A*dz)
    
    # Reproduction terms
    B1_A = reproduction_conv(n1A, n1A, N1, epsilon, zmax, Nz) + 1/2*reproduction_conv(n1A, n1a, N1, epsilon, zmax, Nz) + 1/2*reproduction_conv(n1a, n1A, N1, epsilon, zmax, Nz)
    B1_a = reproduction_conv(n1a, n1a, N1, epsilon, zmax, Nz) + 1/2*reproduction_conv(n1A, n1a, N1, epsilon, zmax, Nz) + 1/2*reproduction_conv(n1a, n1A, N1, epsilon, zmax, Nz)
    B2_A = reproduction_conv(n2A, n2A, N2, epsilon, zmax, Nz) + 1/2*reproduction_conv(n2A, n2a, N2, epsilon, zmax, Nz) + 1/2*reproduction_conv(n2a, n2A, N2, epsilon, zmax, Nz)
    B2_a = reproduction_conv(n2a, n2a, N2, epsilon, zmax, Nz) + 1/2*reproduction_conv(n2A, n2a, N2, epsilon, zmax, Nz) + 1/2*reproduction_conv(n2a, n2A, N2, epsilon, zmax, Nz)
    
    

    B12_A = np.array((B1_A.flatten('F'), B2_A.flatten('F'))).flatten()
    B12_a = np.array((B1_a.flatten('F'), B2_a.flatten('F'))).flatten()

    n12_a = np.array((n1a.flatten('F'), n2a.flatten('F'))).flatten()
    n12_A = np.array((n1A.flatten('F'), n2A.flatten('F'))).flatten()
    

    competition = - (1+r*dt)*kappa*sp.diags( np.array([N1*np.ones(Nz).flatten(), N2*np.ones(Nz).flatten()]).flatten(), 0, (2*Nz, 2*Nz) )
    naux_A =  (Id+dt*M_migration).dot(scisplin.spsolve( (Id - dt  *(M_selection_A + competition ) ),  (n12_A + dt * r * B12_A ) ) )
    naux_a =  (Id+dt*M_migration).dot(scisplin.spsolve( (Id - dt  *(M_selection_a + competition ) ),  (n12_a + dt * r * B12_a ) ) )

    n1Anew, n2Anew = naux_A[:Nz], naux_A[Nz:]
    n1anew, n2anew = naux_a[:Nz], naux_a[Nz:]
    return(n1anew, n1Anew, n2anew, n2Anew)

######## Function that runs the whole scheme, given the initial distributions n1 and n2
def run_model_continuous(n1a, n1A, n2a, n2A, parameters, Nsim, z, Nz, dz, workdir):
    epsilon, r, kappa, Kslim, theta, eta, g1, g2, m1, m2, dt, time_factor = parameters
    
    M_selection_1_a = -g1 * (z - eta + theta)**2
    M_selection_1_A = -g1 * (z + eta + theta)**2
    M_selection_2_a = -g2 * (z - eta - theta)**2
    M_selection_2_A = -g2 * (z + eta - theta)**2
    
    M_selection_A = sp.spdiags( np.array([M_selection_1_A.flatten('F'), M_selection_2_A.flatten('F')]).flatten(), 0, 2*Nz, 2*Nz )
    M_selection_a = sp.spdiags( np.array([M_selection_1_a.flatten('F'), M_selection_2_a.flatten('F')]).flatten(), 0, 2*Nz, 2*Nz )
    
    ######## Migration matrix ###########
    M_migration = sp.coo_matrix( np.block([[-np.eye(Nz)*m1, np.eye(Nz)*m2],[np.eye(Nz)*m1,-np.eye(Nz)*m2]]) )
    
    ######### Migration - selection matrix ##########
    
    Id = sp.diags(np.ones(2*Nz), 0, (2*Nz, 2*Nz) )*1.
    
    
    moments_1a, moments_1A = np.zeros((4, Nsim)), np.zeros((4, Nsim))
    moments_2a, moments_2A = np.zeros((4, Nsim)), np.zeros((4, Nsim))
    
    for t in range(Nsim):        
        moments_1a[:, t] = moments(n1a, z, dz)
        moments_1A[:, t] = moments(n1A, z, dz)
        moments_2a[:, t] = moments(n2a, z, dz)
        moments_2A[:, t] = moments(n2A, z, dz)
        n1a, n1A, n2a, n2A = update(n1a, n1A, n2a, n2A, parameters, z, dz, Nz, M_selection_A, M_selection_a, M_migration, Id)
        if (t%50 == 0):
            print(t)
    
    
    np.save(workdir +'/n1a.npy', n1a)
    np.save(workdir +'/n1A.npy', n1A)
    np.save(workdir +'/n2a.npy', n2a)
    np.save(workdir +'/n2A.npy', n2A)
    np.save(workdir +'/moments_1a', moments_1a)
    np.save(workdir +'/moments_1A', moments_1A)
    np.save(workdir +'/moments_2a', moments_2a)
    np.save(workdir +'/moments_2A', moments_2A)
    


def run_maj_locus_inf_bg(parameters, Nsim, Ngen, Nstart, path, path_slim_files, list_files, Nreplicates):
    a, r, K, Kslim, theta, eta, g1, g2, m1, m2, dt, time_factor = parameters
    g, m = g1, m1
    subpath_slim = path_slim_files + '/' + 'm=%4.2f'%m + '_g=%4.2f'%g+'_eta=%4.2f'%eta+'_a=%4.2f'%a+'_Ngen=%i'%Ngen+'_Kslim=%i'%Kslim + '_r=%4.2f'%r + '_dt=%4.2f'%dt
    workdir = path + '/small_variance' + '_r=%4.2f'%r + '_g1=%4.2f'%g1 + '_g2=%4.2f'%g2 + '_m1=%4.2f'%m1 + '_m2=%4.2f'%m2+ '_dt=%4.3f'%dt+ '_time_factor=%i'%time_factor + '_Nstart=%i'%(Nstart/time_factor)

    create_directory(workdir, False)
    ### Discrete time deterministic model
    
    zmax, Nz = 1.2, 211
    z, dz = np.linspace(-zmax, zmax, Nz, retstep=True)
    
    #### Run determinist simulation initiated at the replicate Nstart state and save the simulation's output in a new folder indexed by the number of replicate
    pool = multiprocessing.Pool(processes = 10)
    print(Nreplicates)
    inputs = [*zip(range(Nreplicates), repeat(parameters), repeat(list_files), repeat(subpath_slim), repeat(z), repeat(dz), repeat(Nz), repeat(workdir), repeat(Nstart), repeat(Nsim))]
    pool.starmap(run_maj_locus_inf_bg_per_replicate, inputs)
    
    #run_maj_locus_inf_bg_per_replicate(16, parameters, list_files, subpath_slim, z, dz, Nz, workdir, Nstart, Nsim)
    
    #### Collect the dynamics of the allelic frequencies and moments from the ith deterministic simulation.

    
    moments_1a_all, moments_1A_all, moments_2a_all, moments_2A_all = np.zeros((4, Nsim, Nreplicates)), np.zeros((4, Nsim, Nreplicates)), np.zeros((4, Nsim, Nreplicates)), np.zeros((4, Nsim, Nreplicates))
    p1_all, p2_all = np.zeros((Nsim, Nreplicates)), np.zeros((Nsim, Nreplicates))
    
    for i in range(Nreplicates):
        #p1_all[:, i], p2_all[:, i] = collect_p1_p2_per_simulation(i+16, workdir)
        #moments_1a_all[:, :, i], moments_1A_all[:, :, i], moments_2a_all[:, :, i], moments_2A_all[:, :, i] = collect_moments_per_simulation(i+16, workdir)

        p1_all[:, i], p2_all[:, i] = collect_p1_p2_per_simulation(i, workdir)
        moments_1a_all[:, :, i], moments_1A_all[:, :, i], moments_2a_all[:, :, i], moments_2A_all[:, :, i] = collect_moments_per_simulation(i, workdir)

        
    var_p1, var_p2 = np.mean(p1_all*(1-p1_all), axis = 1), np.mean(p2_all*(1-p2_all), axis = 1)
    
    #### Save the averaged frequencies for plotting with IBS
    np.save(workdir +'/var_p1_deterministic.npy', var_p1)
    np.save(workdir +'/var_p2_deterministic.npy', var_p2)
    #### Save the averaged frequencies for plotting with IBS
    np.save(workdir +'/p1_deterministic_all.npy', p1_all)
    np.save(workdir +'/p2_deterministic_all.npy', p2_all)
    #### Save the moments
    np.save(workdir +'/moments_1a_deterministic_all.npy', moments_1a_all)
    np.save(workdir +'/moments_1A_deterministic_all.npy', moments_1A_all)
    np.save(workdir +'/moments_2a_deterministic_all.npy', moments_2a_all)
    np.save(workdir +'/moments_2A_deterministic_all.npy', moments_2A_all)
    #plot_p1_p2(p1*(1-p1), p2*(1-p2), workdir, 'variance_maj_locus', Nsim, xlabel ='Time in generations', ylabel = 'Variance at the major locus', logscale = False)

def run_maj_locus_inf_bg_per_replicate(i, parameters, list_files, subpath_slim, z, dz, Nz, workdir, Nstart, Nsim):
    workdir_current = workdir + '/simulation_%i'%i
    create_directory(workdir_current, False)
    a, r, K, Kslim, theta, eta, g1, g2, m1, m2, dt, time_factor = parameters
    #n1ainitial, n2ainitial, n1Ainitial, n2Ainitial, epsilon = 0.7*Gauss(-0.5, 0.101, z), 0.3*Gauss(-0.49, 0.098, z), 0.29*Gauss(0.493, 0.099, z), 0.71*Gauss(0.501, 0.1, z), 0.1
    #n1ainitial, n2ainitial, n1Ainitial, n2Ainitial, epsilon = extract_initial_maj_locus(i, list_files, subpath_slim, z, dz, Nz, workdir_current, Nstart)
    n1ainitial, n2ainitial, n1Ainitial, n2Ainitial, epsilon = extract_initial_moments_maj_locus(i, list_files, subpath_slim, z, dz, Nz, workdir_current, Nstart, r, dt, eta, time_factor)
    parameters = epsilon, r, K, Kslim, theta, eta, g1, g2, m1, m2, dt, time_factor

    
    # Recursions
    run_model_continuous(n1ainitial, n1Ainitial, n2ainitial, n2Ainitial, parameters, Nsim, z, Nz, dz, workdir_current)
    
def plot_upon_result(parameters, Nsim, Ngen, Nstart, path, path_slim_files, list_files, Nreplicates):
    
    a, r, K, Kslim, theta, eta, g1, g2, m1, m2, dt, time_factor = parameters
    g, m = g1, m1
    subpath_slim = path_slim_files + '/' + 'm=%4.2f'%m + '_g=%4.2f'%g+'_eta=%4.2f'%eta+'_a=%4.2f'%a+'_Ngen=%i'%Ngen+'_Kslim=%i'%Kslim + '_r=%4.2f'%r + '_dt=%4.2f'%dt
    workdir = path + '/small_variance' + '_r=%4.2f'%r + '_g1=%4.2f'%g1 + '_g2=%4.2f'%g2 + '_m1=%4.2f'%m1 + '_m2=%4.2f'%m2+ '_dt=%4.3f'%dt+ '_time_factor=%i'%time_factor + '_Nstart=%i'%(Nstart/time_factor)

    
    p1_all = np.load(workdir +'/p1_deterministic_all.npy')
    p2_all = np.load(workdir +'/p2_deterministic_all.npy')
    
    plot_p1_p2(np.quantile(p1_all*(1-p1_all), 0.5, axis =1), np.quantile(p2_all*(1-p2_all), 0.5, axis = 1), workdir, 'variance_maj_locus', Nsim*time_factor, xlabel ='Time in generations', ylabel = 'Variance at the major locus', logscale = False, time_factor = time_factor)
 
def collect_p1_p2_per_simulation(i, workdir):
    workdir_current = workdir + '/simulation_%i'%i

    moments_1a, moments_1A, moments_2a, moments_2A = np.load(workdir_current +'/moments_1a.npy'), np.load(workdir_current +'/moments_1A.npy'), np.load(workdir_current +'/moments_2a.npy'), np.load(workdir_current +'/moments_2A.npy') 
    p1 = moments_1A[0, :]/(moments_1A[0, :] + moments_1a[0, :])
    p2 = moments_2A[0, :]/(moments_2A[0, :] + moments_2a[0, :])
    return(p1, p2)

def collect_moments_per_simulation(i, workdir):
    workdir_current = workdir + '/simulation_%i'%i

    moments_1a, moments_1A, moments_2a, moments_2A = np.load(workdir_current +'/moments_1a.npy'), np.load(workdir_current +'/moments_1A.npy'), np.load(workdir_current +'/moments_2a.npy'), np.load(workdir_current +'/moments_2A.npy') 

    return(moments_1a, moments_1A, moments_2a, moments_2A)

########## Auxiliary: extract densities from list of phenotypes on ssh, build n1.npy, n2.npy, and copy from ssh to local

#### Estimates the density associated to the data, on the grid z
def data_to_n_z(data, z, dz):
    fun_n = scst.gaussian_kde(data)
    if (np.sum(np.fromiter(map(fun_n, z), float)) > 0):
        out = np.fromiter(map(fun_n, z), float)/(np.sum(np.fromiter(map(fun_n, z), float))*dz)
    else: 
        out = np.zeros(np.size(z))
    return(out)

#### Extract initial stochastic distributions and sizes and turn into densities
def extract_initial_maj_locus(i, list_files, slim_storage, z, dz, Nz, workdir, Nstart):
    slim_storage_path_current = slim_storage+'_%i'%i +'/'
    file_n1a, file_n1A, file_n2a, file_n2A, file_N1a, file_N1A, file_N2a, file_N2A, file_varseg1, file_varseg2  = list_files
    n1a_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_n1a, header =None)).flatten()
    n2a_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_n2a, header =None)).flatten()
    n1A_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_n1A, header =None)).flatten()
    n2A_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_n2A, header =None)).flatten()   
    N1a_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_N1a, header =None)).flatten()
    N2a_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_N2a, header =None)).flatten()
    N1A_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_N1A, header =None)).flatten()
    N2A_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_N2A, header =None)).flatten()
    varseg1_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_varseg1, header =None)).flatten()
    varseg2_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_varseg2, header =None)).flatten()
    try:
        n1a = N1a_data[0]*data_to_n_z(n1a_data, z, dz)
    except np.linalg.LinAlgError:
        n1a = np.zeros(Nz)
    try:
        n2a = N2a_data[0]*data_to_n_z(n2a_data, z, dz)
    except np.linalg.LinAlgError:
        n2a = np.zeros(Nz) 
    try:
        n1A = N1A_data[0]*data_to_n_z(n1A_data, z, dz)
    except np.linalg.LinAlgError:
        n1A = np.zeros(Nz)
    try:
        n2A = N2A_data[0]*data_to_n_z(n2A_data, z, dz)
    except np.linalg.LinAlgError:
        n2A = np.zeros(Nz) 
        
    np.save(slim_storage_path_current + '/n1a_start.npy', n1a)
    np.save(slim_storage_path_current + '/n2a_start.npy', n2a)
    np.save(slim_storage_path_current + '/n1A_start.npy', n1A)
    np.save(slim_storage_path_current + '/n2A_start.npy', n2A)
    
    return(n1a, n2a, n1A, n2A, 1/2*np.sqrt(2)*(np.sqrt(varseg1_data[Nstart]) + np.sqrt(varseg2_data[Nstart])))

#### Extract initial IBS moments at generation 10 after burn-in and return Gaussian densities
def extract_initial_moments_maj_locus(i, list_files, slim_storage, z, dz, Nz, workdir, Nstart, r, dt, eta, time_factor):
    slim_storage_path_current = slim_storage+'/slimstorage_%i'%i +'/'
    file_N1a, file_N1A, file_N2a, file_N2A, file_z1a, file_z1A, file_z2a, file_z2A, file_v1a, file_v1A, file_v2a, file_v2A, file_varseg1, file_varseg2  = list_files
    
    varseg1_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_varseg1, header =None)).flatten()
    varseg2_data = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+ file_varseg2, header =None)).flatten()
    N1a = np.load(slim_storage +"/N1a_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    N1A = np.load(slim_storage +"/N1A_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    N2a = np.load(slim_storage +"/N2a_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    N2A = np.load(slim_storage +"/N2A_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    mean_trait_1a = np.load(slim_storage +"/mean_trait_1a_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    mean_trait_1A = np.load(slim_storage +"/mean_trait_1A_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    mean_trait_2a = np.load(slim_storage +"/mean_trait_2a_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    mean_trait_2A = np.load(slim_storage +"/mean_trait_2A_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    print(mean_trait_1A)
    print(mean_trait_1a)
    variance_trait_1a = np.load(slim_storage +"/variance_trait_1a_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    variance_trait_1A = np.load(slim_storage +"/variance_trait_1A_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    variance_trait_2a = np.load(slim_storage +"/variance_trait_2a_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    variance_trait_2A = np.load(slim_storage +"/variance_trait_2A_all.npy")[int((Nstart-time_factor+1)/time_factor), i]
    
    ### For the deterministic model, z is the quantitative bg and mean_trait is the total trait. Need to substract strong effect
    
    n1a = N1a*Gauss(mean_trait_1a - (-eta), np.sqrt(variance_trait_1a), z)
    n2a = N2a*Gauss(mean_trait_2a - (-eta), np.sqrt(variance_trait_2a), z)
    n1A = N1A*Gauss(mean_trait_1A - eta, np.sqrt(variance_trait_1A), z)
    n2A = N2A*Gauss(mean_trait_2A - eta, np.sqrt(variance_trait_2A), z)
    print(np.sqrt(variance_trait_1a))
    epsilon = 1/2*np.sqrt(2)*(np.sqrt(varseg1_data[int((Nstart-time_factor+1)/time_factor)]) + np.sqrt(varseg2_data[int((Nstart-time_factor+1)/time_factor)]))
    print(epsilon)
    time.sleep(5)
    np.save(slim_storage_path_current + '/n1a_start.npy', n1a)
    np.save(slim_storage_path_current + '/n2a_start.npy', n2a)
    np.save(slim_storage_path_current + '/n1A_start.npy', n1A)
    np.save(slim_storage_path_current + '/n2A_start.npy', n2A)
    
    return(n1a, n2a, n1A, n2A, epsilon)