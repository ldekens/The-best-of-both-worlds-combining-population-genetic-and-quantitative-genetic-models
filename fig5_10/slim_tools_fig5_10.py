#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:18:31 2020

@author: dekens
"""

import numpy as np
import pandas as pan
import shutil
import os
import scipy.stats as scist
import matplotlib.pyplot as plt
import matplotlib as ml
from matplotlib import cm
import multiprocessing
from itertools import repeat

ml.rcParams['mathtext.fontset'] = 'stix'
ml.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams.update({
    "text.usetex": True})
    
viridis = cm.get_cmap('viridis', 300)


#### First function called by main code. It runs the IBS replicates in paralle via an auxiliary function run_replicate.
#### It also calls the function change_parameters_in_slim_file that actualizes the parameters in the locally stored .txt file which is read by Slim, for each of the replicates.
def run_replicate_serie_no_plot(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, theta, dt):
        ### Creating directory ###
    if (Mut):
        subtitle = 'm=%4.2f'%m + '_g=%4.2f'%g+'_eta=%4.2f'%eta+'_a=%4.2f'%a+'_Ngen=%i'%Ngen+'_logmut =%i'%logmut+'_Kslim = %i'%Kslim + 'r=%4.2f'%r + 'dt=%4.2f'%dt
    else:
        subtitle = 'm=%4.2f'%m + '_g=%4.2f'%g+'_eta=%4.2f'%eta+'_a=%4.2f'%a+'_Ngen=%i'%Ngen+'_Kslim=%i'%Kslim + '_r=%4.2f'%r + '_dt=%4.2f'%dt
    
    ### Creating directory ###
    workdir = date+'_'+title+'_%i'%Nalleles
    subworkdir = workdir+'/'+subtitle
    slim_storage = subworkdir + '/slimstorage'
    create_directory(workdir, remove = False)
    create_directory(subworkdir, remove = True)
    
    #### Handling of the local .txt files interpreted by Slim (used by each replicates)
    shutil.copyfile(path_slim_file +'.txt', subworkdir + '/' + path_slim_file + '_local.txt') ## Copying the source slim file in local folder
    path_local_slim_file_txt = subworkdir + '/' + path_slim_file + '_local.txt'
    N01, N02 = np.floor(Kslim*4/5), np.floor(Kslim*4/5)
    ## Change the parameters in the .txt file according to the setting in the main code
    change_parameters_in_slim_file(path_local_slim_file_txt, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 26, 29, 222, 261, 373],  ['\t'+'defineConstant("dt",%4.3f'%dt+'); \n', 
                                                                                            '\t'+'defineConstant("N01",%i'%N01+'); \n', 
                                                                                            '\t'+'defineConstant("N02",%i'%N02+'); \n', 
                                                                                            '\t'+'defineConstant("K",%i'%Kslim+'); \n',
                                                                                            '\t'+'defineConstant("r",%4.1f'%r+'); \n',
                                                                                            '\t'+'defineConstant("gamma",%4.2f'%g+'); \n',
                                                                                            '\t'+'defineConstant("m",%4.2f'%m+'); \n',
                                                                                            '\t'+'defineConstant("theta",%4.1f'%theta+'); \n',
                                                                                            '\t'+'defineConstant("eta",%4.1f'%eta+'); \n',
                                                                                            '\t'+'defineConstant("Ngen",%i'%Ngen+'); \n',
                                                                                            '\t'+'defineConstant("logmut",%i'%logmut+'); \n',
                                                                                            '\t'+'defineConstant("M",%i'%Nalleles+'); \n',
                                                                                            '\t'+'defineConstant("a",%4.2f'%a+'); \n',
                                                                                            '101:%i'%Ngen+' early() { \n',
                                                                                            '101:%i'%(Ngen-1)+' late() { \n',
                                                                                            's1 %i'%Ngen+' late() \n'])

    
    
    ######## Slim ##########
    
    ###Parallel IBS ###
    
    pool = multiprocessing.Pool(processes = 10)
    inputs = [*zip(range(Nreplicate), repeat(slim_storage), repeat(path_slim_file), repeat(path_local_slim_file_txt))]
    pool.starmap(run_replicate, inputs)
    
    
    ####### Outcomes - store various variables dynamics for each replicates ########
    
    Mat_storage_replicate_p1 = np.zeros((Nsim, Nreplicate)) ####### store the dynamics of p1 and p2
    Mat_storage_replicate_p2 = np.zeros((Nsim, Nreplicate))
    
    
    Mat_storage_replicate_mean_segvar_1, Mat_storage_replicate_mean_segvar_2 = np.zeros((Nsim, Nreplicate)), np.zeros((Nsim, Nreplicate)) ####### store the dynamics of p1 and p2
    Mat_storage_replicate_variance_segvar_1, Mat_storage_replicate_variance_segvar_2 = np.zeros((Nsim, Nreplicate)), np.zeros((Nsim, Nreplicate))
    
    Mat_storage_replicate_N1A, Mat_storage_replicate_N2A = np.zeros((Nsim, Nreplicate)), np.zeros((Nsim, Nreplicate)) 
    Mat_storage_replicate_mean_trait_1A, Mat_storage_replicate_mean_trait_2A = np.zeros((Nsim, Nreplicate)), np.zeros((Nsim, Nreplicate))
    Mat_storage_replicate_variance_trait_1A, Mat_storage_replicate_variance_trait_2A = np.zeros((Nsim, Nreplicate)), np.zeros((Nsim, Nreplicate))
    
    Mat_storage_replicate_N1a, Mat_storage_replicate_N2a = np.zeros((Nsim, Nreplicate)), np.zeros((Nsim, Nreplicate)) 
    Mat_storage_replicate_mean_trait_1a, Mat_storage_replicate_mean_trait_2a = np.zeros((Nsim, Nreplicate)), np.zeros((Nsim, Nreplicate))
    Mat_storage_replicate_variance_trait_1a, Mat_storage_replicate_variance_trait_2a = np.zeros((Nsim, Nreplicate)), np.zeros((Nsim, Nreplicate))

    for i in range(Nreplicate):
        slim_storage_path_current = slim_storage+'_%i'%i
        Mat_storage_replicate_p1[:,i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/p1.txt',header =None)).flatten()
        Mat_storage_replicate_p2[:,i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/p2.txt',header =None)).flatten()
        Mat_storage_replicate_N1A[:,i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/N1A.txt',header =None)).flatten()[1:]
        Mat_storage_replicate_N1a[:,i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/N1a.txt',header =None)).flatten()[1:]
        Mat_storage_replicate_N2A[:,i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/N2A.txt',header =None)).flatten()[1:]
        Mat_storage_replicate_N2a[:,i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/N2a.txt',header =None)).flatten()[1:]
        
        Mat_storage_replicate_mean_trait_1A[:, i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/mean_1A.txt', header =None)).flatten()
        Mat_storage_replicate_variance_trait_1A[:, i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/variance_1A.txt', header =None)).flatten()
        
        Mat_storage_replicate_mean_trait_2A[:, i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/mean_2A.txt', header =None)).flatten()
        Mat_storage_replicate_variance_trait_2A[:, i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/variance_2A.txt', header =None)).flatten()
        
        Mat_storage_replicate_mean_trait_1a[:, i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/mean_1a.txt', header =None)).flatten()
        Mat_storage_replicate_variance_trait_1a[:, i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/variance_1a.txt', header =None)).flatten()
        
        Mat_storage_replicate_mean_trait_2a[:, i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/mean_2a.txt', header =None)).flatten()
        Mat_storage_replicate_variance_trait_2a[:, i] = pan.DataFrame.to_numpy(pan.read_table(slim_storage_path_current+'/variance_2a.txt', header =None)).flatten()
        print(i)
        
    np.save(subworkdir +"/mean_segvar_1_all", Mat_storage_replicate_mean_segvar_1)
    np.save(subworkdir +"/mean_segvar_2_all", Mat_storage_replicate_mean_segvar_2)
    np.save(subworkdir +"/variance_segvar_1_all", Mat_storage_replicate_variance_segvar_1)
    np.save(subworkdir +"/variance_segvar_2_all", Mat_storage_replicate_variance_segvar_2)

    np.save(subworkdir +"/N1A_all", Mat_storage_replicate_N1A)
    np.save(subworkdir +"/N2A_all", Mat_storage_replicate_N2A)
    np.save(subworkdir +"/mean_trait_1A_all", Mat_storage_replicate_mean_trait_1A)
    np.save(subworkdir +"/mean_trait_2A_all", Mat_storage_replicate_mean_trait_2A)
    np.save(subworkdir +"/variance_trait_1A_all", Mat_storage_replicate_variance_trait_1A)
    np.save(subworkdir +"/variance_trait_2A_all", Mat_storage_replicate_variance_trait_2A)
    np.save(subworkdir +"/N1a_all", Mat_storage_replicate_N1a)
    np.save(subworkdir +"/N2a_all", Mat_storage_replicate_N2a)
    np.save(subworkdir +"/mean_trait_1a_all", Mat_storage_replicate_mean_trait_1a)
    np.save(subworkdir +"/mean_trait_2a_all", Mat_storage_replicate_mean_trait_2a)
    np.save(subworkdir +"/variance_trait_1a_all", Mat_storage_replicate_variance_trait_1a)
    np.save(subworkdir +"/variance_trait_2a_all", Mat_storage_replicate_variance_trait_2a)
    np.save(subworkdir +"/p1_all", Mat_storage_replicate_p1)
    np.save(subworkdir +"/p2_all", Mat_storage_replicate_p2)


### Auxiliary function called by run_replicate_serie_no_plot, to actualize the .txt located at file.
### The lines corresponding to the line_numbers are changed thanks to stringlist.
def change_parameters_in_slim_file(file, line_numbers, stringlist):
    auxfile=open(file,'r')
    lines=auxfile.readlines()
    auxfile.close()
    for i in range(len(line_numbers)):
        lines[line_numbers[i]-1] = stringlist[i]
    auxfile=open(file,'w')
    auxfile.write("".join(lines))
    auxfile.close()

### Auxiliary function called by run_replicate_serie_no_plot to run the replicates in paralell.
### Precisely, it runs the i-th replicate by actualizing path_local_slim_file_txt to a local folder.
def run_replicate(i, slim_storage, path_slim_file, path_local_slim_file_txt):
        slim_storage_path_current = slim_storage+'_%i'%i
        create_directory(slim_storage_path_current, remove = True)
        path_local_replicate_slim_file_txt = slim_storage_path_current + '/' + path_slim_file + '_local_replicate.txt'
        shutil.copyfile(path_local_slim_file_txt, path_local_replicate_slim_file_txt)
        change_parameters_in_slim_file(path_local_replicate_slim_file_txt, line_numbers = [18], stringlist = ['\t'+'defineConstant("path", "'+ slim_storage_path_current +'"); \n'])
        os.system("./slim "+ path_local_replicate_slim_file_txt)
      
def plot_median_8_2_replicate(Mat_replicate1, Mat_replicate2, path_directory, name_variable, Ngen):
    median1 = np.quantile(Mat_replicate1, 0.5, axis = 1)
    quant_8_1 = np.quantile(Mat_replicate1, 0.8, axis = 1)
    quant_2_1 = np.quantile(Mat_replicate1, 0.2, axis = 1)
    median2 = np.quantile(Mat_replicate2, 0.5, axis = 1)
    quant_8_2 = np.quantile(Mat_replicate2, 0.8, axis = 1)
    quant_2_2 = np.quantile(Mat_replicate2, 0.2, axis = 1)
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 1/4)
    ax.set_ylabel('$p(1-p)$', fontsize = 40)
    ax.set_xlabel('Time', fontsize = 40)

    ax.plot(range(Ngen),median1, color='goldenrod', linewidth = 4, label = 'Deme 1')
    ax.fill_between(range(Ngen), median1, quant_8_1, alpha = 0.2, color='goldenrod', linewidth = 0.4, linestyle = 'dashed')
    ax.fill_between(range(Ngen), quant_2_1, median1, alpha = 0.2, color='goldenrod', linewidth = 0.4, linestyle = 'dashed')
    ax.plot(range(Ngen), median2, color='darkblue', linewidth = 4,  label = 'Deme 2')
    ax.fill_between(range(Ngen), median2, quant_8_2, color='darkblue', alpha = 0.2, linewidth = 0.4, linestyle = 'dashed')
    ax.fill_between(range(Ngen), quant_2_2, median2, alpha = 0.2, color='darkblue', linewidth = 0.4, linestyle = 'dashed')
    
    plt.legend(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.savefig(path_directory+'/'+name_variable+'.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    

def create_directory(workdir, remove):
    if (os.path.exists(workdir))&remove:
        shutil.rmtree(workdir)
    try:
    # Create target Directory
        os.mkdir(workdir)
        print("Directory " , workdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , workdir ,  " already exists")
    return

    
def slim_treatment_after_burn_in(slimstordir,Nalleles,Ngen_post_burn_in,T0):
    try:
        n1A0_slim=pan.read_table(slimstordir+'/n1A0_%i'%Nalleles+'_0.5.txt',header =None)[0]
    except:
        n1A0_slim=list()
    try:
        n2a0_slim=pan.read_table(slimstordir+'/n2a0_%i'%Nalleles+'_0.5.txt',header =None)[0]
    except:
        n2a0_slim=list()
    n1a0_slim=pan.read_table(slimstordir+'/n1a0_%i'%Nalleles+'_0.5.txt',header =None)[0]
    n2A0_slim=pan.read_table(slimstordir+'/n2A0_%i'%Nalleles+'_0.5.txt',header =None)[0]
    
    p1_slim = pan.read_table(slimstordir+'/p1_%i'%Nalleles+'_0.5.txt',header =None)[0]
    p2_slim = pan.read_table(slimstordir+'/p2_%i'%Nalleles+'_0.5.txt',header =None)[0]

    N1_slim=pan.read_table(slimstordir+'/N1_%i'%Nalleles+'_0.5.txt',header =None)[0]
    N2_slim=pan.read_table(slimstordir+'/N2_%i'%Nalleles+'_0.5.txt',header =None)[0]
    
    varseg1slim=pan.read_table(slimstordir+'/varseg%i'%Nalleles+'_0.5generation%i'%T0+'pop1.txt',header =None)[0]
    varseg2slim=pan.read_table(slimstordir+'/varseg%i'%Nalleles+'_0.5generation%i'%T0+'pop2.txt',header =None)[0]
    exitvarseg1=np.mean(varseg1slim)
    exitvarseg2=np.mean(varseg2slim)
    
    return(list([n1A0_slim,n1a0_slim,n2A0_slim,n2a0_slim,p1_slim,p2_slim,N1_slim,N2_slim,exitvarseg1,exitvarseg2]))

def initial_state_grid_est(n0est,z):
    try:
        kden=scist.gaussian_kde(n0est)
        n0 = np.zeros(np.size(z))
        for k in range(np.size(z)):
            n0[k] = kden(z[k])
    except:
        n0 = np.zeros(np.size(z))
    return(n0)
    
def graphic_comp(T,X1,X2,X1_slim,X2_slim,title,file_location):
    fig = plt.figure()
    fig.suptitle('Evolution of the '+title+' of the subpopulations in both models', fontsize=10, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_xlabel('Time $t$ (log scale)')
    ax.set_ylabel(title)
    ax.set_ylim(0,2)
    ax.plot(T,X1,color="darkblue",label = 'N_1')
    ax.plot(T,X2,color="navy",label = 'N_2')
    ax.plot(T,X1_slim,color="goldenrod",label = 'N_1 SLiM')
    ax.plot(T,X2_slim,color="darkgoldenrod",label = 'N_2 SLiM')
    plt.xscale('log')
    plt.legend(loc=0)
    plt.savefig(file_location+title+".png")
    


    
def run_plot_upon_result(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, dt, time_factor, workdir_deterministic, Nstart_deterministic):
    if (Mut):
        subtitle = 'm=%4.2f'%m + '_g=%4.2f'%g+'_eta=%4.2f'%eta+'_a=%4.2f'%a+'_Ngen=%i'%Ngen+'_logmut =%i'%logmut+'_Kslim = %i'%Kslim + 'r=%4.2f'%r + 'dt=%4.2f'%dt
    else:
        subtitle = 'm=%4.2f'%m + '_g=%4.2f'%g+'_eta=%4.2f'%eta+'_a=%4.2f'%a+'_Ngen=%i'%Ngen+'_Kslim=%i'%Kslim+ '_r=%4.2f'%r + '_dt=%4.2f'%dt
    
    ### Creating directory ###
    workdir = date+'_'+title+'_%i'%Nalleles
    subworkdir = workdir+'/'+subtitle
    Mat_storage_replicate_p1 = np.load(subworkdir +"/p1_all.npy")
    Mat_storage_replicate_p2 = np.load(subworkdir +"/p2_all.npy")
    #p1_deterministic_all = np.load(workdir_deterministic +"/p1_deterministic_all.npy")
    #p2_deterministic_all = np.load(workdir_deterministic +"/p2_deterministic_all.npy")
    gen_time = int(np.floor(1/dt))
    plot_median_8_2_replicate(Mat_storage_replicate_p1[::gen_time, :]*(1-Mat_storage_replicate_p1[::gen_time, :]), Mat_storage_replicate_p2[::gen_time, :]*(1-Mat_storage_replicate_p2[::gen_time, :]), subworkdir, 'p1_p2_8_2', int(Nsim/gen_time))
    #plot_p1_p2_median_8_2_replicate_deterministic_all(Mat_storage_replicate_p1[::gen_time, :]*(1-Mat_storage_replicate_p1[::gen_time, :]), Mat_storage_replicate_p2[::gen_time, :]*(1-Mat_storage_replicate_p2[::gen_time, :]), p1_deterministic_all[::(time_factor*gen_time), :]*(1-p1_deterministic_all[::(time_factor*gen_time), :]), p2_deterministic_all[::(time_factor*gen_time), :]*(1-p2_deterministic_all[::(time_factor*gen_time), :]), subworkdir, 'p1_p2_8_2_all_m=%4.2f'%m + '_g=%4.2f'%g, int(Nsim/gen_time), int(Nstart_deterministic/gen_time))
    #plot_p1_p2_median_8_2_replicate_deterministic_all(Mat_storage_replicate_p1[::gen_time, :]*(1-Mat_storage_replicate_p1[::gen_time, :]), Mat_storage_replicate_p2[::gen_time, :]*(1-Mat_storage_replicate_p2[::gen_time, :]), p1_deterministic_all[::(time_factor*gen_time), :]*(1-p1_deterministic_all[::(time_factor*gen_time), :]), p2_deterministic_all[::(time_factor*gen_time), :]*(1-p2_deterministic_all[::(time_factor*gen_time), :]), workdir, 'p1_p2_8_2_all_m=%4.2f'%m + '_g=%4.2f'%g, int(Nsim/gen_time), int(Nstart_deterministic/gen_time))

    ### Additional plots of moments
    Mat_storage_replicate_N1a = np.load(subworkdir +"/N1a_all.npy")
    Mat_storage_replicate_N1A = np.load(subworkdir +"/N1A_all.npy")
    Mat_storage_replicate_N2a = np.load(subworkdir +"/N2a_all.npy")
    Mat_storage_replicate_N2A = np.load(subworkdir +"/N2A_all.npy")
    Mat_storage_replicate_mean_trait_1a = np.load(subworkdir +"/mean_trait_1a_all.npy")
    Mat_storage_replicate_mean_trait_1A = np.load(subworkdir +"/mean_trait_1A_all.npy")
    Mat_storage_replicate_mean_trait_2a = np.load(subworkdir +"/mean_trait_2a_all.npy")
    Mat_storage_replicate_mean_trait_2A = np.load(subworkdir +"/mean_trait_2A_all.npy")
    Mat_storage_replicate_variance_trait_1a = np.load(subworkdir +"/variance_trait_1a_all.npy")
    Mat_storage_replicate_variance_trait_1A = np.load(subworkdir +"/variance_trait_1A_all.npy")
    Mat_storage_replicate_variance_trait_2a = np.load(subworkdir +"/variance_trait_2a_all.npy")
    Mat_storage_replicate_variance_trait_2A = np.load(subworkdir +"/variance_trait_2A_all.npy")

    #moments1a_deterministic_all = np.load(workdir_deterministic +"/moments_1a_deterministic_all.npy")
    #moments1A_deterministic_all = np.load(workdir_deterministic +"/moments_1A_deterministic_all.npy")
    #moments2a_deterministic_all = np.load(workdir_deterministic +"/moments_2a_deterministic_all.npy")
    #moments2A_deterministic_all = np.load(workdir_deterministic +"/moments_2A_deterministic_all.npy")
    gen_time = int(np.floor(1/dt))
    plot_median_8_2_replicate(Mat_storage_replicate_p1[::gen_time, :]*(1-Mat_storage_replicate_p1[::gen_time, :]), Mat_storage_replicate_p2[::gen_time, :]*(1-Mat_storage_replicate_p2[::gen_time, :]), subworkdir, 'p1_p2_8_2', int(Nsim/gen_time))
    #plot_p1_p2_median_8_2_replicate_deterministic_all(Mat_storage_replicate_p1[::gen_time, :]*(1-Mat_storage_replicate_p1[::gen_time, :]), Mat_storage_replicate_p2[::gen_time, :]*(1-Mat_storage_replicate_p2[::gen_time, :]), p1_deterministic_all[::(time_factor*gen_time), :]*(1-p1_deterministic_all[::(time_factor*gen_time), :]), p2_deterministic_all[::(time_factor*gen_time), :]*(1-p2_deterministic_all[::(time_factor*gen_time), :]), workdir_deterministic, 'p1_p2_8_2_all_m=%4.2f'%m + '_g=%4.2f'%g, int(Nsim/gen_time), int(Nstart_deterministic/gen_time))
    #plot_moments_median_8_2_replicate_deterministic_all((1+r*dt)*Mat_storage_replicate_N1a[::gen_time, 9], (1+r*dt)*Mat_storage_replicate_N1A[::gen_time, 9], (1+r*dt)*Mat_storage_replicate_N2a[::gen_time, 9], (1+r*dt)*Mat_storage_replicate_N2A[::gen_time, 9], moments1a_deterministic_all[0, ::(time_factor*gen_time), 9], moments1A_deterministic_all[0, ::(time_factor*gen_time), 9], moments2a_deterministic_all[0, ::(time_factor*gen_time), 9], moments2A_deterministic_all[0, ::(time_factor*gen_time), 9],  workdir_deterministic, 'N_all_m=%4.2f'%m + '_g=%4.2f'%g, 'Local population sizes', int(Nsim/gen_time), int(Nstart_deterministic/gen_time))
    #plot_moments_median_8_2_replicate_deterministic_all(Mat_storage_replicate_mean_trait_1a[::gen_time, 9], Mat_storage_replicate_mean_trait_1A[::gen_time, 9], Mat_storage_replicate_mean_trait_2a[::gen_time, 9], Mat_storage_replicate_mean_trait_2A[::gen_time, 9], moments1a_deterministic_all[1, ::(time_factor*gen_time), 9] - eta, moments1A_deterministic_all[1, ::(time_factor*gen_time), 9]+ eta, moments2a_deterministic_all[1, ::(time_factor*gen_time), 9]- eta, moments2A_deterministic_all[1, ::(time_factor*gen_time), 9] + eta,  workdir_deterministic, 'mean_all_m=%4.2f'%m + '_g=%4.2f'%g, 'Local mean trait',  int(Nsim/gen_time), int(Nstart_deterministic/gen_time))
    #plot_moments_median_8_2_replicate_deterministic_all(Mat_storage_replicate_variance_trait_1a[::gen_time, 9], Mat_storage_replicate_variance_trait_1A[::gen_time, 9], Mat_storage_replicate_variance_trait_2a[::gen_time, 9], Mat_storage_replicate_variance_trait_2A[::gen_time, 9], moments1a_deterministic_all[2, ::(time_factor*gen_time), 9], moments1A_deterministic_all[2, ::(time_factor*gen_time), 9], moments2a_deterministic_all[2, ::(time_factor*gen_time), 9], moments2A_deterministic_all[2, ::(time_factor*gen_time), 9],  workdir_deterministic, 'variance_all_m=%4.2f'%m + '_g=%4.2f'%g, 'Local variance in trait', int(Nsim/gen_time), int(Nstart_deterministic/gen_time))

    plot_moments_median_8_2_replicate_only_IBS(Mat_storage_replicate_N1a[::gen_time, :], Mat_storage_replicate_N1A[::gen_time, :], Mat_storage_replicate_N2a[::gen_time, :], Mat_storage_replicate_N2A[::gen_time, :],  workdir, 'N_all_m=%4.2f'%m + '_g=%4.2f'%g, 'Local population sizes', int(Nsim/gen_time))
    plot_moments_median_8_2_replicate_only_IBS(Mat_storage_replicate_mean_trait_1a[::gen_time, :], Mat_storage_replicate_mean_trait_1A[::gen_time, :], Mat_storage_replicate_mean_trait_2a[::gen_time, :], Mat_storage_replicate_mean_trait_2A[::gen_time, :], workdir, 'mean_all_m=%4.2f'%m + '_g=%4.2f'%g, 'Local mean trait',  int(Nsim/gen_time))
    plot_moments_median_8_2_replicate_only_IBS(Mat_storage_replicate_variance_trait_1a[::gen_time, :], Mat_storage_replicate_variance_trait_1A[::gen_time, :], Mat_storage_replicate_variance_trait_2a[::gen_time, :], Mat_storage_replicate_variance_trait_2A[::gen_time, :],  workdir, 'variance_all_m=%4.2f'%m + '_g=%4.2f'%g, 'Local variance in trait', int(Nsim/gen_time))
    if (g==1):
        plot_moments_median_8_2_replicate_only_IBS_tweak(Mat_storage_replicate_N1a[::gen_time, :] + Mat_storage_replicate_N1A[::gen_time, :], Mat_storage_replicate_N2a[::gen_time, :] + Mat_storage_replicate_N2A[::gen_time, :], 0*Mat_storage_replicate_p2[-1, :], workdir, 'N_partbis_m=%4.2f'%m + '_g=%4.2f'%g, 'Pop. sizes', int(Nsim/gen_time))
        plot_moments_median_8_2_replicate_only_IBS_tweak(Mat_storage_replicate_N1a[::gen_time, :] + Mat_storage_replicate_N1A[::gen_time, :], Mat_storage_replicate_N2a[::gen_time, :] + Mat_storage_replicate_N2A[::gen_time, :], Mat_storage_replicate_p2[-1, :], workdir, 'N_part_m=%4.2f'%m + '_g=%4.2f'%g, 'Pop. sizes', int(Nsim/gen_time))
    else:
        plot_moments_median_8_2_replicate_only_IBS_tweak(Mat_storage_replicate_N1a[::gen_time, :] + Mat_storage_replicate_N1A[::gen_time, :], Mat_storage_replicate_N2a[::gen_time, :] + Mat_storage_replicate_N2A[::gen_time, :], 0*Mat_storage_replicate_p2[-1, :], workdir, 'N_allbis_m=%4.2f'%m + '_g=%4.2f'%g, 'Pop. sizes', int(Nsim/gen_time))

def plot_moments_median_8_2_replicate_only_IBS(Mat_replicate1a, Mat_replicate1A, Mat_replicate2a, Mat_replicate2A, path_directory, name_variable, ylabel, Ngen):
    median1a = np.quantile(Mat_replicate1a, 0.5, axis = 1)
    quant_8_1a = np.quantile(Mat_replicate1a, 0.8, axis = 1)
    quant_2_1a = np.quantile(Mat_replicate1a, 0.2, axis = 1)
    median2a = np.quantile(Mat_replicate2a, 0.5, axis = 1)
    quant_8_2a = np.quantile(Mat_replicate2a, 0.8, axis = 1)
    quant_2_2a = np.quantile(Mat_replicate2a, 0.2, axis = 1)
    median1A = np.quantile(Mat_replicate1A, 0.5, axis = 1)
    quant_8_1A = np.quantile(Mat_replicate1A, 0.8, axis = 1)
    quant_2_1A = np.quantile(Mat_replicate1A, 0.2, axis = 1)
    median2A = np.quantile(Mat_replicate2A, 0.5, axis = 1)
    quant_8_2A = np.quantile(Mat_replicate2A, 0.8, axis = 1)
    quant_2_2A = np.quantile(Mat_replicate2A, 0.2, axis = 1)
    
    
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    #plt.xlim((5, 130))
    ax.set_ylabel(ylabel, fontsize = 40)
    ax.set_xlabel('Time', fontsize = 40)
    ax.plot(range(Ngen), median1a, color='goldenrod', linewidth = 4,  label = 'Deme 1 - a')
    ax.fill_between(range(Ngen), median1a, quant_8_1a, color='goldenrod', alpha = 0.2, linewidth = 0.4, linestyle = 'dashed')
    ax.fill_between(range(Ngen), quant_2_1a, median1a, alpha = 0.2, color='goldenrod', linewidth = 0.4, linestyle = 'dashed')
    
    ax.plot(range(Ngen), median2a, color='darkblue', linewidth = 4, label = 'Deme 2 - a')
    ax.fill_between(range(Ngen), median2a, quant_8_2a, alpha = 0.2, color='darkblue', linewidth = 0.4, linestyle = 'dashed')
    ax.fill_between(range(Ngen), quant_2_2a, median2a, alpha = 0.2, color='darkblue', linewidth = 0.4, linestyle = 'dashed')
    
    ax.plot(range(Ngen), median1A, color='peru', linewidth = 4,  label = 'Deme 1 - A')
    ax.fill_between(range(Ngen), median1A, quant_8_1A, color='peru', alpha = 0.2, linewidth = 0.4, linestyle = 'dashed')
    ax.fill_between(range(Ngen), quant_2_1A, median1A, alpha = 0.2, color='peru', linewidth = 0.4, linestyle = 'dashed')
    
    ax.plot(range(Ngen), median2A, color='midnightblue', linewidth = 4, label = 'Deme 2 - A')
    ax.fill_between(range(Ngen), median2A, quant_8_2A, alpha = 0.2, color='midnightblue', linewidth = 0.4, linestyle = 'dashed')
    ax.fill_between(range(Ngen), quant_2_2A, median2A, alpha = 0.2, color='midnightblue', linewidth = 0.4, linestyle = 'dashed')
    #if (ylabel =='Local population sizes'):
    #    ax.plot(range(Ngen), median2A + median2a, color='midnightblue', linewidth = 2, linestyle = 'dashed')
    #    ax.plot(range(Ngen), median1A + median1a, color='peru', linewidth = 2,  linestyle = 'dashed')


    plt.legend(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.savefig(path_directory+'/'+name_variable+'.png', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_moments_median_8_2_replicate_only_IBS_tweak(Mat_replicate1, Mat_replicate2, p2final, path_directory, name_variable, ylabel, Ngen):
    idx = (p2final<0.5)
    print(idx)
    median1 = np.quantile(Mat_replicate1[:, idx], 0.5, axis = 1)
    quant_8_1 = np.quantile(Mat_replicate1[:, idx], 0.8, axis = 1)
    quant_2_1 = np.quantile(Mat_replicate1[:, idx], 0.2, axis = 1)
    
    median2 = np.quantile(Mat_replicate2[:, idx], 0.5, axis = 1)
    quant_8_2 = np.quantile(Mat_replicate2[:, idx], 0.8, axis = 1)
    quant_2_2 = np.quantile(Mat_replicate2[:, idx], 0.2, axis = 1)
    
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    #plt.xlim((5, 130))
    ax.set_ylabel(ylabel, fontsize = 80)
    ax.set_xlabel('Time', fontsize = 80)
    ax.plot(range(Ngen), median1, color='goldenrod', linewidth = 13,  label = 'Deme 1')
    ax.fill_between(range(Ngen), median1, quant_8_1, color='goldenrod', alpha = 0.4, linewidth = 0.4, linestyle = 'dashed')
    ax.fill_between(range(Ngen), quant_2_1, median1, alpha = 0.4, color='goldenrod', linewidth = 0.4, linestyle = 'dashed')
    
    ax.plot(range(Ngen), median2, color='darkblue', linewidth = 10, label = 'Deme 2')
    ax.fill_between(range(Ngen), median2, quant_8_2, alpha = 0.4, color='darkblue', linewidth = 0.4, linestyle = 'dashed')
    ax.fill_between(range(Ngen), quant_2_2, median2, alpha = 0.4, color='darkblue', linewidth = 0.4, linestyle = 'dashed')
    
    
    

    plt.ylim((0, 1))
    #plt.legend(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.savefig(path_directory+'/'+name_variable+'.png', bbox_inches='tight')
    plt.show()
    plt.close()
def plot_p1_p2_median_8_2_replicate_deterministic_all(Mat_replicate1, Mat_replicate2, M1, M2, path_directory, name_variable, Ngen, Nstart_deterministic):
    median1 = np.quantile(Mat_replicate1, 0.5, axis = 1)
    quant_8_1 = np.quantile(Mat_replicate1, 0.8, axis = 1)
    quant_2_1 = np.quantile(Mat_replicate1, 0.2, axis = 1)
    median2 = np.quantile(Mat_replicate2, 0.5, axis = 1)
    quant_8_2 = np.quantile(Mat_replicate2, 0.8, axis = 1)
    quant_2_2 = np.quantile(Mat_replicate2, 0.2, axis = 1)
    median1_b = np.quantile(M1, 0.5, axis = 1)
    median2_b = np.quantile(M2, 0.5, axis = 1)

    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 1/4)
    ax.set_ylabel('$p(1-p)$', fontsize = 40)
    ax.set_xlabel('Time', fontsize = 40)
    
    ax.plot(range(Ngen), median1, color='goldenrod', linewidth = 4,  label = 'Deme 1')
    ax.fill_between(range(Ngen), median1, quant_8_1, color='goldenrod', alpha = 0.2, linewidth = 0.4, linestyle = 'dashed')
    ax.fill_between(range(Ngen), quant_2_1, median1, alpha = 0.2, color='goldenrod', linewidth = 0.4, linestyle = 'dashed')
    ax.plot(range(Ngen), median2, color='darkblue', linewidth = 4, label = 'Deme 2')
    ax.fill_between(range(Ngen), median2, quant_8_2, alpha = 0.2, color='darkblue', linewidth = 0.4, linestyle = 'dashed')
    ax.fill_between(range(Ngen), quant_2_2, median2, alpha = 0.2, color='darkblue', linewidth = 0.4, linestyle = 'dashed')
    ax.plot(range(Ngen)[Nstart_deterministic:], median2_b, color='darkblue', linewidth = 4, linestyle = 'dashed')
    ax.plot(range(Ngen)[Nstart_deterministic:], median1_b, color='goldenrod', linewidth = 4, linestyle = 'dashed')
    print(median1_b)
    print(median2_b)    
    plt.legend(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.savefig(path_directory+'/'+name_variable+'.png', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_moments_median_8_2_replicate_deterministic_all(Mat_replicate1a, Mat_replicate1A, Mat_replicate2a, Mat_replicate2A, M1a, M1A, M2a, M2A, path_directory, name_variable, ylabel, Ngen, Nstart_deterministic):
    #median1a = np.quantile(Mat_replicate1a, 0.5, axis = 1)
    median1a = Mat_replicate1a
    #quant_8_1a = np.quantile(Mat_replicate1a, 0.8, axis = 1)
    #quant_2_1a = np.quantile(Mat_replicate1a, 0.2, axis = 1)
    #median2a = np.quantile(Mat_replicate2a, 0.5, axis = 1)
    median2a = Mat_replicate2a
    #quant_8_2a = np.quantile(Mat_replicate2a, 0.8, axis = 1)
    #quant_2_2a = np.quantile(Mat_replicate2a, 0.2, axis = 1)
    #median1a_b = np.quantile(M1a, 0.5, axis = 1)
    #median2a_b = np.quantile(M2a, 0.5, axis = 1)
    median1a_b = M1a
    median2a_b = M2a
    #median1A = np.quantile(Mat_replicate1A, 0.5, axis = 1)
    median1A = Mat_replicate1A
    #quant_8_1A = np.quantile(Mat_replicate1A, 0.8, axis = 1)
    #quant_2_1A = np.quantile(Mat_replicate1A, 0.2, axis = 1)
    #median2A = np.quantile(Mat_replicate2A, 0.5, axis = 1)
    median2A = Mat_replicate2A
    #quant_8_2A = np.quantile(Mat_replicate2A, 0.8, axis = 1)
    #quant_2_2A = np.quantile(Mat_replicate2A, 0.2, axis = 1)
    #median1A_b = np.quantile(M1A, 0.5, axis = 1)
    #median2A_b = np.quantile(M2A, 0.5, axis = 1)
    median1A_b = M1A
    median2A_b = M2A
    
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    #plt.xlim((5, 130))
    ax.set_ylabel(ylabel, fontsize = 40)
    ax.set_xlabel('Time', fontsize = 40)
    ax.plot(range(Ngen), median1a, color='goldenrod', linewidth = 4,  label = 'Deme 1 - a')
    #ax.fill_between(range(Ngen), median1a, quant_8_1a, color='goldenrod', alpha = 0.2, linewidth = 0.4, linestyle = 'dashed')
    #ax.fill_between(range(Ngen), quant_2_1a, median1a, alpha = 0.2, color='goldenrod', linewidth = 0.4, linestyle = 'dashed')
    ax.plot(range(Ngen)[Nstart_deterministic:], median1a_b, color='goldenrod', linewidth = 4, linestyle = 'dashed')

    ax.plot(range(Ngen), median2a, color='darkblue', linewidth = 4, label = 'Deme 2 - a')
    ax.plot(range(Ngen)[Nstart_deterministic:], median2a_b, color='darkblue', linewidth = 4, linestyle = 'dashed')
    #ax.fill_between(range(Ngen), median2a, quant_8_2a, alpha = 0.2, color='darkblue', linewidth = 0.4, linestyle = 'dashed')
    #ax.fill_between(range(Ngen), quant_2_2a, median2a, alpha = 0.2, color='darkblue', linewidth = 0.4, linestyle = 'dashed')
    
    ax.plot(range(Ngen), median1A, color='peru', linewidth = 4,  label = 'Deme 1 - A')
    #ax.fill_between(range(Ngen), median1A, quant_8_1A, color='peru', alpha = 0.2, linewidth = 0.4, linestyle = 'dashed')
    #ax.fill_between(range(Ngen), quant_2_1A, median1A, alpha = 0.2, color='peru', linewidth = 0.4, linestyle = 'dashed')
    ax.plot(range(Ngen)[Nstart_deterministic:], median1A_b, color='peru', linewidth = 4, linestyle = 'dashed')

    ax.plot(range(Ngen), median2A, color='midnightblue', linewidth = 4, label = 'Deme 2 - A')
    ax.plot(range(Ngen)[Nstart_deterministic:], median2A_b, color='midnightblue', linewidth = 4, linestyle = 'dashed')
    #ax.fill_between(range(Ngen), median2A, quant_8_2A, alpha = 0.2, color='midnightblue', linewidth = 0.4, linestyle = 'dashed')
    #ax.fill_between(range(Ngen), quant_2_2A, median2A, alpha = 0.2, color='midnightblue', linewidth = 0.4, linestyle = 'dashed')
    
    plt.legend(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.savefig(path_directory+'/'+name_variable+'.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
def run_plot_upon_result_control(a, Nalleles, Nreplicate, date, title, Ngen, Nsim, K, Kslim, g, m, path_slim_file, eta, Mut, logmut, r, dt, time_factor):
    if (Mut):
        subtitle = 'm=%4.2f'%m + '_g=%4.2f'%g+'_eta=%4.2f'%eta+'_a=%4.2f'%a+'_Ngen=%i'%Ngen+'_logmut =%i'%logmut+'_Kslim = %i'%Kslim + 'r=%4.2f'%r + 'dt=%4.2f'%dt
    else:
        subtitle = 'm=%4.2f'%m + '_g=%4.2f'%g+'_eta=%4.2f'%eta+'_a=%4.2f'%a+'_Ngen=%i'%Ngen+'_Kslim=%i'%Kslim+ '_r=%4.2f'%r + '_dt=%4.2f'%dt
    
    ### Creating directory ###
    workdir = date+'_'+title+'_%i'%Nalleles
    subworkdir = workdir+'/'+subtitle
    Mat_storage_replicate_p1 = np.load(subworkdir +"/p1_all.npy")
    Mat_storage_replicate_p2 = np.load(subworkdir +"/p2_all.npy")

    gen_time = int(np.floor(1/dt))
    plot_median_8_2_replicate(Mat_storage_replicate_p1[::gen_time, :]*(1-Mat_storage_replicate_p1[::gen_time, :]), Mat_storage_replicate_p2[::gen_time, :]*(1-Mat_storage_replicate_p2[::gen_time, :]), subworkdir, 'p1_p2_8_2_control_m=%4.2f'%m + '_g=%4.2f'%g, int(Nsim/gen_time))
    plot_median_8_2_replicate(Mat_storage_replicate_p1[::gen_time, :]*(1-Mat_storage_replicate_p1[::gen_time, :]), Mat_storage_replicate_p2[::gen_time, :]*(1-Mat_storage_replicate_p2[::gen_time, :]), workdir, 'p1_p2_8_2_control_m=%4.2f'%m + '_g=%4.2f'%g, int(Nsim/gen_time))

    