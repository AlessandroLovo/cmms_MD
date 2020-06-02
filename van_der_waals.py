#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:04:44 2020

@author: alessandro
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import time


def visualize(folder):
    file = open(folder+'/vmd_in.txt','w')
    file.write('topo readlammpsdata %s atomic\n' %(folder+'/lattice.data'))
    file.write('animate write psf %s\n' %(folder+'/lattice.psf'))
    file.write('mol delete 0\n')
    file.write('mol load psf %s dcd %s\n' %(folder+'/lattice.psf', folder+'/trajectory.dcd'))
    file.write('mol rep vdw\n')
    file.write('mol addrep 1\n')
    file.close()
    print('vmd -e %s ' %(folder+'/vmd_in.txt'))
    return 

def make_input_data(folder,Nx,Ny=0,Nz=0,L=1.):
    folder = folder.rstrip('/')
    if not os.path.exists(folder):
        os.mkdir(folder)
    if Ny == 0:
        Ny = Nx
    if Nz == 0:
        Nz = Nx
    file = open(folder+'/lattice_parameters.txt','w')
    file.write('%d %d %d\n' %(Nx,Ny,Nz))
    file.write('%f %f %f\n' %(L,L,L))
    file.write('lattice.data')
    file.close()
    current_dir = os.path.abspath(os.path.curdir)
    os.chdir(folder)
    os.system('../../../../SourceF/lattice_atoms < lattice_parameters.txt')
    os.chdir(current_dir)
    return


def correlation(data,start,end,key):
    mx = np.mean(data['Step'][start:end])
    my = np.mean(data[key][start:end])
    Exy = np.mean(data['Step'][start:end]*data[key][start:end]) - mx*my
    return Exy/(np.std(data['Step'][start:end])*np.std(data[key][start:end]))

def block_average(data,n,coarse_sampling=1,save_stds=False):
    coarse_data = 0
    if coarse_sampling == 1:
        coarse_data = data
    else:
        coarse_data = pd.DataFrame(data=[],columns=data.keys())
        for i in np.arange(0,len(data))[::coarse_sampling]:
            coarse_data.loc[len(coarse_data)] = data.values[i]
    if n == 1:
        return coarse_data
    
    new_keys = []
    if save_stds:
        for k in coarse_data.keys():
            new_keys.append(k)
            new_keys.append(k+'_std')
    else:
        new_keys = coarse_data.keys()
    new_data = pd.DataFrame(data=[],columns=new_keys)
    
    for i in range(int(len(coarse_data)/n)):
        v = []
        for k in coarse_data.keys():
            v.append(np.mean(coarse_data[k][n*i:n*(i + 1)]))
            if save_stds:
                v.append(np.std(coarse_data[k][n*i:n*(i + 1)],ddof=1))
        new_data.loc[len(new_data)] = v
    
    return new_data
        


def simulate(folder,input_data_file,mode='nvt',mass=1.0,r_c=3.0,shift=True,T0=1.0,seed=0,mom=True,rot=True,dist='',
                       skin_depth=0.3,neigh_update_rate=20,delay=0,check=False,timestep=0.004,
                       T=1.,delta_T=0.,tau_T=0.8,P=1.,delta_P=0.,tau_P=5.,dump_rate=1000,dump_file='trajectory',thermo_rate=100,run=200000,
                       thermo_keys=['step','temp','press','etotal','pe'],analyze=True,block_average_size=100,
                       use_correlation = False, minstep = 20000,
                       correlation_window=15,correlation_keys=['Temp','Press'],average_keys=['Temp','Press'],verbose=False):
    
    start_time = time.time()
    # write input file for LAMMPS
    folder = folder.rstrip('/')
    if not os.path.exists(folder):
        os.mkdir(folder)
    ofilename = folder+'/in.inp'
    file = open(ofilename, 'w')
    file.write('log %s/settings.log\n' %folder)
    file.write('boundary p p p\n')
    file.write('units lj\n')
    file.write('atom_style atomic\n')
    file.write('read_data '+input_data_file+'\n\n')
    
    file.write('mass 1 %f\n' %mass)
    file.write('pair_style lj/cut %f \n' %r_c)
    if shift:
        file.write('pair_modify shift yes \n')
    file.write('pair_coeff 1 1 1.0 1.0\n\n')
    
    if seed == 0:
        seed = np.random.randint(10**5)
    file.write('velocity all create %f %d ' %(T0,seed))
    if mom:
        file.write('mom yes ')
    if rot:
        file.write('rot yes ')
    if len(dist) > 0:
        file.write('dist '+dist)
    file.write('\n\n')
    
    file.write('neighbor %f bin\n' %skin_depth)
    file.write('neigh_modify every %d delay %d check ' %(neigh_update_rate,delay))
    if check:
        file.write('yes\n\n')
    else:
        file.write('no\n\n')
        
    file.write('timestep %f\n\n' %timestep)
    
    if mode == 'nve':
        file.write('fix 1 all nve\n\n')
    elif mode == 'nvt':
        file.write('fix 1 all nvt temp %f %f %f\n\n' %(T,T+delta_T,tau_T))
    elif mode == 'npt':
        file.write('fix 1 all npt temp %f %f %f iso %f %f %f\n\n' %(T,T+delta_T,tau_T,P,P+delta_P,tau_P))
    else:
        raise ValueError('Unknown mode %s' %mode)
    
    file.write('dump dumpeq all dcd %d %s/%s.dcd\n\n' %(dump_rate, folder, dump_file))
    
    file.write('thermo %d \n' %thermo_rate)
    s = ''
    for k in thermo_keys:
        s += k + ' '
    file.write('thermo_style custom %s\n\n' %s)
    
    file.write('log %s/results.log\n' %folder)
    file.write('run %d\n\n' %run)
    
    file.write('write_restart %s/end1.restart' %folder)
    
    file.close()
    
    if verbose:
        print('written file '+ofilename)
    
    
    # run LAMMPS
    os.system('lmp -in %s/in.inp' %folder)

    
    # collect LAMMPS results
    lmp_data = pd.read_csv(folder+'/results.log',sep=' ',header = 14,usecols=np.arange(len(thermo_keys)),
                           nrows=(run/thermo_rate + 1),skipinitialspace=True)
    if not analyze:
        return lmp_data
    
    # do block averages:
    if block_average_size > 1:
        lmp_data = block_average(lmp_data,block_average_size)
    
    # discard equilibration of the system
    min_index = 0
    if use_correlation:
        for key in correlation_keys:
            c0 = correlation(lmp_data,0,correlation_window,key)
            for i in range(1,len(lmp_data['Step']) - correlation_window):
                c = correlation(lmp_data,i,i+correlation_window,key)
                if np.sign(c*c0) < 0:
                    if i > min_index:
                        min_index = i
                    break
    else:
        for i,s in enumerate(lmp_data['Step']):
            if s > minstep:
                min_index = i
                break
    
    if verbose:
        print('data production starts from step %d' %lmp_data['Step'][min_index])
        
    # collect averages
    averages = []
    n_good_data = len(lmp_data['Step']) - min_index
    for key in average_keys:
        averages.append(np.mean(lmp_data[key][min_index:]))
        averages.append(np.std(lmp_data[key][min_index:],ddof=1)/np.sqrt(n_good_data))
    
    end_time = time.time()
    if verbose:
        print(end_time - start_time)
    
    return np.array(averages), n_good_data, end_time - start_time


