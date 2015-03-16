#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pylab as plt
from random import sample, seed, gauss
from os.path import getsize as getFileSize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import Circle, PathPatch
import cPickle
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
from matplotlib.colors import colorConverter
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


# ================================================================================
# Basic variables
# ================================================================================

# Set up some basic attributes of the run

SIM = 'MM'
space = 'physical'
shuffle = 'None'


if ((SIM == 'SML') | (SIM == 'RSM') | (SIM == 'SSM')) :
	box_side_h1 = 20.
	pcl_mass_h1 = 0.001616 * 1.0e10
	
if ((SIM == 'G3') | (SIM == 'AG3') | (SIM == 'SG3')) :
	box_side_h1 = 50.
	pcl_mass_h1 = 0.010266 * 1.0e10
	npcls = 464.**3 
	 
if ((SIM == 'RG4') | (SIM == 'AG4') | (SIM == 'SG4')) :
	box_side_h1 = 100.
	pcl_mass_h1 = 0.1093 * 1.0e10
	 
if (SIM == 'LRG') :
	box_side_h1 = 248.
	pcl_mass_h1 = 1.0 * 1.0e10

if (SIM == 'MM') :
	box_side_h1 = 62.5
	pcl_mass_h1 = 8.6 * 1.0e8
	
Hubble_h = 1. #0.727
f_b = 0.17

box_side = box_side_h1/Hubble_h
pcl_mass = pcl_mass_h1/Hubble_h


font = {'family':'serif',
	'size':20, 
	'serif':'Times New Roman'}


matplotlib.rcdefaults()
plt.rc('axes', color_cycle=[
	'm',
	'c',
	'k',
	'b',
	'g',
	'0.5',
	], labelsize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')
plt.rc('lines', linewidth='2.0')
plt.rc('font', **font)
plt.rc('legend', numpoints=1, fontsize='x-large')
#plt.rc('text', usetex=True)

halo_dir = './rvir_files/'
pcl_dir = './rvir_files/'
env_dir = './rvir_files/'
win_dir = './rvir_files/'
plot_dir = './plots/' 
#filename = 'grid_all'
OutputFormat = '.png'
TRANSPARENT = False


z = [127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 
	10.944, 10.073, 9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 
		3.308, 3.060, 2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 
			0.989, 0.905, 0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 
				0.208, 0.175, 0.144, 0.116, 0.089, 0.064, 0.041, 0.020, 0.000]

z_fits = []
z_fits_fixed = []
z_fits_err = []
z_fits_fixed_err = []
z_colors = []
n_env_bins = 4
z_in_bin = []
pvalues = np.zeros((10, len(z)))
pvalues_bins = np.zeros((4, 10, len(z)))


mv = r'M$_{\mathrm{vir}}$'


def line_fixed_slope(x, b) :
	return 1. * x + b 

def line(x, m, b) :
	return m * x + b 


class ReadData :

	def read_gals(self, snapnum) :
		snap = "%03d" % snapnum
		galfile = halo_dir + 'halos_109_' + 'comoving' + '_' + snap
		type = []
		Mvir = []
		Rvir = []
		Mstars = []
		ColdGas = []
		IDs = []
		for item in file(galfile) :
			item = item.split()
			type.append(int(item[0]))
			Mvir.append(float(item[3])*1.0e10)
			ColdGas.append(float(item[5])*1.0e10)
			Mstars.append(float(item[6])*1.0e10)
			IDs.append(np.int64(item[1]))
#			Rvir.append(float(item[4]))
	
		type = np.array(type)
		Mvir = np.array(Mvir)
#		Rvir = np.array(Rvir)
		Mstars = np.array(Mstars)
		ColdGas = np.array(ColdGas)
		IDs = np.array(IDs)
		
#		print np.sum(ColdGas), 'Msun Cold gas at z=', z[snapnum]

#		print 'min, max Mvir:', np.log10(min(Mvir)), np.log10(max(Mvir))


		return type, Mvir, Mstars, ColdGas, IDs

	def read_radial_distributions(self, snapnum, ngals) :
		snap = "%03d" % snapnum
		pclfile = pcl_dir + 'unbound_particles_around_halos_Rvir_' + snap
		annuli = np.zeros((ngals, 10))
		cumulative = np.zeros((ngals, 10))
		gg = 0
		for item in file(pclfile) :
			item = item.split()
			for ii in range(len(item)) : 
				annuli[gg][ii] = int(item[ii])
			gg = gg + 1
		
		for gg in range(ngals) :
			for ii in range(10) :
				cumulative[gg][ii] = np.sum(annuli[gg][0:ii + 1])
				
		Mdiffuse = cumulative*pcl_mass
		return Mdiffuse
		
	def read_aperture_distributions(self, snapnum, ngals) :
		snap = "%03d" % snapnum
		pclfile = pcl_dir + 'unbound_particles_around_halos_FA_' + snap
		annuli = np.zeros((ngals, 10))
		cumulative = np.zeros((ngals, 10))
		gg = 0
		for item in file(pclfile) :
			item = item.split()
			for ii in range(len(item)) : 
				annuli[gg][ii] = int(item[ii])
			gg = gg + 1
		
		for gg in range(ngals) :
			for ii in range(10) :
				cumulative[gg][ii] = np.sum(annuli[gg][0:ii + 1])
				
		Mdiffuse = cumulative*pcl_mass
		return Mdiffuse
		
	def read_bound_distributions(self, snapnum, ngals) :
		snap = "%03d" % snapnum
		pclfile = pcl_dir + 'diffuse_halos_rvir_' + snap
		annuli = np.zeros((ngals, 10))
		cumulative = np.zeros((ngals, 10))
		gg = 0
		for item in file(pclfile) :
			item = item.split()
			for ii in range(len(item)) : 
				annuli[gg][ii] = int(item[ii])
			gg = gg + 1
		
		for gg in range(ngals) :
			for ii in range(10) :
				cumulative[gg][ii] = np.sum(annuli[gg][0:ii + 1])
				
		Mbound = cumulative*pcl_mass
		return Mbound
		
	def read_fixed_aperture(self, snapnum, ngals) :
		snap = "%03d" % snapnum
		envfile = env_dir + 'FA_N_' + space + '_' + snap
		rhorhobar_FA = np.zeros((ngals, 10))
		gg = 0
		annuli = np.zeros((ngals, 10))
		cumulative = np.zeros((ngals, 10))

		for item in file(envfile) :
			item = item.split()
			for rr in range(10) : annuli[gg][rr] = int(item[rr])
			gg = gg + 1
		
		for gg in range(ngals) :
			for ii in range(10) :
				cumulative[gg][ii] = np.sum(annuli[gg][0:ii + 1])
		
		index = np.linspace(1,10,10)
#		print index
		radii = index/2.
		volume = 4./3.*np.pi*radii**3
#		print radii, volume
		
		
		rhobar = ngals/((box_side/(1. + z[snapnum]))**3)
		rhorhobar_FA = cumulative/volume/rhobar
		
		print rhorhobar_FA[0]

#		print 'min, max log(rho/rhobar) for snap', snapnum, ',', np.log10(min(rhorhobar_FA)), np.log10(max(rhorhobar_FA))
		return rhorhobar_FA

	def read_unbound(self, z) :
		
		N_unbound = []
		for item in file('rvir_files/N_unbound.txt') :
			item = item.split()
			if len(item) > 0 : N_unbound.append(int(item[0]))
			
		N_unbound = np.array(N_unbound)
		
		return N_unbound

	def read_window(self, galnum, mass_range) :
		snap = "%03d" % snapnum
		gal = "%03d" % galnum
		winfile = win_dir + 'halo_window_' + str(mass_range) + '_' + gal + '_' + snap
		print 'reading window file', winfile
		pp = 0
		xpos = []
		ypos = []
		zpos = []
		bound = []
		for item in file(winfile) :
			item = item.split()
			if len(item) == 3 : 
				Mvir = float(item[1])*1.0e10
				Rvir = float(item[2])
			else :
				xpos.append(float(item[0]))
				ypos.append(float(item[1]))
				zpos.append(float(item[2]))
				bound.append(float(item[3]))
		
		xpos = np.array(xpos)
		ypos = np.array(ypos)
		zpos = np.array(zpos)
		bound = np.array(bound)


		return Rvir, xpos, ypos, zpos, bound

	def read_fixed_window(self, galnum, mass_range, snapnum) :
		snap = "%03d" % snapnum
		gal = "%03d" % galnum
		winfile = win_dir + 'halo_fixed_window_' + str(mass_range) + '_' + gal + '_' + snap
		print 'reading window file', winfile
		pp = 0
		xpos = []
		ypos = []
		zpos = []
		bound = []
		for item in file(winfile) :
			item = item.split()
			if len(item) == 3 : 
				Mvir = float(item[1])*1.0e10
				Rvir = float(item[2])
			else :
				xpos.append(float(item[0]))
				ypos.append(float(item[1]))
				zpos.append(float(item[2]))
				bound.append(float(item[3]))
		
		xpos = np.array(xpos)
		ypos = np.array(ypos)
		zpos = np.array(zpos)
		bound = np.array(bound)


		return Rvir, xpos, ypos, zpos, bound

	def read_pvalues(self, N1, N2) :
		
		pfile = halo_dir + N1 + '_vs_'  + N2 + '_' + space + '_pvalues.dat'

		rr = 0
		for item in file(pfile) :
			item = item.split()
			for zz in range(len(item)) :
				pvalues[rr][zz] = item[zz]
			rr = rr + 1


class Results : 




	def Mvir_vs_Mdiffuse_radial(self, snapnum, nRvir, type, Mvir, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([9.5,14.1,9.5,14.1])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)/10000*(15-8.5) + 8.5
		Hz = np.sqrt(0.7 + 0.3*(1 + z[snapnum])**3)
		print Hz
		Rpoints = (10**points / (2.32443891e14 * Hz**2))**(1./3.)

		mean_density_diff = N_unbound[snapnum]/(box_side**3)
#		print mean_density_diff
		Mdiff_expected = 4./3.*np.pi*Rpoints**3*(nRvir**3)*mean_density_diff
#		print Mdiff_expected

		Mdiff = Mdiffuse[: , nRvir - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'M$_{\mathrm{unbound}}$(<' + str(nRvir) + 'R$_{\mathrm{vir}}$)'
		col = cm.spectral((nRvir-1)/7.)

		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		ax.scatter(logMvir, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times$ ' + mv)
		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times$ ' + mv)
		ax.plot(points, points + np.log10(10.), c = 'k', lw = 3, ls = '--', label = r'$10.\times$ ' + mv)
		medMvir = np.average(logMvir)
		medMdiff = np.average(logMdiff)
		cx = 'k'
		if nRvir == 1 : cx = 'white'
		if nRvir == 9 : cx = 'white'
		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, mec = 'white', lw = 4, label = 'Median Value', ls = 'None')
#		ax.plot(points, np.log10(Mdiff_expected), c = 'magenta', lw = 4, label = 'expected', ls = 'None')
		

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'log$_{10}($M$_{\mathrm{unbound}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=2)
		plt.text(14, 9.6, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='right')

		outputFile = plot_dir + 'Mvir_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mvir_vs_Mdiffuse_aperture(self, snapnum, r_FA, type, Mvir, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([9.5,14.1,11.,15.6])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-8.5) + 8.5

		rindex = int(r_FA*2) - 1
		Mdiff = Mdiffuse[: , rindex]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse within', r_FA, 'Mpc/h for', len(w), 'galaxies'
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'M$_{\mathrm{unbound}}$(<' + str(r_FA) + '$h^{-1}$Mpc)'
		col = cm.gnuplot((r_FA+2)/8.)


		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		ax.scatter(logMvir, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times$ ' + mv)
		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times$ ' + mv)
		ax.plot(points, points + np.log10(10.), c = 'k', lw = 3, ls = '--', label = r'$10.\times$ ' + mv)
		medMvir = np.average(logMvir)
		medMdiff = np.average(logMdiff)
		cx = 'k'
		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, mec = 'white', lw = 4, label = 'Median Value', ls = 'None')
		
		if (r_FA == 1.) : loc = 1
		if ((r_FA == 2.) & (snapnum == 63)) : loc = 1
		if ((r_FA == 2.) & (snapnum == 32)) : loc = 4
		if (r_FA == 5.) : loc = 4
		ax.set_xlabel(r'log$_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'log$_{10}($M$_{\mathrm{unbound}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		plt.text(9.7, 15.5, '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='left')

		outputFile = plot_dir + 'Mvir_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_radial(self, snapnum, nRvir, type, Mstars, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([8.5,13.1,8.5,13.1])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-8.5) + 8.5

		Mdiff = Mdiffuse[: , nRvir - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mstars vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'M$_{\mathrm{unbound}}$(<' + str(nRvir) + 'R$_{\mathrm{vir}}$)'
		col = cm.spectral((nRvir-1)/7.)

		logMstars = np.log10(Mstars[w])
		logMdiff = np.log10(Mdiff[w]*f_b)
		ax.scatter(logMstars, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times$ ' + mv)
		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times$ ' + mv)
		ax.plot(points, points + np.log10(10.), c = 'k', lw = 3, ls = '--', label = r'$10.\times$ ' + mv)
		medMvir = np.average(logMstars)
		medMdiff = np.average(logMdiff)
		cx = 'k'
		if nRvir == 1 : cx = 'white'
		if nRvir == 9 : cx = 'white'
		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, mec = 'white', lw = 4, label = 'Median Value', ls = 'None')
		

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\star}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'log$_{10}(f_{b}\times$M$_{\mathrm{unbound}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=4)
		plt.text(8.7, 13, '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='left')

		outputFile = plot_dir + 'Mstars_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_aperture(self, snapnum, r_FA, type, Mstars, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([8.5,13.1,10.0,14.6])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-8.5) + 8.5

		rindex = int(r_FA*2) - 1
		Mdiff = Mdiffuse[: , rindex]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mstars vs Mdiffuse within', r_FA, 'Mpc/h for', len(w), 'galaxies'
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'M$_{\mathrm{unbound}}$(<' + str(r_FA) + '$h^{-1}$Mpc)'
		col = cm.gnuplot((r_FA+2)/8.)

		logMstars = np.log10(Mstars[w])
		logMdiff = np.log10(Mdiff[w]*f_b)
		ax.scatter(logMstars, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times$ ' + mv)
		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times$ ' + mv)
		ax.plot(points, points + np.log10(10.), c = 'k', lw = 3, ls = '--', label = r'$10.\times$ ' + mv)
		medMvir = np.average(logMstars)
		medMdiff = np.average(logMdiff)
		cx = 'k'
		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, mec = 'white', lw = 4, label = 'Median Value', ls = 'None')
		

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\star}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'log$_{10}(f_{b}\times$M$_{\mathrm{unbound}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=4)
		plt.text(8.7, 14.5, '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='left')

		outputFile = plot_dir + 'Mstars_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def rho_vs_Mdiffuse_radial(self, snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([-2.1,2.1,9.5,14.1])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-8.5) + 8.5

		Mdiff = Mdiffuse[: , nRvir - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse within', nRvir, 'Rvir and with a', r_FA, 'Mp/h aperture'
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'M$_{\mathrm{unbound}}$(<' + str(nRvir) + 'R$_{\mathrm{vir}}$)'
		col = cm.spectral((nRvir-1)/7.)

		logrho = np.log10(rho[w])
		logMdiff = np.log10(Mdiff[w])
		ax.scatter(logrho, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
# 		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times$ ' + mv)
# 		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times$ ' + mv)
# 		ax.plot(points, points + np.log10(10.), c = 'k', lw = 3, ls = '--', label = r'$10.\times$ ' + mv)
		medrho = np.average(logrho)
		medMdiff = np.average(logMdiff)
		cx = 'k'
		ax.plot(medrho, medMdiff, marker = '*', ms = 20, c = cx, mec = 'white', lw = 4, label = 'Median Value', ls = 'None')
		

		ax.set_xlabel(r'log$_{10}(\rho/\bar{\rho})$ [' + str(r_FA) + '$h^{-1}$Mpc]', fontsize=34) #54 for square
		ax.set_ylabel(r'log$_{10}($M$_{\mathrm{unbound}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=2)
		plt.text(2., 14., '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='right')

		outputFile = plot_dir + 'rho_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + str(r_FA) + 'Mpc_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def rho_vs_Mdiffuse_aperture(self, snapnum, r_FA, type, rhorhobar_FA, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([-1.6,2.6,10.5,15.1])

		points = np.linspace(0, 10000)*(15-8.5) + 8.5

		Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse within', r_FA, 'Mpc/h for', len(w), 'galaxies'
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'M$_{\mathrm{unbound}}$(<' + str(r_FA) + '$h^{-1}$Mpc)'
		col = cm.gnuplot((r_FA+2)/8.)

		logrho = np.log10(rho[w])
		logMdiff = np.log10(Mdiff[w])
		ax.scatter(logrho, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
# 		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times$ ' + mv)
# 		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times$ ' + mv)
# 		ax.plot(points, points + np.log10(10.), c = 'k', lw = 3, ls = '--', label = r'$10.\times$ ' + mv)
		medrho = np.average(logrho)
		medMdiff = np.average(logMdiff)
		cx = 'k'
		ax.plot(medrho, medMdiff, marker = '*', ms = 20, c = cx, mec = 'white', lw = 4, label = 'Median Value', ls = 'None')
		

		ax.set_xlabel(r'log$_{10}(\rho/\bar{\rho})$ [' + str(r_FA) + '$h^{-1}$Mpc]', fontsize=34) #54 for square
		ax.set_ylabel(r'log$_{10}($M$_{\mathrm{unbound}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=3)
		plt.text(2.5, 15., '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='right')

		outputFile = plot_dir + 'rho_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def rho_vs_Mdiffuse_radial_Mstar_bins(self, snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse, Mstars) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = 0.15, hspace = 0.04, top = .96, bottom = .12, left = .12, right = .95)

		

		Mdiff = Mdiffuse[: , nRvir - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse within', nRvir, 'Rvir for a', r_FA, 'Mpc aperture'
	
		label = r'M$_{\mathrm{unbound}}$(<' + str(nRvir) + 'R$_{\mathrm{vir}}$)'
		label1 = r'log$_{10}(\rho/\bar{\rho})$ [' + str(r_FA) + '$h^{-1}$Mpc]'
		label2 = r'log$_{10}($M$_{\mathrm{unbound}}/h^{-1}\mathrm{M}_{\odot})$'
		label3 = r'log$_{10}($M$_{\star}/h^{-1}\mathrm{M}_{\odot})$'
		col = cm.spectral((nRvir-1)/7.)

		logrho = np.log10(rho[w])
		logMdiff = np.log10(Mdiff[w])
		logMstars = np.log10(Mstars[w])	
		print logMstars
	
		Mmin = np.min(logMstars)
		bin = (np.max(logMstars) - np.min(logMstars))/4.
		for mm in range(4) : 
			mbin_min = (Mmin + mm*bin)
			mbin_max = (Mmin + (mm + 1)*bin)
			Mbin_min = "%2.1f" % mbin_min
			Mbin_max = "%2.1f" % mbin_max
			w = np.where((mbin_min <= logMstars) & (logMstars < mbin_max))[0]
			print len(w), 'galaxies in mass bin', Mbin_min, '-', Mbin_max
			
			logrho_w = logrho[w]
			logMdiff_w = logMdiff[w]

			ax = fig.add_subplot(2,2,mm + 1)

			plt.axis([-1.6,2.6,9.5,14.1])

			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(18)

			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(18)
				
			if mm < 2 : ax.set_xticklabels([])
#			if mm == 1 : ax.set_yticklabels([])
#			if mm == 3 : ax.set_yticklabels([])

			ax.scatter(logrho_w, logMdiff_w, c = col, edgecolor = col, alpha = 0.3, label = label)
		

			if mm == 2 : 
				ax.legend(prop = matplotlib.font_manager.FontProperties(size=18),
					fancybox=True,loc=4, ncol = 3, borderpad = 0.1, columnspacing = 0.1)
			log = r'$\mathrm{log}_{10}$'
			Msun = r'$\mathrm{M}_{\odot}$'

			text2 = Mbin_min + r'$< (\mathrm{M}_{\star}/ h^{-1} \mathrm{M}_{\odot}) <$' + Mbin_max
			plt.text(0.5, 14, text2, size = 20, color = 'k', 
				verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.9))
			if mm == 3 : plt.text(2.5, 9.6, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='right')


		fig.text(0.5, 0.04, label1, ha='center', size = 30)
		fig.text(0.04, 0.5, label2, va='center', rotation='vertical', size = 30)

		outputFile = plot_dir + 'Mstars_bins_rho_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + str(r_FA) + 'Mpc_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def rho_vs_Mdiffuse_aperture_Mstar_bins(self, snapnum, r_FA, type, rhorhobar_FA, Mdiffuse, Mstars) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = 0.15, hspace = 0.04, top = .96, bottom = .12, left = .12, right = .95)

		

		Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse within', r_FA, 'Mpc/h for', len(w), 'galaxies'
	
		label = r'M$_{\mathrm{unbound}}$(<' + str(r_FA) + '$h^{-1}$Mpc)'
		label1 = r'log$_{10}(\rho/\bar{\rho})$ [' + str(r_FA) + '$h^{-1}$Mpc]'
		label2 = r'log$_{10}($M$_{\mathrm{unbound}}/h^{-1}\mathrm{M}_{\odot})$'
		label3 = r'log$_{10}($M$_{\star}/h^{-1}\mathrm{M}_{\odot})$'

		col = cm.gnuplot((r_FA+2)/8.)


		logrho = np.log10(rho[w])
		logMdiff = np.log10(Mdiff[w])
		logMstars = np.log10(Mstars[w])	
		print logMstars
	
		Mmin = np.min(logMstars)
		bin = (np.max(logMstars) - np.min(logMstars))/4.
		for mm in range(4) : 
			mbin_min = (Mmin + mm*bin)
			mbin_max = (Mmin + (mm + 1)*bin)
			Mbin_min = "%2.1f" % mbin_min
			Mbin_max = "%2.1f" % mbin_max
			w = np.where((mbin_min <= logMstars) & (logMstars < mbin_max))[0]
			print len(w), 'galaxies in mass bin', Mbin_min, '-', Mbin_max
			
			logrho_w = logrho[w]
			logMdiff_w = logMdiff[w]

			fit = np.polyfit(logrho_w, logMdiff_w, 1)
			print 'log(Mdiff) =', fit[0], '* log(rho/rhobar) +', fit[1]


			ax = fig.add_subplot(2,2,mm + 1)

			plt.axis([-1.6,2.6,9.5,14.1])

			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(18)

			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(18)
				
			if mm < 2 : ax.set_xticklabels([])
#			if mm == 1 : ax.set_yticklabels([])
#			if mm == 3 : ax.set_yticklabels([])

			ax.scatter(logrho_w, logMdiff_w, c = col, edgecolor = col, alpha = 0.3, label = label)
		

			if mm == 2 : 
				ax.legend(prop = matplotlib.font_manager.FontProperties(size=18),
					fancybox=True,loc=0, ncol = 3, borderpad = 0.1, columnspacing = 0.1)
			log = r'$\mathrm{log}_{10}$'
			Msun = r'$\mathrm{M}_{\odot}$'

			text2 = Mbin_min + r'$< (\mathrm{M}_{\star}/ h^{-1} \mathrm{M}_{\odot}) <$' + Mbin_max
			plt.text(0.5, 9.6, text2, size = 20, color = 'k', 
				verticalalignment='bottom', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.9))
			if mm == 1 : plt.text(2.5, 14, '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='right')


		fig.text(0.5, 0.04, label1, ha='center', size = 30)
		fig.text(0.04, 0.5, label2, va='center', rotation='vertical', size = 30)

		outputFile = plot_dir + 'Mstars_bins_rho_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def Mstar_vs_Mdiffuse_radial_rho_bins(self, snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse, Mstars) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = 0.15, hspace = 0.04, top = .96, bottom = .12, left = .12, right = .95)

		

		Mdiff = Mdiffuse[: , nRvir - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse within', nRvir, 'Rvir for a', r_FA, 'Mpc aperture'
	
		label = r'M$_{\mathrm{unbound}}$(<' + str(nRvir) + 'R$_{\mathrm{vir}}$)'
		label1 = r'log$_{10}(\rho/\bar{\rho})$ [' + str(r_FA) + '$h^{-1}$Mpc]'
		label2 = r'log$_{10}($M$_{\mathrm{unbound}}/h^{-1}\mathrm{M}_{\odot})$'
		label3 = r'log$_{10}($M$_{\star}/h^{-1}\mathrm{M}_{\odot})$'
		col = cm.spectral((nRvir-1)/7.)

		logrho = np.log10(rho[w])
		logMdiff = np.log10(Mdiff[w])
		logMstars = np.log10(Mstars[w])	
		print logMstars
	
		Mmin = np.min(logrho)
		bin = (np.max(logrho) - np.min(logrho))/4.
		for mm in range(4) : 
			mbin_min = (Mmin + mm*bin)
			mbin_max = (Mmin + (mm + 1)*bin)
			Mbin_min = "%2.1f" % mbin_min
			Mbin_max = "%2.1f" % mbin_max
			w = np.where((mbin_min <= logrho) & (logrho < mbin_max))[0]
			print len(w), 'galaxies in rho bin', Mbin_min, '-', Mbin_max
			
			logMstars_w = logMstars[w]
			logMdiff_w = logMdiff[w]

			ax = fig.add_subplot(2,2,mm + 1)

			plt.axis([8.5,13.1,9.5,14.1])

			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(18)

			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(18)
				
			if mm < 2 : ax.set_xticklabels([])
#			if mm == 1 : ax.set_yticklabels([])
#			if mm == 3 : ax.set_yticklabels([])

			ax.scatter(logMstars_w, logMdiff_w, c = col, edgecolor = col, alpha = 0.3, label = label)
		

			if mm == 2 : 
				ax.legend(prop = matplotlib.font_manager.FontProperties(size=18),
					fancybox=True,loc=4, ncol = 3, borderpad = 0.1, columnspacing = 0.1)
			log = r'$\mathrm{log}_{10}$'
			Msun = r'$\mathrm{M}_{\odot}$'

			text2 = Mbin_min + r'$< (\rho/\bar{\rho}) <$' + Mbin_max
			plt.text(10.8, 14, text2, size = 20, color = 'k', 
				verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.9))
			if mm == 3 : plt.text(13, 9.6, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='right')


		fig.text(0.5, 0.04, label3, ha='center', size = 30)
		fig.text(0.04, 0.5, label2, va='center', rotation='vertical', size = 30)

		outputFile = plot_dir + 'rho_bins_Mstars_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + str(r_FA) + 'Mpc_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()





	def point_to_line_distance(self, xp, yp, slope, intercept) :
		num = -(slope * xp - yp + intercept)
		den_sq = slope*slope + 1.
		
		distance = num/np.sqrt(den_sq)
		
		return distance
				
	def Mvir_vs_Mdiffuse_Mbound(self, snapnum, nRvir, type, Mvir, Mdiff, Mbound) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([9.5,14.1,0.01,1000])
		ax.set_yscale('log')

		points = np.linspace(0, 10000)*(15-9.5) + 9.5

		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'$\mathrm{M}_{\mathrm{diffuse}}(<' + str(nRvir+1) + '\mathrm{R}_{\mathrm{vir}})$'
		col = cm.jet((nRvir-1)/8.)

		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		logMbound = np.log10(Mbound[w] - Mvir[w])
		ax.scatter(logMvir, Mdiff[w]/(Mbound[w] - Mvir[w]), c = col, edgecolor = col, alpha = 0.3, label = label)
#		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times \mathrm{M}_{\mathrm{vir}}$')
#		ax.plot(points, points + np.log10(0.5), c = 'k', lw = 3, ls = '--', label = r'$0.5\times \mathrm{M}_{\mathrm{vir}}$')
#		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times \mathrm{M}_{\mathrm{vir}}$')
#		medMvir = np.average(logMvir)
#		medMdiff = np.average(logMdiff)
#		cx = 'k'
#		if nRvir == 1 : cx = 'white'
#		if nRvir == 9 : cx = 'white'
#		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, lw = 4)
		

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'$\mathrm{M}_{\mathrm{diffuse}}/\mathrm{M}_{\mathrm{bound}}$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
		plt.text(12.5, 9.6, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Mvir_vs_Mdiff_Mbound_' + str(nRvir+1) + 'Rvir_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def Mvir_vs_Mdiffuse_Mvir_vs_FA(self, snapnum, nRvir, type, Mvir, Mdiff, rhorhobar_FA) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .15, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([9.5,14.1,-1.,1.])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-9.5) + 9.5

		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'$\mathrm{M}_{\mathrm{diffuse}}(<' + str(nRvir+1) + '\mathrm{R}_{\mathrm{vir}})$'
		col = cm.jet((nRvir-1)/8.)

		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		MdiffMvir = logMdiff - logMvir
		env = rhorhobar_FA[w]
		f_min = np.min(env)
		f_max = np.max(env)

		pts = ax.scatter(logMvir, MdiffMvir, cmap = cm.jet, c = env, vmin = f_min, vmax = f_max, 
			norm = colors.LogNorm(), edgecolor = 'none', alpha = 1, label = label, s = 3)
#		ax.scatter(logMvir, MdiffMvir, c = col, edgecolor = col, alpha = 0.3, label = label)
#		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times \mathrm{M}_{\mathrm{vir}}$')
#		ax.plot(points, points + np.log10(0.5), c = 'k', lw = 3, ls = '--', label = r'$0.5\times \mathrm{M}_{\mathrm{vir}}$')
		ax.plot(points, 0.*points, c = 'k', lw = 2, label = r'$1.0\times \mathrm{M}_{\mathrm{vir}}$')
		medMvir = np.average(logMvir)
		medMdiff = np.average(logMdiff)
		cx = 'k'
		if nRvir == 1 : cx = 'white'
		if nRvir == 9 : cx = 'white'
#		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, lw = 4)
		
		cbar = plt.colorbar(pts)
		cbar.set_label(r'$\mathrm{log}_{10}(\rho/\bar{\rho})$', fontsize=34) #54 for square

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{diffuse}}/\mathrm{M}_{\mathrm{vir}})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
		plt.text(12.5, -0.7, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')
		plt.text(12.5, -0.9, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Mvir_vs_MdiffMvir_vs_' + str(int(r_FA)) + 'Mpc_FA_' + str(nRvir + 1) + 'Rvir_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def Mvir_vs_FA(self, snapnum, type, Mvir, rhorhobar_FA) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]

		w = np.where((type == 0))[0]
		print 'plotting Mvir vs FA for a', r_FA, 'Mpc aperture for', len(w), 'galaxies'
		logMvir = np.log10(Mvir[w])
		logenv = np.log10(rhorhobar_FA[w])
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)


		xmin = 9.5
		xmax = 14.1
		ymin = -0.4
		ymax = 1.5
		plt.axis([xmin,xmax, ymin,ymax])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-9.5) + 9.5
		fit = np.polyfit(logMvir, logenv, 1)
		fit_points = points * fit[0] + fit[1]


	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'$2h^{-1}\mathrm{Mpc}\ \mathrm{FA}$' 
		col = cm.jet((2-1)/8.)

		nbins = 10
		H, yedges, xedges = np.histogram2d(logenv, logMvir, bins = nbins)
		print yedges
		x_off = (xedges[1] - xedges[0])/2.
		y_off = (yedges[1] - yedges[0])/2.
		X, Y = np.meshgrid(xedges[0:nbins] + x_off, yedges[0:nbins] + y_off)
		print np.shape(H), np.shape(X), np.shape(Y)

		ax.scatter(logMvir, logenv, c = col, edgecolor = col, alpha = 0.3, label = label)
		ax.plot(points, fit_points, c = 'purple', lw = 4, ls = '-', label = r'$\mathrm{Best}\ \mathrm{fit}$')
		ax.contour(X,Y, H)
#		ax.plot(points, points + np.log10(0.5), c = 'k', lw = 3, ls = '--', label = r'$0.5\times \mathrm{M}_{\mathrm{vir}}$')
#		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times \mathrm{M}_{\mathrm{vir}}$')
#		medMvir = np.average(logMvir)
#		medMdiff = np.average(logMdiff)
#		cx = 'k'
#		if nRvir == 1 : cx = 'white'
#		if nRvir == 9 : cx = 'white'
#		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, lw = 4)
		

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'$\mathrm{log}_{10}(\rho/\bar{\rho})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
		plt.text(12.5, -0.3, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Mvir_vs_' + str(r_FA) + 'FA_' + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def calculate_pvalues(self, snapnum, V1, V2, N1, N2, nRvir, type) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]

		if N1 == 'Mdiff' :
			w = np.where((type == 0) & (V1 > 0.))[0]
		elif N2 == 'Mdiff' :
			w = np.where((type == 0) & (V2 > 0.))[0]
		else:
			w = np.where((type == 0))[0]

		print 'calculating', N1, 'vs', N2, 'for z = ', snap, 'for', len(w), 'galaxies'
		logV1 = np.log10(V1[w])
		logV2 = np.log10(V2[w])
		
		pval = pearsonr(logV1, logV2)
		print 'Pearson p-value = ', pval
		pv = "%1.3f" % pval[0]
		pvalues[nRvir][snapnum] = pval[0]

	def calculate_pvalues_Mbins(self, snapnum, Mvir, V1, V2, N1, N2, nRvir, type) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]

		if N1 == 'Mdiff' :
			w = np.where((type == 0) & (V1 > 0.))[0]
		elif N2 == 'Mdiff' :
			w = np.where((type == 0) & (V2 > 0.))[0]
		else:
			w = np.where((type == 0))[0]

		print 'calculating', N1, 'vs', N2, 'for z = ', snap, 'for', len(w), 'galaxies'
		logMvir = np.log10(Mvir[w])
		logV1 = np.log10(V1[w])
		logV2 = np.log10(V2[w])
		if shuffle == 'All' :
			np.random.shuffle(logV2)
		
		Mmin = 10. #np.min(logMvir)
		bin = 1. #(np.max(logMvir) - np.min(logMvir))/4.
		for mm in range(4) :
			ww = np.where((Mmin + mm*bin <= logMvir) & (logMvir <= Mmin + (mm + 1)*bin))[0]
			print len(ww), 'galaxies in bin'
			logV2_w = logV2[ww]
			if shuffle == 'Mbins' :
				np.random.shuffle(logV2_w)
			pval = pearsonr(logV1[ww], logV2_w)
			print 'Pearson p-value = ', pval[0]
			pv = "%1.3f" % pval[0]
			pvalues_bins[mm][nRvir][snapnum] = pval[0]
			
		print pvalues_bins.shape

	def calculate_pvalues_rhobins(self, snapnum, rhorhobar_FA, V1, V2, N1, N2, nRvir, type) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]

		if N1 == 'Mdiff' :
			w = np.where((type == 0) & (V1 > 0.))[0]
		elif N2 == 'Mdiff' :
			w = np.where((type == 0) & (V2 > 0.))[0]
		else:
			w = np.where((type == 0))[0]

		print 'calculating', N1, 'vs', N2, 'for z = ', snap, 'for', len(w), 'galaxies'
		logrho = np.log10(rhorhobar_FA[w])
		logV1 = np.log10(V1[w])
		logV2 = np.log10(V2[w])
		if shuffle == 'All' :
			np.random.shuffle(logV2)
		
		rhomin = -.12 #np.min(logMvir)
		bin = .65 #(np.max(logMvir) - np.min(logMvir))/4.
		for pp in range(4) :
			ww = np.where((rhomin + pp*bin <= logrho) & (logrho <= rhomin + (pp + 1)*bin))[0]
			print len(ww), 'galaxies in bin'
			if len(ww) > 0 :
				logV2_w = logV2[ww]
				if shuffle == 'rhobins' :
					np.random.shuffle(logV2_w)
				pval = pearsonr(logV1[ww], logV2_w)
				print 'Pearson p-value = ', pval[0]
				pv = "%1.3f" % pval[0]
				pvalues_bins[pp][nRvir][snapnum] = pval[0]
			else :
				pvalues_bins[pp][nRvir][snapnum] = -10.
				

			
		print pvalues_bins.shape

	def Mvir_vs_Mdiffuse_vs_Mstars(self, snapnum, nRvir, type, Mvir, Mdiff, Mstars) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse vs Mstars within', nRvir, 'Rvir for', len(w), 'galaxies'
		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		logMstars = np.log10(Mstars[w])

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .94)
		ax = fig.add_subplot(1,1,1)

		plt.axis([9.5,14.1,9.5,14.1])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-9.5) + 9.5

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'$\mathrm{M}_{\mathrm{diffuse}}(<' + str(nRvir+1) + '\mathrm{R}_{\mathrm{vir}})$'
		f_stars = (Mstars[w]/Mvir[w])
		f_min = np.min(f_stars)
		f_max = np.max(f_stars)
		f_range = f_max - f_min
#		col = cm.jet(f_stars, vmin = min_f, vmax = max_f)

		pts = ax.scatter(logMvir, logMdiff, cmap = cm.jet, c = f_stars, vmin = f_min, vmax = f_max, 
			norm = colors.LogNorm(), edgecolor = 'none', alpha = 1, label = label, s = 3)
		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times \mathrm{M}_{\mathrm{vir}}$')
		ax.plot(points, points + np.log10(0.5), c = 'k', lw = 3, ls = '--', label = r'$0.5\times \mathrm{M}_{\mathrm{vir}}$')
		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times \mathrm{M}_{\mathrm{vir}}$')
		medMvir = np.average(logMvir)
		medMdiff = np.average(logMdiff)
		cx = 'k'
		if nRvir == 1 : cx = 'white'
		if nRvir == 9 : cx = 'white'
		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, lw = 4)
		
#		cax = divider.append_axes("right", size="5%", pad=0.1)
		cbar = plt.colorbar(pts)
		cbar.set_label(r'$\mathrm{M}_{\star}/\mathrm{M}_{\mathrm{vir}}$')


		

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{diffuse}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
		plt.text(12.5, 9.6, r'$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Mvir_vs_Mdiff_vs_Mstars_' + str(nRvir+1) + 'Rvir_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def Mvir_vs_Mdiffuse_vs_ColdGas(self, snapnum, nRvir, type, Mvir, Mdiff, ColdGas) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		w = np.where((type == 0) & (Mdiff > 0) & (ColdGas > 0.))[0]
		print 'plotting Mvir vs Mdiffuse vs Cold Gas within', nRvir, 'Rvir for', len(w), 'galaxies'
		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		logMstars = np.log10(Mstars[w])

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .94)
		ax = fig.add_subplot(1,1,1)

		plt.axis([9.5,14.1,9.5,14.1])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-9.5) + 9.5

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'$\mathrm{M}_{\mathrm{diffuse}}(<' + str(nRvir+1) + '\mathrm{R}_{\mathrm{vir}})$'
		f_gas = (ColdGas[w]/Mvir[w])
		f_min = np.min(f_gas)
		f_max = np.max(f_gas)
		f_range = f_max - f_min
#		col = cm.jet(f_stars, vmin = min_f, vmax = max_f)

		pts = ax.scatter(logMvir, logMdiff, cmap = cm.jet, c = f_gas, vmin = f_min, vmax = f_max, 
			norm = colors.LogNorm(), edgecolor = 'none', alpha = 1, label = label, s = 3)
		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times \mathrm{M}_{\mathrm{vir}}$')
		ax.plot(points, points + np.log10(0.5), c = 'k', lw = 3, ls = '--', label = r'$0.5\times \mathrm{M}_{\mathrm{vir}}$')
		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times \mathrm{M}_{\mathrm{vir}}$')
		medMvir = np.average(logMvir)
		medMdiff = np.average(logMdiff)
		cx = 'k'
		if nRvir == 1 : cx = 'white'
		if nRvir == 9 : cx = 'white'
		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, lw = 4)
		
#		cax = divider.append_axes("right", size="5%", pad=0.1)
		cbar = plt.colorbar(pts)
		cbar.set_label(r'$\mathrm{M}_{\mathrm{Cold Gas}}/\mathrm{M}_{\mathrm{vir}}$')


		

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{diffuse}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
		plt.text(12.5, 9.6, r'$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Mvir_vs_Mdiff_vs_ColdGas_' + str(nRvir+1) + 'Rvir_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def Mvir_vs_Mdiffuse_vs_FA(self, snapnum, nRvir, type, Mvir, Mdiff, rhorhobar_FA) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse vs FA within', nRvir, 'Rvir for', len(w), 'galaxies'
		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])

		fit = np.polyfit(logMvir, logMdiff, 1)
		print 'log(Mdiff) =', fit[0], '* log(Mvir) +', fit[1]


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .94)
		ax = fig.add_subplot(1,1,1)

		plt.axis([9.5,14.1,9.5,14.1])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-9.5) + 9.5

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'$\mathrm{M}_{\mathrm{diffuse}}(<' + str(nRvir+1) + '\mathrm{R}_{\mathrm{vir}})$'
		env = rhorhobar_FA[w]
		f_min = np.min(env)
		f_max = np.max(env)

		pts = ax.scatter(logMvir, logMdiff, cmap = cm.jet, c = env, vmin = f_min, vmax = f_max, 
			norm = colors.LogNorm(), edgecolor = 'none', alpha = 1, label = label, s = 3)
		ax.plot(points, points + np.log10(0.1), c = 'k', lw = 4, ls = ':', label = r'$0.1\times \mathrm{M}_{\mathrm{vir}}$')
		ax.plot(points, points + np.log10(0.5), c = 'k', lw = 3, ls = '--', label = r'$0.5\times \mathrm{M}_{\mathrm{vir}}$')
		ax.plot(points, points, c = 'k', lw = 2, label = r'$1.0\times \mathrm{M}_{\mathrm{vir}}$')
		ax.plot(points, fit[0]*points + fit[1], c = 'purple', lw = 5, ls = '-', label = r'$\mathrm{best}\ \mathrm{fit}$')
		medMvir = np.average(logMvir)
		medMdiff = np.average(logMdiff)
		cx = 'k'
		if nRvir == 1 : cx = 'white'
		if nRvir == 9 : cx = 'white'
		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, lw = 4)
		
#		cax = divider.append_axes("right", size="5%", pad=0.1)
		cbar = plt.colorbar(pts)
		cbar.set_label(r'$\rho/{\bar{\rho}}$')


		

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{diffuse}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
		plt.text(12.5, 9.6, r'$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')
		plt.text(12.5, 9.9, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Mvir_vs_Mdiff_vs_' + str(int(r_FA)) + 'Mpc_FA_' + str(nRvir + 1) + 'Rvir_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()
				
	def Mvir_vs_Mdiffuse_vs_FA_bins(self, snapnum, nRvir, type, Mvir, Mdiff, rhorhobar_FA) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse vs FA within', nRvir + 1, 'Rvir for', len(w), 'galaxies'
		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		env = rhorhobar_FA[w]
		logenv = np.log10(env)
		
		fit = np.polyfit(logMvir, logMdiff, 1)
		print 'log(Mdiff) =', fit[0], '* log(Mvir) +', fit[1]
		
		wrho = []
		min_env = np.min(logenv - 0.01)
		max_env = np.max(logenv)
		env_range = max_env - min_env
		env_bin_size = (max_env - min_env)/float(n_env_bins)
		bin_fits = []
		bin_fits_fixed = []
		bin_fits_err = []
		bin_fits_fixed_err = []
		sorted_env = np.sort(logenv)
		ngalfrac = int(float(len(logenv))/float(n_env_bins))
		Mean_color = []
		lowest_bins = 0
		in_bin = []
		for ii in range(n_env_bins) :
			bin_start = sorted_env[ii * ngalfrac + lowest_bins]
			bin_end = sorted_env[(ii + 1) * ngalfrac - 1 + lowest_bins]
			if bin_start == bin_end :
				ww = np.where(bin_start == logenv)[0]
				lowest_bins = lowest_bins + len(ww)
				ngalfrac = int(float(len(logenv) - lowest_bins)/float(n_env_bins))
			else :
				bin_start = sorted_env[ii * ngalfrac + lowest_bins]
				bin_end = sorted_env[(ii + 1) * ngalfrac - 1 + lowest_bins]
				ww = np.where((bin_start < logenv) & (logenv <= bin_end))[0]				
			in_bin.append(float(len(ww))/float(len(logenv)))
			wrho.append(ww)
			print len(ww), 'galaxies in env between', bin_start, 'and', bin_end
			popt, pcov = curve_fit(line, logMvir[ww], logMdiff[ww])
#			print popt[1], np.sqrt(pcov[1,1])
			bin_fits.append(popt)
			perr = np.sqrt(np.diag(pcov))
			bin_fits_err.append(perr)
			popt, pcov = curve_fit(line_fixed_slope, logMvir[ww], logMdiff[ww])
			bin_fits_fixed.append(popt)
			perr = np.sqrt(pcov[0])
			bin_fits_fixed_err.append(perr)
			Mean_color.append((np.average(logenv[ww])- min_env)/env_range)

		env90 = sorted_env[len(logenv)*9/10]
		ww = np.where(logenv > env90)[0]
		print len(ww), 'galaxies in top 10%'
		popt, pcov = curve_fit(line, logMvir[ww], logMdiff[ww])
		bin_fits.append(popt)
		perr = np.sqrt(np.diag(pcov))
		bin_fits_err.append(perr)
		popt, pcov = curve_fit(line_fixed_slope, logMvir[ww], logMdiff[ww])
		bin_fits_fixed.append(popt)
		perr = np.sqrt(pcov[0])
		bin_fits_fixed_err.append(perr)
		
		popt, pcov = curve_fit(line, logMvir, logMdiff)
		bin_fits.append(popt)
		perr = np.sqrt(np.diag(pcov))
		bin_fits_err.append(perr)
		popt, pcov = curve_fit(line_fixed_slope, logMvir, logMdiff)
		bin_fits_fixed.append(popt)
		perr = np.sqrt(pcov[0])
		bin_fits_fixed_err.append(perr)

		z_fits.append(np.array(bin_fits))
		z_fits_fixed.append(np.array(bin_fits_fixed))
		z_fits_err.append(np.array(bin_fits_err))
		z_fits_fixed_err.append(np.array(bin_fits_fixed_err))
		z_colors.append(np.array(Mean_color))
		z_in_bin.append(np.array(in_bin))

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .94)
		ax = fig.add_subplot(1,1,1)

		plt.axis([9.5,14.1,9.5,14.1])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-9.5) + 9.5

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label = r'$\mathrm{M}_{\mathrm{diffuse}}(<' + str(nRvir+1) + '\mathrm{R}_{\mathrm{vir}})$'
		env = rhorhobar_FA[w]
		f_min = np.min(env)
		f_max = np.max(env)

		pts = ax.scatter(logMvir, logMdiff, cmap = cm.spectral, c = env, vmin = f_min, vmax = f_max, 
			norm = colors.LogNorm(), edgecolor = 'none', alpha = 1, label = label, s = 3)
		for ii in range(n_env_bins) :
			X = points
			Y = bin_fits[ii][0] * points + bin_fits[ii][1]	
			label = str(ii*25)+ '-' + str((ii + 1)*25) + r'$\%\ \mathrm{densest}$'
			ax.plot(X, Y, c = cm.gist_ncar(Mean_color[ii]), lw = 4, ls = '-', label = label)
# 		medMvir = np.average(logMvir)
# 		medMdiff = np.average(logMdiff)
# 		cx = 'k'
# 		if nRvir == 1 : cx = 'white'
# 		if nRvir == 9 : cx = 'white'
# 		ax.plot(medMvir, medMdiff, marker = '*', ms = 20, c = cx, lw = 4)
		
#		cax = divider.append_axes("right", size="5%", pad=0.1)
		cbar = plt.colorbar(pts)
		cbar.set_label(r'$\rho/{\bar{\rho}}$')


		

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.set_ylabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{diffuse}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
		plt.text(12.5, 9.6, r'$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')
		plt.text(12.5, 9.9, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Mvir_vs_Mdiff_vs_' + str(int(r_FA)) + 'Mpc_FA_bins_' + str(nRvir + 1) + 'Rvir_' + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()
				
	def Mvir_Mdiffuse_scatter_vs_FA(self, snapnum, nRvir, type, Mvir, Mdiff, rhorhobar_FA) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse vs FA within', nRvir, 'Rvir for', len(w), 'galaxies'
		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])

		label = r'$\mathrm{M}_{\mathrm{diffuse}}(<' + str(nRvir+1) + '\mathrm{R}_{\mathrm{vir}})$'
		env = rhorhobar_FA[w]
		logenv = np.log10(env)
		
		fit = np.polyfit(logMvir, logMdiff, 1)
		print 'log(Mdiff) =', fit[0], '* log(Mvir) +', fit[1]
		scatter_from_fit = res.point_to_line_distance(logMvir, logMdiff, fit[0], fit[1])


		wrho = []
		n_env_bins = 4
		min_env = np.min(logenv - 0.01)
		max_env = np.max(logenv)
		env_bin_size = (max_env - min_env)/float(n_env_bins)
		Mean_Mvir = []
		Mean_scatter = []
		Mean_color =[]
		sorted_env = np.sort(logenv)
		ngalfrac = int(float(len(logenv))/float(n_env_bins))
		for ii in range(n_env_bins) :
			bin_start = sorted_env[ii * ngalfrac]
			bin_end = sorted_env[(ii + 1) * ngalfrac - 1]
			ww = np.where((bin_start <= logenv) & (logenv <= bin_end))[0]
			wrho.append(ww)
			print len(ww), 'galaxies in env between', bin_start, 'and', bin_end
			Mean_Mvir.append(np.average(logMvir[ww]))
			Mean_scatter.append(np.average(scatter_from_fit[ww]))
			Mean_color.append(np.average(logenv[ww]))
			

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .13, right = .94)
		ax = fig.add_subplot(1,1,1)

		plt.axis([10.1,14.1,-0.5,0.5])
		ax.set_xticks([11,12,13,14])

		points = np.linspace(0, 10000)*(15-9.5) + 9.5

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		f_min = np.min(env)
		f_max = np.max(env)

		pts = ax.scatter(logMvir, scatter_from_fit, cmap = cm.jet, c = env, vmin = f_min, vmax = f_max, 
			norm = colors.LogNorm(), edgecolor = 'none', alpha = 1, label = label, s = 3)
		ax.scatter(Mean_Mvir, Mean_scatter, c = Mean_color, cmap = cm.jet, vmin = np.min(logenv), vmax = np.max(logenv),
			marker = '*', s = 200)
		
#		cax = divider.append_axes("right", size="5%", pad=0.1)
		cbar = plt.colorbar(pts)
		cbar.set_label(r'$\mathrm{log}_{10}(\rho/\bar{\rho})$', fontsize=34) #54 for square


		

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=34) 
		ax.set_ylabel(r'$\mathrm{Scatter}\ \mathrm{from}\ \mathrm{fit}$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
		plt.text(12.5, -0.45, r'$z = $' + zed, size = 30, color = 'k', verticalalignment='bottom', horizontalalignment='left')
		plt.text(12.5, -0.4, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Mvir_Mdiff_scatter_vs_' + str(int(r_FA)) + 'Mpc_FA_' + str(nRvir + 1) + 'Rvir_' + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()


		print len(logMvir)
		xh = logMvir
		zh = logMdiff
		yh = logenv
		A = np.column_stack((np.ones(len(xh)), xh, yh))
		c, resid,rank,sigma = np.linalg.lstsq(A,zh)

		print '3D - Mvir vs. Env vs. Mdiff'
		print 'c', c
		print 'resid', resid
		print 'rank', rank
		print 'sigma', sigma

		A = np.column_stack((np.ones(len(xh)), xh, zh))
		c, resid,rank,sigma = np.linalg.lstsq(A,yh)

		print '3D - Mvir vs. Mdiff vs. Env'
		print 'c', c
		print 'resid', resid
		print 'rank', rank
		print 'sigma', sigma

	def Mvir_vs_Mdiffuse_vs_FA_3D(self, snapnum, nRvir, type, Mvir, Mdiff, rhorhobar_FA, Mstars) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse vs FA within', nRvir, 'Rvir for', len(w), 'galaxies'
		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		logMstars = np.log10(Mstars[w])
		logenv = np.log10(rhorhobar_FA[w])


		print len(logMvir)
		xh = logMvir
		zh = logMdiff
		yh = logenv
		ave_xh = np.average(xh)
		ave_yh = np.average(yh)
		ave_zh = np.average(zh)

# 		xh = xh/ave_xh
# 		yh = yh/ave_yh
# 		zh = zh/ave_zh
		
		print 'median xh, yh, zh:', ave_xh, ave_yh, ave_zh
				
		A = np.column_stack((np.ones(len(xh)), xh, yh))
		c, resid,rank,sigma = np.linalg.lstsq(A,zh)

		print '3D - Mvir vs. Env vs. Mdiff'
		print 'c', c
		print 'resid', resid
		print 'rank', rank
		print 'sigma', sigma

		A2 = np.column_stack((np.ones(len(xh)), xh))
		c, resid,rank,sigma = np.linalg.lstsq(A2,zh)
		
		print '2D - Mvir vs. Mdiff'
		print 'c', c
		print 'resid', resid
		print 'rank', rank
		print 'sigma', sigma


		A = np.column_stack((np.ones(len(xh)), xh, zh))
		c, resid,rank,sigma = np.linalg.lstsq(A,yh)

		print '3D - Mvir vs. Mdiff vs. Env'
		print 'c', c
		print 'resid', resid
		print 'rank', rank
		print 'sigma', sigma



		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .08, left = .04, right = .94)
		ax = fig.add_subplot(111, projection='3d')
#		plt.axis([9.5,14.1,9.5,14.1])
#		ax.set_xscale('log')

		points = np.linspace(0, 10000)*(15-9.5) + 9.5
		env_plane = c[0] + c[1]*logMvir + c[2]*logMdiff
		print 'predicted env', env_plane

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(20)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(20)
		for tick in ax.zaxis.get_major_ticks():
			tick.label1.set_fontsize(20)

		ax.plot_surface(logMvir, logMdiff, env_plane, color = 'k')
#		ax.scatter(logMvir, logMdiff, env_plane)

		label = r'$\mathrm{M}_{\mathrm{diffuse}}(<' + str(nRvir+1) + '\mathrm{R}_{\mathrm{vir}})$'
		env = rhorhobar_FA[w]
		logenv = np.log10(env)
		f_min = np.min(logMstars)
		f_max = np.max(logMstars)

		pts = ax.scatter(logMvir, logMdiff, logenv, cmap = cm.jet, c = logMstars, vmin = f_min, vmax = f_max, 
		edgecolor = 'none', alpha = 1, label = label, s = 3)

		cbar = plt.colorbar(pts, ticks = [9.0, 9.5, 10.0, 10.5, 11.0])
		cbar.set_label(r'$\mathrm{log}_{10}(\mathrm{M}_{\star}/h^{-1} \mathrm{M}_{\odot})$', size = 24)
		cbar.ax.tick_params(labelsize=20) 

	

		ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=24) #54 for square
		ax.set_ylabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{diffuse}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=24) #54 for square
		ax.set_zlabel(r'$\mathrm{log}_{10}(\rho/\bar{\rho})$', fontsize=24) #54 for square
		ax.text2D(0.05, 0.98, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$' + "\n" + r'$z = $' + zed, 
			transform=ax.transAxes, fontsize = 30, verticalalignment = 'top')


		outputFile = plot_dir + '3D_Mvir_vs_Mdiff_vs_' + str(int(r_FA)) + 'Mpc_FA_' + str(nRvir + 1) + 'Rvir_' + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def Mvir_vs_Mdiffuse_vs_FA_3D_movie(self, snapnum, nRvir, type, Mvir, Mdiff, rhorhobar_FA, Mstars, nframes) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		w = np.where((type == 0) & (Mdiff > 0))[0]
		print 'plotting Mvir vs Mdiffuse vs FA within', nRvir, 'Rvir for', len(w), 'galaxies'
		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		logMstars = np.log10(Mstars[w])
		env = rhorhobar_FA


		step = int(360./float(nframes))
		for ii in xrange(0,360,5):
			angle = "%03d" % ii
			print angle, 'degrees'

			fig = plt.figure(figsize=(12.,10))	
			fig.subplots_adjust(wspace = .0,top = .96, bottom = .08, left = .04, right = .94)
			ax = fig.add_subplot(111, projection='3d')
	#		plt.axis([9.5,14.1,9.5,14.1])
	#		ax.set_xscale('log')

			points = np.linspace(0, 10000)*(15-9.5) + 9.5

	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(20)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(20)
			for tick in ax.zaxis.get_major_ticks():
				tick.label1.set_fontsize(20)

			label = r'$\mathrm{M}_{\mathrm{diffuse}}(<' + str(nRvir+1) + '\mathrm{R}_{\mathrm{vir}})$'
			env = rhorhobar_FA[w]
			logenv = np.log10(env)
			f_min = np.min(env)
			f_max = np.max(env)

			ax.view_init(elev=10., azim=ii)

			pts = ax.scatter(logMvir, logMdiff, logenv, cmap = cm.jet, c = env, vmin = f_min, vmax = f_max, 
			norm = colors.LogNorm(), edgecolor = 'none', alpha = 1, label = label, s = 3)

			cbar = plt.colorbar(pts, ticks = [9.0, 9.5, 10.0, 10.5, 11.0])
			cbar.set_label(r'$\mathrm{log}_{10}(\rho/\bar{\rho})$', size = 24)
			cbar.ax.tick_params(labelsize=20) 

		

			ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=24) #54 for square
			ax.set_ylabel(r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{diffuse}}/h^{-1}\mathrm{M}_{\odot})$', fontsize=24) #54 for square
			ax.set_zlabel(r'$\mathrm{log}_{10}(\rho/\bar{\rho})$', fontsize=24) #54 for square
			ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

	#		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
	#		plt.text(12.5, 9.6, r'$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')
	#		plt.text(12.5, 9.9, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')
			ax.text2D(0.05, 0.98, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$' + "\n" + r'$z = $' + zed, 
				transform=ax.transAxes, fontsize = 30, verticalalignment = 'top')


			plt.savefig(plot_dir + 'movie' + angle + '.png')

			plt.close()

	def Mvir_Mdiff_FA_evolution(self, z_fits, z_fits_err, z_fits_fixed, z_fits_fixed_err, z_colors) :
		
		z_fits = np.array(z_fits)
		z_fits_fixed = np.array(z_fits_fixed)
		z_fits_err = np.array(z_fits_err)
		z_fits_fixed_err = np.array(z_fits_fixed_err)
		z_colors = np.array(z_colors)
		z_range = z[-len(z_fits):len(z)]
		print z_range
		
		print (z_fits[:, 0, 0])
		print len(z_range)
		print z_colors[:,0]
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .15, right = .95)
		ax = fig.add_subplot(1,1,1)


		xmin = 0.
		xmax = 2.07
		ymin = .81
		ymax = 1.05
#		plt.axis([xmin,xmax, ymin,ymax])
		ax.set_xlim(xmin, xmax)

		print np.shape(z_colors)
		print z_colors[-1]

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		for ii in range(n_env_bins) :
			label = str(ii*25)+ '-' + str((ii + 1)*25) + r'$\%\ \mathrm{densest}$'
			data = z_fits[:, ii, 0]
			colors = z_colors[-1,ii]
			errors = z_fits_err[:, ii, 0]
			ax.fill_between(z_range, data-errors, data + errors, facecolor = cm.jet((colors - 0.2)/0.5), 
				edgecolor = 'none', label = label, lw = 4, alpha = 0.5)
#			ax.errorbar(z_range, data, yerr = errors, c = cm.jet((colors - 0.2)/0.5), label = label, lw = 4)
			ax.plot(z_range, data, c = cm.jet((colors - 0.2)/0.5), label = label, lw = 4)

		ii = n_env_bins
		label = r'$10\%\ \mathrm{densest}$'
		data = z_fits[:, ii, 0]
		errors = z_fits_err[:, ii, 0]
		ax.fill_between(z_range, data-errors, data + errors, facecolor = 'purple', 
			edgecolor = 'none', label = label, lw = 4, alpha = 0.5)
#			ax.errorbar(z_range, data, yerr = errors, c = cm.jet((colors - 0.2)/0.5), label = label, lw = 4)
		ax.plot(z_range, data, c = 'purple', label = label, lw = 4)

		ii = n_env_bins + 1
		label = r'$\mathrm{All}\ \mathrm{Galaxies}$'
		data = z_fits[:, ii, 0]
		errors = z_fits_err[:, ii, 0]
		ax.fill_between(z_range, data-errors, data + errors, facecolor = 'k', 
			edgecolor = 'none', label = label, lw = 4, alpha = 0.5)
#			ax.errorbar(z_range, data, yerr = errors, c = cm.jet((colors - 0.2)/0.5), label = label, lw = 4)
		ax.plot(z_range, data, c = 'k', label = label, lw = 4)
		

		ax.set_xlabel(r'$\mathrm{z}$', fontsize=40) #54 for square
		ax.set_ylabel(r'$\mathrm{slope}$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
#		plt.text(12.5, -0.3, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'slope_Mvir_vs_Mdiff_vs_' + str(r_FA) + 'FA_' + str(nRvir + 1) + 'Rvir_' + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .15, right = .95)
		ax = fig.add_subplot(1,1,1)


		xmin = 0.
		xmax = 2.07
		ymin = -0.9
		ymax = 2.0
#		plt.axis([xmin,xmax, ymin,ymax])
		ax.set_xlim(xmin, xmax)

		print np.shape(z_fits)

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		for ii in range(n_env_bins) :
			label = str(ii*25)+ '-' + str((ii + 1)*25) + r'$\%\ \mathrm{densest}$'
			data = z_fits[:, ii, 1]
			colors = z_colors[-1,ii]
			errors = z_fits_err[:, ii, 1]
			ax.fill_between(z_range, data-errors, data + errors, facecolor = cm.jet((colors - 0.2)/0.5), 
				edgecolor = 'none', label = label, lw = 4, alpha = 0.5)
			ax.plot(z_range, data, c = cm.jet((colors - 0.2)/0.5), label = label, lw = 4)
		
		ii = n_env_bins
		label = r'$10\%\ \mathrm{densest}$'
		data = z_fits[:, ii, 1]
		errors = z_fits_err[:, ii, 1]
		ax.fill_between(z_range, data-errors, data + errors, facecolor = 'purple', 
			edgecolor = 'none', label = label, lw = 4, alpha = 0.5)
#			ax.errorbar(z_range, data, yerr = errors, c = cm.jet((colors - 0.2)/0.5), label = label, lw = 4)
		ax.plot(z_range, data, c = 'purple', label = label, lw = 4)

		ii = n_env_bins + 1
		label = r'$\mathrm{All}\ \mathrm{Galaxies}$'
		data = z_fits[:, ii, 1]
		errors = z_fits_err[:, ii, 1]
		ax.fill_between(z_range, data-errors, data + errors, facecolor = 'k', 
			edgecolor = 'none', label = label, lw = 4, alpha = 0.5)
#			ax.errorbar(z_range, data, yerr = errors, c = cm.jet((colors - 0.2)/0.5), label = label, lw = 4)
		ax.plot(z_range, data, c = 'k', label = label, lw = 4)

		ax.set_xlabel(r'$\mathrm{z}$', fontsize=40) #54 for square
		ax.set_ylabel(r'$\mathrm{amplitude}\ [\mathrm{log}(h^{-1}\mathrm{M}_{\odot})]$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
#		plt.text(12.5, -0.3, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'amplitude_Mvir_vs_Mdiff_vs_' + str(r_FA) + 'FA_' + str(nRvir + 1) + 'Rvir_' + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()
		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .15, right = .95)
		ax = fig.add_subplot(1,1,1)


		xmin = 0.
		xmax = 2.07
		ymin = 0.6
		ymax = 3.5
#		plt.axis([xmin,xmax, ymin,ymax])
		ax.set_xlim(xmin, xmax)

		print np.shape(z_fits_fixed)
		print np.shape(z_fits_fixed_err)

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		for ii in range(n_env_bins) :
			label = str(ii*25)+ '-' + str((ii + 1)*25) + r'$\%\ \mathrm{densest}$'
			data = z_fits_fixed[:, ii, 0]
			colors = z_colors[-1,ii]
			errors = z_fits_fixed_err[:, ii, 0]
#			print errors
#			print np.shape(data), np.shape(errors), np.shape(z_range)
			ax.fill_between(z_range, data-errors, data + errors, facecolor = cm.jet((colors - 0.2)/0.5), 
				edgecolor = 'none', label = label, lw = 4, alpha = 0.5)
			ax.plot(z_range, data, c = cm.jet((colors - 0.2)/0.5), label = label, lw = 4)
		
		ii = n_env_bins
		label = r'$10\%\ \mathrm{densest}$'
		data = z_fits[:, ii, 0]
		errors = z_fits_err[:, ii, 0]
		ax.fill_between(z_range, data-errors, data + errors, facecolor = 'purple', 
			edgecolor = 'none', label = label, lw = 4, alpha = 0.5)
#			ax.errorbar(z_range, data, yerr = errors, c = cm.jet((colors - 0.2)/0.5), label = label, lw = 4)
		ax.plot(z_range, data, c = 'purple', label = label, lw = 4)

		ii = n_env_bins + 1
		label = r'$\mathrm{All}\ \mathrm{Galaxies}$'
		data = z_fits[:, ii, 0]
		errors = z_fits_err[:, ii, 0]
		ax.fill_between(z_range, data-errors, data + errors, facecolor = 'k', 
			edgecolor = 'none', label = label, lw = 4, alpha = 0.5)
#			ax.errorbar(z_range, data, yerr = errors, c = cm.jet((colors - 0.2)/0.5), label = label, lw = 4)
		ax.plot(z_range, data, c = 'k', label = label, lw = 4)

		ax.set_xlabel(r'$\mathrm{z}$', fontsize=40) #54 for square
		ax.set_ylabel(r'$\mathrm{amplitude}\ [\mathrm{log}(h^{-1}\mathrm{M}_{\odot})]$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0)
#		plt.text(12.5, -0.3, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'amplitude_fixed_Mvir_vs_Mdiff_vs_' + str(r_FA) + 'FA_' + str(nRvir + 1) + 'Rvir_' + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()
		
	def Find_Galaxies_to_Evolve(self, snaps) :
		Type = []
		Mvir = []
		IDs = []
		for zz in snaps :
			type, mvir, _ , _, ids = res.read_gals(zz)
			print 'max Mvir', np.max(mvir)
			ww = np.where(type == 0)[0]
			if (zz == 63) : ww = np.where((type == 0) & (1.0e11 < mvir) & (mvir < 1.5e11))[0]
			Type.append(type[ww])
			Mvir.append(mvir[ww])
			IDs.append(ids[ww])
			print len(ww), 'galaxies in snap', zz
			
		IDs01 = set(IDs[0]).intersection(IDs[1])
		print len(IDs01), 'galaxies in snapshots 0 and 1'
		IDs23 = set(IDs[2]).intersection(IDs[3])
		print len(IDs23), 'galaxies in snapshots 2 and 3'

		IDs_all = set(IDs01).intersection(IDs23)
		
		print len(IDs_all), 'galaxies in all 4 snapshots in mass range'
#		print IDs_all
		
		f = open(halo_dir + 'IDs_evolution_11.dat','w')
		for id in IDs_all : 
			f.write(str(id) + '\n')
		f.close() 
		
	def Galaxy_Window(self, galnum, snapnum, mass_range, nRvir, Rvir, xpos, ypos, zpos, bound) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		gal = "%03d" % galnum
		
		print np.sum(bound), '= bound - unbound'
		
		ww = np.where((-1. < zpos) & (zpos < 1.))[0]
		print len(ww), 'particles shown out of', len(xpos)
		xpos = xpos[ww]
		ypos = ypos[ww]
		zpos = zpos[ww]
		bound = bound[ww]
		

# 		xpos = np.where((-box_side/2. < xpos) & (xpos < box_side/2.), xpos, xpos - box_side/2.)
# 		ypos = np.where((-box_side/2. < ypos) & (ypos < box_side/2.), ypos, ypos - box_side/2.)
# 		zpos = np.where((-box_side/2. < zpos) & (zpos < box_side/2.), zpos, zpos - box_side/2.)

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .13, right = .94)
		ax = fig.add_subplot(1,1,1, aspect = 'equal')

		plt.axis([-8.*Rvir,8.*Rvir,-8.*Rvir,8.*Rvir])
#		ax.set_xticks([11,12,13,14])

		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		ax.scatter(0., 0., c = cm.brg(2.5/2.7), 
			label = r'$\mathrm{Bound}\ \mathrm{Particles}$', edgecolor = 'none')
		ax.scatter(0., 0., c = cm.brg(0.5/2.7),
			label = r'$\mathrm{Unbound}\ \mathrm{Particles}$', edgecolor = 'none')
		pts = ax.scatter(xpos, ypos, cmap = cm.brg, c = bound, vmin = -1.5, vmax = 1.2, 
			edgecolor = 'none', alpha = 0.7, s = 10)

		Rvir1=plt.Circle((0,0),Rvir,color='k',fill=False, lw = 3)
#			, label = r'$1\ \mathrm{R}_{\mathrm{vir}}$')
		Rvir6=plt.Circle((0,0),(nRvir + 1)*Rvir,color='k',fill=False, lw = 5, ls = 'dashed')
#			, label = r'$6\ \mathrm{R}_{\mathrm{vir}}$')
		ax.add_patch(Rvir1)
		ax.add_patch(Rvir6)

#		cax = divider.append_axes("right", size="5%", pad=0.1)
#		cbar = plt.colorbar(pts)
#		cbar.set_label(r'$\mathrm{log}_{10}(\rho/\bar{\rho})$', fontsize=34) #54 for square


		
		text1 = r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}(z=0)/h^{-1} \mathrm{M}_{\odot}) = $' + str(mass_range)
		text2 = r'$z = $' + zed
		ax.set_xlabel(r'$\mathrm{X}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) 
		ax.set_ylabel(r'$\mathrm{Y}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=22),fancybox=True,loc=0)
		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$11$' + '\n' + text2, size = 34, color = 'k', 
			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
#		plt.text(12.5, -0.4, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Galaxy_Window_' + str(mass_range) + '_' + gal + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def Galaxy_Window_contours(self, galnum, snapnum, mass_range, nRvir, Rvir, xpos, ypos, zpos, bound) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		gal = "%03d" % galnum
		
		print np.sum(bound), '= bound - unbound'
		
		ww = np.where((-1. < zpos) & (zpos < 1.))[0]
		print len(ww), 'particles shown out of', len(xpos)
		xpos = xpos[ww]
		ypos = ypos[ww]
		zpos = zpos[ww]
		bound = bound[ww]
		
		wb = np.where(bound > 0.)[0]
		xpos_b = xpos[wb]
		ypos_b = ypos[wb]
		zpos_b = zpos[wb]

		wu = np.where(bound < 0.)[0]
		xpos_u = xpos[wu]
		ypos_u = ypos[wu]
		zpos_u = zpos[wu]
		
		axis_range = [-8.*Rvir,8.*Rvir,-8.*Rvir,8.*Rvir]
		
		n_hist_bins = 20
		hist_bins = np.linspace(-8.,8., n_hist_bins + 1)*Rvir
		H_a, xedges, yedges = np.histogram2d(ypos, xpos, bins = hist_bins)
		H_ab, xedges, yedges = np.histogram2d(ypos, xpos, bins = hist_bins, weights = bound)
		H_u, xedges, yedges = np.histogram2d(ypos_u, xpos_u, bins = hist_bins)
		H_b, xedges, yedges = np.histogram2d(ypos_b, xpos_b, bins = hist_bins)
		dumX = hist_bins[:n_hist_bins] + (hist_bins[1] - hist_bins[0])/2.
		X, Y = np.meshgrid(dumX, dumX)
		
		bin_side = (axis_range[1] - axis_range[0])/float(n_hist_bins)
		H_a = H_a * (1. / bin_side**3)/(270./box_side)**3
		H_ab = H_ab * (1. / bin_side**3)/(270./box_side)**3
		H_u = H_u * (1. / bin_side**3)/(270./box_side)**3
		H_b = H_b * (1. / bin_side**3)/(270./box_side)**3
		min_rho_per_bin = (1. / bin_side**3)/(270./box_side)**3

		print np.size(X), np.size(Y), np.size(H_b)
		
		logH_u = np.log10(H_u)
		logH_u = np.where(H_u > 0, np.log10(H_u), np.log10(min_rho_per_bin))
		logH_b = np.log10(H_b)
		logH_bb = np.where(H_b > 0, np.log10(H_b), np.log10(min_rho_per_bin))
		logH_a = np.log10(H_a)

# 		xpos = np.where((-box_side/2. < xpos) & (xpos < box_side/2.), xpos, xpos - box_side/2.)
# 		ypos = np.where((-box_side/2. < ypos) & (ypos < box_side/2.), ypos, ypos - box_side/2.)
# 		zpos = np.where((-box_side/2. < zpos) & (zpos < box_side/2.), zpos, zpos - box_side/2.)

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .9, bottom = .12, left = .13, right = .9)
		ax = fig.add_subplot(1,1,1, aspect = 'equal')
		ax.xaxis.set_label_position('top') 
		ax.xaxis.tick_top()

		plt.axis(axis_range)
#		ax.set_xticks([11,12,13,14])

		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		plt.axis(axis_range)


#		cmap_b = colors.ListedColormap(['Greens'])

		boundedness = (H_ab/H_a + 1.)/2.
		density_b = logH_bb
		norm = colors.Normalize(density_b.min(), density_b.max())
		bound_array = plt.get_cmap('Greens')(norm(density_b)*0.9)
		bound_array[..., 3] = boundedness*0.9  # <- some alpha values between 0.5-1
#		print red_array

		
		im2 = plt.imshow(logH_b, interpolation='sinc', cmap=cm.Greens,
                origin='lower', extent=axis_range,
                vmax=logH_b.max(), vmin=logH_u.min(), alpha = 0.7)
		im = plt.imshow(logH_u, interpolation='sinc', cmap=cm.Purples,
                origin='lower', extent=axis_range,
                vmax=logH_u.max(), vmin=logH_u.min())
		plt.imshow(bound_array, interpolation='spline36', 
                origin='lower', extent=axis_range)
#		ax.scatter(xpos_u, ypos_u, s = 1, alpha = 0.1, edgecolor = 'none')

		levels = [1., 1.5, 2.0, 2.5, 3.0, 3.5]
		if (np.max(logH_b) < 3.5) : levels = [1., 1.5, 2.0, 2.5, 3.0]
		
		cs = plt.contour(X, Y, logH_b, levels = levels, cmap = cm.YlGn)

		Rvir1=plt.Circle((0,0),Rvir,color='k',fill=False, lw = 3)
#			, label = r'$1\ \mathrm{R}_{\mathrm{vir}}$')
		Rvir6=plt.Circle((0,0),(nRvir + 1)*Rvir,color='k',fill=False, lw = 5, ls = 'dashed')
#			, label = r'$6\ \mathrm{R}_{\mathrm{vir}}$')
		ax.add_patch(Rvir1)
		ax.add_patch(Rvir6)

		text1 = r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}(z=0)/h^{-1} \mathrm{M}_{\odot}) = $' + str(mass_range)
		text2 = r'$z = $' + zed
		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$13$' + '\n' + text2, size = 34, color = 'k', 
			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))


		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.1)
		cax2 = divider.append_axes("bottom", size="5%", pad=0.1)
		cbar = plt.colorbar(im, cax = cax, ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
		cbar2 = plt.colorbar(im2, cax = cax2, orientation='horizontal', ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
		cbar.set_label(r'$\mathrm{log}_{10}(\rho_{\mathrm{Unbound}}/\bar{\rho})$', fontsize=34) #54 for square
		cbar2.set_label(r'$\mathrm{log}_{10}(\rho_{\mathrm{Bound}}/\bar{\rho})$', fontsize=34) #54 for square


		
		ax.set_xlabel(r'$\mathrm{X}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) 
		ax.set_ylabel(r'$\mathrm{Y}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) #54 for square
# 		ax.legend(prop = matplotlib.font_manager.FontProperties(size=22),fancybox=True,loc=0)
#		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$13$' + '\n' + text2, size = 34, color = 'k', 
#			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
#		plt.text(12.5, -0.4, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Galaxy_Window_contours_' + str(mass_range) + '_' + gal + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def Galaxy_Window_contours2(self, galnum, snapnum, mass_range, nRvir, Rvir, xpos, ypos, zpos, bound) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		gal = "%03d" % galnum
		
		print np.sum(bound), '= bound - unbound'
		
		ww = np.where((-1. < zpos) & (zpos < 1.))[0]
		print len(ww), 'particles shown out of', len(xpos)
		xpos = xpos[ww]
		ypos = ypos[ww]
		zpos = zpos[ww]
		bound = bound[ww]
		
		wb = np.where(bound > 0.)[0]
		xpos_b = xpos[wb]
		ypos_b = ypos[wb]
		zpos_b = zpos[wb]

		wu = np.where(bound < 0.)[0]
		xpos_u = xpos[wu]
		ypos_u = ypos[wu]
		zpos_u = zpos[wu]
		
		axis_range = [-8.*Rvir,8.*Rvir,-8.*Rvir,8.*Rvir]
		
		n_hist_bins = 30
		hist_bins = np.linspace(-8.,8., n_hist_bins + 1)*Rvir
		H_a, xedges, yedges = np.histogram2d(ypos, xpos, bins = hist_bins)
		H_ab, xedges, yedges = np.histogram2d(ypos, xpos, bins = hist_bins, weights = bound)
		H_u, xedges, yedges = np.histogram2d(ypos_u, xpos_u, bins = hist_bins)
		H_b, xedges, yedges = np.histogram2d(ypos_b, xpos_b, bins = hist_bins)
		dumX = hist_bins[:n_hist_bins] + (hist_bins[1] - hist_bins[0])/2.
		X, Y = np.meshgrid(dumX, dumX)
		
		bin_side = (axis_range[1] - axis_range[0])/float(n_hist_bins)
		H_a = H_a * (1. / bin_side**3)/(270./box_side)**3
		H_ab = H_ab * (1. / bin_side**3)/(270./box_side)**3
		H_u = H_u * (1. / bin_side**3)/(270./box_side)**3
		H_b = H_b * (1. / bin_side**3)/(270./box_side)**3
		min_rho_per_bin = (1. / bin_side**3)/(270./box_side)**3

		print np.size(X), np.size(Y), np.size(H_b)
		
		logH_u = np.log10(H_u)
		logH_u = np.where(H_u > 0, np.log10(H_u), np.log10(min_rho_per_bin))
		logH_b = np.log10(H_b)
		logH_a0 = np.where(H_a > 0, 0., 1.)
		logH_a = np.where(H_a > 0, np.log10(H_a), np.log10(min_rho_per_bin))

		cdict1 = {'red':   ((0.0, 0.0, 0.0),
						   (0.5, 0.0, 0.1),
						   (1.0, 1.0, 1.0)),

				 'green': ((0.0, 0.0, 0.0),
						   (1.0, 0.0, 0.0)),

				 'blue':  ((0.0, 0.0, 1.0),
						   (0.5, 0.1, 0.0),
						   (1.0, 0.0, 0.0))
				}


		blue_red1 = colors.LinearSegmentedColormap('BlueRed1', cdict1)
		plt.register_cmap(cmap=blue_red1)


# 		xpos = np.where((-box_side/2. < xpos) & (xpos < box_side/2.), xpos, xpos - box_side/2.)
# 		ypos = np.where((-box_side/2. < ypos) & (ypos < box_side/2.), ypos, ypos - box_side/2.)
# 		zpos = np.where((-box_side/2. < zpos) & (zpos < box_side/2.), zpos, zpos - box_side/2.)

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .9, bottom = .12, left = .13, right = .9)
		ax = fig.add_subplot(1,1,1, aspect = 'equal')
# 		ax.xaxis.set_label_position('top') 
# 		ax.xaxis.tick_top()

		plt.axis(axis_range)
#		ax.set_xticks([11,12,13,14])

		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		plt.axis(axis_range)


		cmap = colors.ListedColormap(['blue', 'purple', 'red'])
		bounds=[0, 0.33, 0.67, 1.0]
		cmap_w = colors.ListedColormap(['white'])
#		cmap = colors.ListedColormap(['blue', 'red'])
#		bounds=[0, 0.5, 1.0]
		norm_b = colors.BoundaryNorm(bounds, cmap.N)


		density = logH_a
		boundedness = (H_ab/H_a + 1.)/2.
		norm = colors.Normalize(density.min(), density.max())
#		color_array = plt.get_cmap('BlueRed1')(boundedness)
		color_array = cmap(boundedness)
		white_array = cmap_w(logH_a0)
		print color_array.shape
		color_array[..., 3] = 0.5 + 0.5*norm(density)  # <- some alpha values between 0.5-1
		white_array[..., 3] = logH_a0  # <- some alpha values between 0.5-1
#		print color_array

		white_array
		
		im = plt.imshow(color_array, interpolation='spline16', 
                origin='lower', extent=axis_range)
		plt.imshow(white_array, interpolation='spline16',
                origin='lower', extent=axis_range, alpha = 0.5)

		
		cs = plt.contour(X, Y, logH_b, 10, cmap = cm.Reds)

		Rvir1=plt.Circle((0,0),Rvir,color='k',fill=False, lw = 3)
#			, label = r'$1\ \mathrm{R}_{\mathrm{vir}}$')
		Rvir6=plt.Circle((0,0),(nRvir + 1)*Rvir,color='k',fill=False, lw = 5, ls = 'dashed')
#			, label = r'$6\ \mathrm{R}_{\mathrm{vir}}$')
		ax.add_patch(Rvir1)
		ax.add_patch(Rvir6)

		text1 = r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}(z=0)/h^{-1} \mathrm{M}_{\odot}) = $' + str(mass_range)
		text2 = r'$z = $' + zed
		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$13$' + '\n' + text2, size = 34, color = 'k', 
			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))


#		divider = make_axes_locatable(ax)
#		cax = divider.append_axes("right", size="5%", pad=0.1)
#		cbar = plt.colorbar(im, cax = cax, ticks = [0.0, 0.5, 1.0], cmap = cm.hot)
#		cbar.set_label(r'$\mathrm{Bound}\ \mathrm{Fraction}$', fontsize=34) #54 for square


		
		ax.set_xlabel(r'$\mathrm{X}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) 
		ax.set_ylabel(r'$\mathrm{Y}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) #54 for square
# 		ax.legend(prop = matplotlib.font_manager.FontProperties(size=22),fancybox=True,loc=0)
#		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$13$' + '\n' + text2, size = 34, color = 'k', 
#			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
#		plt.text(12.5, -0.4, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Galaxy_Window_contours_' + str(mass_range) + '_' + gal + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def Galaxy_Window_contours_set(self, galnum, snaps, mass_range) :
		gal = "%03d" % galnum

		Rvir_a = []
		logH_aa = []
		logH_au = []
		logH_ab = []
		logH_abb = []

		axis_range = [-2., 2., -2., 2.]
		
		n_hist_bins = 20
		hist_bins = np.linspace(-2.,2., n_hist_bins + 1)

		win_size = axis_range[1] - axis_range[0]
		bin_side = (axis_range[1] - axis_range[0])/float(n_hist_bins)
		min_rho_per_bin = (1. / (bin_side**2 * win_size))/(270./box_side)**3
		print np.log10(min_rho_per_bin)
		dumX = hist_bins[:n_hist_bins] + (hist_bins[1] - hist_bins[0])/2.
		X, Y = np.meshgrid(dumX, dumX)


		for snapnum in snaps : 
			snap = "%03d" % snapnum
		
			Rvir, xpos, ypos, zpos, bound = res.read_fixed_window(galnum, mass_range, snapnum)
			Rvir_a.append(Rvir)


			print np.sum(bound), '= bound - unbound'
		
			ww = np.where((-2. < zpos) & (zpos < 2.))[0]
			print len(ww), 'particles shown out of', len(xpos)
			xpos = xpos[ww]
			ypos = ypos[ww]
			zpos = zpos[ww]
			bound = bound[ww]
		
			wb = np.where(bound > 0.)[0]
			xpos_b = xpos[wb]
			ypos_b = ypos[wb]
			zpos_b = zpos[wb]

			wu = np.where(bound < 0.)[0]
			xpos_u = xpos[wu]
			ypos_u = ypos[wu]
			zpos_u = zpos[wu]
		
			H_a, xedges, yedges = np.histogram2d(ypos, xpos, bins = hist_bins)
			H_u, xedges, yedges = np.histogram2d(ypos_u, xpos_u, bins = hist_bins)
			H_b, xedges, yedges = np.histogram2d(ypos_b, xpos_b, bins = hist_bins)

		
			H_a = H_a * (1. / (bin_side**2 * win_size))/(270./box_side)**3
			H_u = H_u * (1. / (bin_side**2 * win_size))/(270./box_side)**3
			H_b = H_b * (1. / (bin_side**2 * win_size))/(270./box_side)**3

			logH_a = np.where(H_a > 0, np.log10(H_a), np.log10(min_rho_per_bin))
			logH_u = np.where(H_u > 0, np.log10(H_u), np.log10(min_rho_per_bin))
			logH_b = np.log10(H_b)
			logH_bb = np.where(H_b > 0, np.log10(H_b), np.log10(min_rho_per_bin))

			logH_aa.append(logH_a)
			logH_au.append(logH_u)
			logH_ab.append(logH_b)
			logH_abb.append(logH_bb)
			
			


			print 'min, max all:', np.min(logH_a), np.max(logH_a)
			print 'min, max unbound:', np.min(logH_u), np.max(logH_u)
			print 'min, max bound:', np.min(logH_b), np.max(logH_b)


		logH_aa = np.array(logH_aa)
		logH_au = np.array(logH_au)
		logH_ab = np.array(logH_ab)
		logH_abb = np.array(logH_abb)

		fig = plt.figure(figsize=(20.,7))	
		fig.subplots_adjust(wspace = .0,top = .89, bottom = .12, left = .1, right = .95)

		for ii in range(len(snaps)) :
			snapnum = snaps[ii]
			zed = "%1.2f" % z[snapnum]


			ax = fig.add_subplot(1,4,ii + 1, aspect = 'equal')
			ax.xaxis.set_label_position('top') 
			ax.xaxis.tick_top()

			plt.axis(axis_range)
			if (ii == 3) : ax.set_xticks([-2, -1, 0, 1, 2])
			else : ax.set_xticks([-2, -1, 0, 1])
			if (ii == 0) : ax.set_yticks([-2, -1, 0, 1, 2])
			else : ax.set_yticks([])



			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)




			boundedness = (10**(logH_ab[ii] - logH_aa[ii]) + 1.)/2.
#			boundedness = np.where(boundedness > 0., boundedness, 0.)
			density_b = logH_abb[ii]
			norm = colors.Normalize(-0.5, 2.5) #logH_abb[ii].min(), logH_abb[ii].max())
			
						
			bound_array = plt.get_cmap('Greens')(norm(density_b)*0.9)

			print np.size(bound_array[..., 3]), np.size(boundedness)

			bound_array[..., 3] = boundedness*0.9  # <- some alpha values between 0.5-1
		
			im2 = plt.imshow(logH_ab[ii], interpolation='sinc', cmap=cm.Greens,
					origin='lower', extent=axis_range,
					vmax=logH_ab.max(), vmin=logH_au.min(), alpha = 0.7)
			im = plt.imshow(logH_au[ii], interpolation='sinc', cmap=cm.Purples,
					origin='lower', extent=axis_range,
					vmax=logH_au.max(), vmin=logH_au.min())
			plt.imshow(bound_array, interpolation='spline36', 
					origin='lower', extent=axis_range)

			levels = [1., 1.5, 2.0, 2.5, 3.0, 3.5]
			if (np.max(logH_ab) < 3.5) : levels = [1., 1.5, 2.0, 2.5, 3.0]
		
			cs = plt.contour(X, Y, logH_ab[ii], levels = levels, cmap = cm.YlGn)


			print Rvir_a[ii]
			Rvir1=plt.Circle((0,0),Rvir_a[ii],color='k',fill=False, lw = 3)
	#			, label = r'$1\ \mathrm{R}_{\mathrm{vir}}$')
			Rvir6=plt.Circle((0,0),(6.)*Rvir_a[ii],color='k',fill=False, lw = 5, ls = 'dashed')
	#			, label = r'$6\ \mathrm{R}_{\mathrm{vir}}$')
			ax.add_patch(Rvir1)
			ax.add_patch(Rvir6)

			text1 = r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}(z=0)/h^{-1} \mathrm{M}_{\odot}) = $' + str(mass_range)
			text2 = r'$z = $' + zed
			plt.text(-1.9, -1.9, text2, size = 34, color = 'k', 
				verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))


			divider = make_axes_locatable(ax)
			cax2 = divider.append_axes("bottom", size="5%", pad=0.1)
			if (ii % 2 == 0) : 
#				cax = divider.append_axes("right", size="5%", pad=0.1)
				cbar = plt.colorbar(im, cax = cax2, orientation='horizontal', ticks = [-0.5, 0.0, 0.5, 1, 2.0, 3.0])
				cbar.set_label(r'$\mathrm{log}_{10}(\rho_{\mathrm{Unbound}}/\bar{\rho})$', fontsize=34) #54 for square
			else :
				cbar2 = plt.colorbar(im2, cax = cax2, orientation='horizontal', ticks = [0.0, 1.0, 2.0, 3.0, 4.0])
				cbar2.set_label(r'$\mathrm{log}_{10}(\rho_{\mathrm{Bound}}/\bar{\rho})$', fontsize=34) #54 for square


		
			ax.set_xlabel(r'$\mathrm{X}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) 
			if (ii == 0) : ax.set_ylabel(r'$\mathrm{Y}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) #54 for square
	# 		ax.legend(prop = matplotlib.font_manager.FontProperties(size=22),fancybox=True,loc=0)
	#		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$13$' + '\n' + text2, size = 34, color = 'k', 
	#			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
	#		plt.text(12.5, -0.4, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')

		outputFile = plot_dir + 'Evolution_Galaxy_Window_contours_' + str(mass_range) + '_' + gal + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def plot_pvalues(self, N1, N2, pvalues, r_FA) :
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([0,2.05,-0.1,1])

		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		label1 = ''
		label2 = ''
		if N1 == 'Mdiff' :
			label1 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
		if N1 == 'Mvir' :
			label1 = r'$\mathrm{M}_{\mathrm{vir}}$'

		if N2 == 'Mdiff' :
			label2 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
		if N2 == str(r_FA) + '.0FA' :
			label2 = r'$\mathrm{FA}$'

		for rr in range(1, 10) :
			label = str(rr + 1) + r'$\mathrm{R}_{\mathrm{vir}}$'
			col = cm.jet((rr-1)/8.)
			ax.plot(z, pvalues[rr], label = label, c = col)
		

		ax.set_xlabel(r'$z$', fontsize=34) #54 for square
		ax.set_ylabel(r'$\mathrm{Pearson}\ \mathrm{p-value}$ (' + label1 + '$\mathrm{vs}$ ' + label2 + ')', fontsize=34) #54 for square
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0, ncol = 3)

		outputFile = plot_dir + N1 + '_vs_' + N2 + '_' + space + '_pvalues' + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def plot_pvalues_Mbins(self, Mvir, N1, N2, pvalues_bins, r_FA) :

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = 0.15, hspace = 0.05, top = .96, bottom = .12, left = .12, right = .95)

		for mm in range(4) : 
			Mmin = 10. #np.min(logMvir)
			bin = 1. #(np.max(logMvir) - np.min(logMvir))/4.
			Mbin_min = "%2.1f" % (Mmin + mm*bin)
			Mbin_max = "%2.1f" % (Mmin + (mm + 1)*bin)

			ax = fig.add_subplot(2,2,mm + 1)

			plt.axis([0,2.05,-0.1,1])

			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(18)

			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(18)
				
			if mm < 2 : ax.set_xticklabels([])
#			if mm == 1 : ax.set_yticklabels([])
#			if mm == 3 : ax.set_yticklabels([])

			label1 = ''
			label2 = ''
			if N1 == 'Mdiff' :
				label1 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
			if N1 == 'Mvir' :
				label1 = r'$\mathrm{M}_{\mathrm{vir}}$'

			if N2 == 'Mdiff' :
				label2 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
			if N2 == str(r_FA) + '.0FA' :
				label2 = r'$\mathrm{FA}$'

			for rr in range(1, 10) :
				label = str(rr + 1) + r'$\mathrm{R}_{\mathrm{vir}}$'
				col = cm.jet((rr-1)/8.)
				ax.plot(z, pvalues_bins[mm,rr], label = label, c = col)
		

			ax.set_xlabel(r'$z$', fontsize=34) #54 for square
#			ax.set_ylabel(r'$\mathrm{Pearson}\ \mathrm{p-value}$ (' + label1 + '$\mathrm{vs}$ ' + label2 + ')', fontsize=34) #54 for square
			if mm == 2 : 
				ax.legend(prop = matplotlib.font_manager.FontProperties(size=18),
					fancybox=True,loc=0, ncol = 3, borderpad = 0.1, columnspacing = 0.1)
			log = r'$\mathrm{log}_{10}$'
			Msun = r'$\mathrm{M}_{\odot}$'

			text2 = Mbin_min + r'$< (\mathrm{M}_{\mathrm{vir}}/ h^{-1} \mathrm{M}_{\odot}) <$' + Mbin_max
			plt.text(2.0, 0.96, text2, size = 20, color = 'k', 
				verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.9))
			ylabel = r'$\mathrm{Pearson}\ \mathrm{p-value}$ (' + label1 + '$\mathrm{vs}$ ' + label2 + ')'
		fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', size = 30)

		outputfile = plot_dir + N1 + '_vs_' + N2 + '_' + space + '_pvalues_bins' 
		if shuffle == 'All' : outputFile = outputfile + '_shuffle'
		if shuffle == 'Mbins' : outputFile = outputfile + '_shuffle_Mbins'
		outputFile = outputFile + OutputFormat
		print 'Saving file to', outputFile
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

	def plot_pvalues_rhobins(self, Mvir, N1, N2, pvalues_bins, r_FA) :

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = 0.15, hspace = 0.05, top = .96, bottom = .12, left = .12, right = .95)

		for pp in range(4) : 
			rhomin = -0.12 #np.min(logMvir)
			bin = 0.65 #(np.max(logMvir) - np.min(logMvir))/4.
			rhobin_min = "%1.2f" % (rhomin + pp*bin)
			rhobin_max = "%1.2f" % (rhomin + (pp + 1)*bin)

			ax = fig.add_subplot(2,2,pp + 1)

			plt.axis([0,2.05,-0.1,1])

			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(18)

			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(18)
				
			if pp < 2 : ax.set_xticklabels([])
#			if pp == 1 : ax.set_yticklabels([])
#			if pp == 3 : ax.set_yticklabels([])

			label1 = ''
			label2 = ''
			if N1 == 'Mdiff' :
				label1 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
			if N1 == 'Mvir' :
				label1 = r'$\mathrm{M}_{\mathrm{vir}}$'

			if N2 == 'Mdiff' :
				label2 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
			if N2 == str(r_FA) + '.0FA' :
				label2 = r'$\mathrm{FA}$'

			for rr in range(1, 10) :
				label = str(rr + 1) + r'$\mathrm{R}_{\mathrm{vir}}$'
				col = cm.jet((rr-1)/8.)
				ax.plot(z, pvalues_bins[pp,rr], label = label, c = col)
		

			ax.set_xlabel(r'$z$', fontsize=34) #54 for square
#			ax.set_ylabel(r'$\mathrm{Pearson}\ \mathrm{p-value}$ (' + label1 + '$\mathrm{vs}$ ' + label2 + ')', fontsize=34) #54 for square
			if pp == 0 : 
				ax.legend(prop = matplotlib.font_manager.FontProperties(size=18),
					fancybox=True,loc=0, ncol = 3, borderpad = 0.1, columnspacing = 0.1)
			log = r'$\mathrm{log}_{10}$'
			Msun = r'$\mathrm{M}_{\odot}$'

			text2 = rhobin_min + r'$< (\rho_{\mathrm{FA}} / \bar{\rho}) <$' + rhobin_max
			plt.text(2.0, 0.96, text2, size = 20, color = 'k', 
				verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.9))
			ylabel = r'$\mathrm{Pearson}\ \mathrm{p-value}$ (' + label1 + '$\mathrm{vs}$ ' + label2 + ')'
		fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', size = 30)

		outputfile = plot_dir + N1 + '_vs_' + N2 + '_' + space + '_pvalues_bins' 
		if shuffle == 'All' : outputFile = outputfile + '_shuffle'
		if shuffle == 'Mbins' : outputFile = outputfile + '_shuffle_Mbins'
		if shuffle == 'None' : outputFile = outputfile
		outputFile = outputFile + OutputFormat
		print 'Saving file to', outputFile
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()

		
		


if __name__ == '__main__':
	rd = ReadData()
	res = Results()
	





#	snaps = [32, 40, 47, 63]
# 	snaps = [63, 47, 40, 32]
# 	galnum = 0
# 	mass_range = 14
# 	res.Galaxy_Window_contours_set(galnum, snaps, mass_range)


# 	for gg in range(5) :
# 		galnum = gg
# 		for snapnum in snaps :
# #		snapnum = 63
# 			nRvir = 5
# 			mass_range = 14
# 			Rvir, xpos, ypos, zpos, bound = res.read_window(galnum, mass_range)
# 			res.Galaxy_Window_contours(galnum, snapnum, mass_range, nRvir, Rvir, xpos, ypos, zpos, bound)
# 			print
# 		print






	snaps = [32, 63]
	ns = [2, 3, 5]
	r_FAs = [1, 2, 5]

	N_unbound = rd.read_unbound(z)

	for snapnum in snaps : 
		print "Reading in Galaxies."
		type, Mvir, Mstars, ColdGas, IDs = rd.read_gals(snapnum)
		print 'highest stellar mass = ', np.log10(np.max(Mstars))
		print "Reading in unbound matter in a fixed aperture." 
		Mdiffuse_FA = rd.read_aperture_distributions(snapnum, len(type))
		print "Reading in unbound matter in a radial aperture." 
		Mdiffuse_Rvir = rd.read_radial_distributions(snapnum, len(type))
		print "Reading in fixed aperture galaxy counts." 
		rhorhobar_FA = rd.read_fixed_aperture(snapnum, len(type))

		for nRvir in ns :
# 			res.Mstars_vs_Mdiffuse_radial(snapnum, nRvir, type, Mstars, Mdiffuse_Rvir)			
#			res.Mvir_vs_Mdiffuse_radial(snapnum, nRvir, type, Mvir, Mdiffuse_Rvir)			
			for r_FA in r_FAs :
#				res.rho_vs_Mdiffuse_radial(snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse_Rvir)
#				res.rho_vs_Mdiffuse_radial_Mstar_bins(snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse_Rvir, Mstars)
				res.Mstar_vs_Mdiffuse_radial_rho_bins(snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse_Rvir, Mstars)

#		for r_FA in r_FAs :
#			res.Mstars_vs_Mdiffuse_aperture(snapnum, r_FA, type, Mstars, Mdiffuse_FA)			
#			res.Mvir_vs_Mdiffuse_aperture(snapnum, r_FA, type, Mvir, Mdiffuse_FA)			
#			res.rho_vs_Mdiffuse_aperture(snapnum, r_FA, type, rhorhobar_FA, Mdiffuse_FA)
#			res.rho_vs_Mdiffuse_aperture_Mstar_bins(snapnum, r_FA, type, rhorhobar_FA, Mdiffuse_FA, Mstars)






# 		for nRvir in range(10) :
# 			Mv = Mvir
# 			Ms = Mstars
# 			Md = Mdiffuse[:, nRvir]
# 			env = rhorhobar_FA
# 			V1 = Md
# 			V2 = env
# #			res.calculate_pvalues(snapnum, V1, V2, N1, N2, nRvir, type)
# # 			res.calculate_pvalues_Mbins(snapnum, Mvir, V1, V2, N1, N2, nRvir, type)
# 			res.calculate_pvalues_rhobins(snapnum, rhorhobar_FA, V1, V2, N1, N2, nRvir, type)



# 	print pvalues[:, -1]
# 	f = open(halo_dir + N1 + '_vs_'  + N2 + '_' + space + '_pvalues.dat','w')
# 	for nR in range(10) : 			
# 		for pval in pvalues[nR] : 
# 			f.write(str(pval) + '\t')
# 		f.write('\n')
# 	f.close() 
# 
# 
# 	r_FA = 2
# 	N1 = 'Mdiff'
# 	N2 = str(r_FA) + '.0FA'
# 	res.read_pvalues(N1, N2)
# #	res.plot_pvalues(N1, N2, pvalues, r_FA)
# 	res.plot_pvalues_Mbins(Mvir, N1, N2, pvalues_bins, r_FA)
# 	res.plot_pvalues_rhobins(rhorhobar_FA, N1, N2, pvalues_bins, r_FA)
# 
# 
# 	for nRvir in range(10) :
# 		for snapnum in range(32, 64) : 
# 			type, Mvir, Mstars, ColdGas, IDs = res.read_gals(snapnum)
# 			r_FA = 2.
# 			Mdiffuse = res.read_distributions(snapnum, len(type))
# 			N_FA, rhorhobar_FA = res.read_fixed_aperture(snapnum, len(type), r_FA)
# # 			res.Mvir_vs_Mdiffuse_vs_FA_bins(snapnum, nRvir, type, Mvir, Mdiffuse[:,nRvir], rhorhobar_FA)					
# 			res.Mdiffuse_vs_FA(snapnum, nRvir, type, Mdiffuse[:, nRvir], rhorhobar_FA)
# 	f = open(halo_dir + 'Mdiff_vs_'  + str(r_FA) + 'FA_' + space + '_pvalues.dat','w')
# 	for nR in range(10) : 			
# 		for pval in pvalues[nR] : 
# 			f.write(str(pval) + '\t')
# 		f.write('\n')
# 	f.close() 
# 
# 		
# 		res.Mvir_Mdiff_FA_evolution(z_fits, z_fits_err, z_fits_fixed, z_fits_fixed_err, z_colors)
# 
# 
# 
# 
# 	snaps = [32, 38, 44, 63]
# 	res.Find_Galaxies_to_Evolve(snaps)
# 
# 		nRvir = 5
# 		res.Mvir_vs_Mdiffuse_vs_FA(snapnum, nRvir, type, Mvir, Mdiffuse[:,nRvir], rhorhobar_FA)
# 		res.Mvir_vs_Mdiffuse(snapnum, nRvir, type, Mvir, Mdiffuse[:, nRvir])
# 		res.Mvir_vs_Mdiffuse_vs_FA_bins(snapnum, nRvir, type, Mvir, Mdiffuse[:,nRvir], rhorhobar_FA)
# 		res.Mvir_vs_FA(snapnum, type, Mvir, rhorhobar_FA)
# #		res.Mvir_vs_Mdiffuse_vs_FA_3D(snapnum, nRvir, type, Mvir, Mdiffuse[:,nRvir], rhorhobar_FA, Mstars)
# 
# 	nRvir = 5
# 	snapnum = 63
# 	nframes = 72
# 	res.Mvir_vs_Mdiffuse_vs_FA_3D_movie(snapnum, nRvir, type, Mvir, Mdiffuse[:,nRvir], rhorhobar_FA, Mstars, nframes)
# 
# 
# #		res.Mvir_Mdiffuse_scatter_vs_FA(snapnum, nRvir, type, Mvir, Mdiffuse[:,nRvir], rhorhobar_FA)
# #		res.Mvir_vs_Mdiffuse_vs_FA(snapnum, nRvir, type, Mvir, Mdiffuse[:,nRvir], rhorhobar_FA)
# 
