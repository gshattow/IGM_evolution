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
from scipy.stats import chisquare

# ================================================================================
# Basic variables
# ================================================================================

# Set up some basic attributes of the run

SIM = 'MM'
space = 'comoving'
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
mu = r'M$_{\mathrm{unbound}}$'
vd = r'$\sigma$'
ms = r'M$_{\star}$'
cg = r'M$_{\mathrm{ColdGas}}$'
log10 = r'log$_{10}$'
msun = r'$h^{-1}\mathrm{M}_{\odot}$'
mpc = r'$h^{-1}\mathrm{Mpc}$'
kms = r'$\mathrm{km}\ \mathrm{s^{-1}}$'
rv = r'R$_{\mathrm{vir}}$'
rhob = r'$\rho/\bar{\rho}$'
fb = r'$f_b$'
zz = r'Z/Z$_{\odot}$'
sfr = 'SFR'
hpy = r'$h\ \mathrm{yr}^{-1}$'




def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return colors.LinearSegmentedColormap('CustomMap', cdict)

c = colors.ColorConverter().to_rgb
rvb = make_colormap(
    [c('black'), c('red'), 0.33, c('red'), c('violet'), 0.66, c('violet'), c('blue'), ])
    

def line_fixed_slope(x, b) :
	return 1. * x + b 

def line(x, m, b) :
	return m * x + b 


class ReadData :

	def read_gals(self, snapnum) :
		snap = "%03d" % snapnum
		galfile = halo_dir + 'halos_109' + '_' + snap
		type = []
		Mvir = []
		Rvir = []
		Mstars = []
		ColdGas = []
		IDs = []
		Vdisp = []
		GasMetals = []
		SFR = []
		for item in file(galfile) :
			item = item.split()
			type.append(int(item[0]))
			Mvir.append(float(item[3])*1.0e10)
			gas = float(item[6])
			stars = float(item[7])
			ColdGas.append(gas*1.0e10)
			Mstars.append(stars*1.0e10)
			IDs.append(np.int64(item[1]))
			Vdisp.append(float(item[5]))
			if gas > 0. : GasMetals.append(float(item[8])/float(item[6]))
			else : GasMetals.append(0.)
			SFR.append(float(item[9]))

	
		type = np.array(type)
		Mvir = np.array(Mvir)
#		Rvir = np.array(Rvir)
		Mstars = np.array(Mstars)
		ColdGas = np.array(ColdGas)
		IDs = np.array(IDs)
		Vdisp = np.array(Vdisp)
		GasMetals = np.array(GasMetals)
		SFR = np.array(SFR)
		
#		print np.sum(ColdGas), 'Msun Cold gas at z=', z[snapnum]

		print 'min, max Metals (Gas):', np.log10(min(GasMetals)/0.02), np.log10(max(GasMetals)/0.02)


		return type, Mvir, Mstars, ColdGas, IDs, Vdisp, GasMetals, SFR

	def read_radial_distributions(self, snapnum, ngals) :
		snap = "%03d" % snapnum
		pclfile = pcl_dir + 'unbound_particles_around_halos_Rvir' + '_' + snap
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
		pclfile = pcl_dir + 'unbound_particles_around_halos_FA_' + space + '_' + snap
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
		pclfile = pcl_dir + 'diffuse_halos_rvir_' + space + '_' + snap
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
		
		
		rhobar = ngals/(box_side**3)
		if space == 'physical' : rhobar = rhobar * (1. + z[snapnum])**3
		rhorhobar_FA = cumulative/volume/rhobar
		
		print rhorhobar_FA[:,9]
		w = np.where(rhorhobar_FA[:,9] > 1.)[0]
		print len(w), 'galaxies out of', ngals, 'have overdense 5Mpc envs'

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


#____________Mvir____________#(6)


	def Mvir_vs_Mdiffuse_radial(self, snapnum, nRvir, type, Mvir, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , nRvir - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		print 'plotting Mvir vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'

		points = np.linspace(8, 15, 100)
		fit = np.polyfit(logMvir, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		p1 = np.poly1d(np.polyfit(logMvir, logMdiff,  1))

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1


		Hz = np.sqrt(0.7 + 0.3*(1 + z[snapnum])**3)
		Rpoints = (10**points / (2.32443891e14 * Hz**2))**(1./3.)
		mean_density_diff = N_unbound[snapnum]/(box_side**3)*pcl_mass
		rad_expected_mass = 4./3.*np.pi*Rpoints**3*(nRvir**3)*mean_density_diff


		x_label = log10 + '(' + mv + '/' + msun + ')'
		y_label = log10 + '(' + mu + '/' + msun + ')'
		label = '< ' + str(nRvir) + rv
		col = cm.spectral((nRvir-1)/7.)


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([9.5,14.1,9.5,14.1])
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		ax.scatter(logMvir, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		ax.plot(points, np.log10(rad_expected_mass), c = 'k', lw = 4, ls = ':', label = 'Expected Mass')

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 

		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=2)
		plt.text(14, 9.6, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='right')

		outputFile = plot_dir + 'Mvir_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  
		print 'Saved file to', outputFile
		plt.close()	

	def Mvir_vs_Mdiffuse_aperture(self, snapnum, r_FA, type, Mvir, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]

		rindex = int(r_FA*2) - 1
		Mdiff = Mdiffuse[: , rindex]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logMvir = np.log10(Mvir[w])
		logMdiff = np.log10(Mdiff[w])
		print 'plotting Mvir vs Mdiffuse within', r_FA, 'Mpc/h for', len(w), 'galaxies'


		points = np.linspace(8, 15, 100)
		fit = np.polyfit(logMvir, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		p1 = np.poly1d(np.polyfit(logMvir, logMdiff,  1))

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1


		ub_pcls = float(N_unbound[snapnum])
		rho_mass = ub_pcls*pcl_mass/box_side**3
		radius = r_FA
		if space == 'physical' : radius = radius*(1. + z[snapnum])
		ap_volume = 4./3.*np.pi*radius**3
		ap_expected_mass = rho_mass*ap_volume


		x_label = log10 + '(' + mv + '/' + msun + ')'
		y_label = log10 + '(' + mu + '/' + msun + ')'
		label = mu + '(<' + str(r_FA) + mpc + ')'
		col = cm.gnuplot((r_FA+2)/8.)


		
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([9.5,14.1,11.,15.6])

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		ax.scatter(logMvir, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		ax.axhline(y = np.log10(ap_expected_mass), c = 'k', lw = 4, ls = ':', label = 'Expected Mass')

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		
		if (r_FA == 1.) : loc = 1
		if ((r_FA == 2.) & (snapnum == 63)) : loc = 1
		if ((r_FA == 2.) & (snapnum == 32)) : loc = 4
		if (r_FA == 5.) : loc = 4
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		plt.text(9.7, 15.5, '$z = $' + zed, size = 40, color = 'k', 
			verticalalignment='top', horizontalalignment='left')

		outputFile = plot_dir + 'Mvir_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  
		print 'Saved file to', outputFile
		plt.close()	

	def Mvir_vs_Mdiffuse_radial_all(self, snapnum, type, Mvir, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)

		
		ii = 1
		for nRvir in ns : 
			Mdiff = Mdiffuse[: , nRvir - 1]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logMvir = np.log10(Mvir[w])
			logMdiff = np.log10(Mdiff[w])
			print 'plotting Mvir vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'


			points = np.linspace(8, 15, 100)
			fit = np.polyfit(logMvir, logMdiff, 1)
			fit_points = points * fit[0] + fit[1]
			p1 = np.poly1d(np.polyfit(logMvir, logMdiff,  1))

			fit_0 = "%1.2f" % fit[0]
			fit_1 = "%1.2f" % fit[1]
			fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1


			Hz = np.sqrt(0.7 + 0.3*(1 + z[snapnum])**3)
			Rpoints = (10**points / (2.32443891e14 * Hz**2))**(1./3.)
			mean_density_diff = N_unbound[snapnum]/(box_side**3)*pcl_mass
			rad_expected_mass = 4./3.*np.pi*Rpoints**3*(nRvir**3)*mean_density_diff

			label = '< ' + str(nRvir) + rv
			col = cm.spectral((nRvir-1)/7.)
			x_label = log10 + '(' + mv + '/' + msun + ')'
			y_label = log10 + '(' + mu + '/' + msun + ')'


			ax = fig.add_subplot(1,len(ns),ii)
			plt.axis([10.1,14.1,9.7,14.2])
			ax.xaxis.set_ticks([11, 12, 13, 14])
			ax.yaxis.set_ticks([11, 12, 13, 14])

	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)


			ax.scatter(logMvir, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
			ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)

			if ii == 1 : ex_label = 'Expected Mass'
			else : ex_label = ''

			ax.plot(points, np.log10(rad_expected_mass), c = 'k', lw = 4, ls = ':', label = ex_label)
		
			ax.set_xlabel(x_label, fontsize=34) 
			if ii == 1 : ax.set_ylabel(y_label, fontsize=34) 
			leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=2)
			leg.get_frame().set_alpha(0.5)
			if ii == 3 : plt.text(14, 9.7, '$z = $' + zed, size = 40, color = 'k', 
				verticalalignment='bottom', horizontalalignment='right')


			ii = ii + 1


		outputFile = plot_dir + 'Mvir_vs_Mdiff_ra_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile) 
		print 'Saved file to', outputFile
		plt.close()	

	def Mvir_vs_Mdiffuse_aperture_all(self, snapnum, type, Mvir, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)


		ii = 1
		for r_FA in r_FAs : 


			rindex = int(r_FA*2) - 1
			Mdiff = Mdiffuse[: , rindex]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logMvir = np.log10(Mvir[w])
			logMdiff = np.log10(Mdiff[w])
			print 'plotting Mvir vs Mdiffuse within', r_FA, 'Mpc/h for', len(w), 'galaxies'


			points = np.linspace(8, 15, 100)
			fit = np.polyfit(logMvir, logMdiff, 1)
			fit_points = points * fit[0] + fit[1]
			p1 = np.poly1d(np.polyfit(logMvir, logMdiff,  1))

			fit_0 = "%1.2f" % fit[0]
			fit_1 = "%1.2f" % fit[1]
			fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1

			ub_pcls = float(N_unbound[snapnum])
			rho_mass = ub_pcls*pcl_mass/box_side**3
			radius = r_FA
			if space == 'physical' : radius = radius*(1. + z[snapnum])
			ap_volume = 4./3.*np.pi*radius**3
			ap_expected_mass = rho_mass*ap_volume



			ax = fig.add_subplot(1,len(r_FAs),ii)
			plt.axis([10.1,14.1,10.8,14.6])
			ax.xaxis.set_ticks([11, 12, 13, 14])
			ax.yaxis.set_ticks([11, 12, 13, 14])

	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)

			label = '< ' + str(r_FA) + mpc
			col = cm.gnuplot((r_FA+2)/8.)
			x_label = log10 + '(' + vd + '/' + kms + ')'
			y_label = log10 + '(' + mu + '/' + msun + ')'





			ax.scatter(logMvir, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)

			ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
			if ii == 3 : ex_label = 'Expected Mass'
			else : ex_label = ''
			ax.axhline(y = np.log10(ap_expected_mass), c = 'k', lw = 4, ls = ':', label = ex_label)


		
			loc = 2
			if (r_FA == 5.) : loc = 4
			ax.set_xlabel(x_label, fontsize=34) 
			if ii == 1 : ax.set_ylabel(y_label, fontsize=34) 
			ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
			if ii == 2 : plt.text(10.2, 10.8, '$z = $' + zed, size = 40, color = 'k', 
				verticalalignment='bottom', horizontalalignment='left')

			ii = ii + 1

		outputFile = plot_dir + 'Mvir_vs_Mdiff_ap_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile) 
		print 'Saved file to', outputFile
		plt.close()	

	def Mvir_vs_Mdiffuse_freefall_radial(self, snapnum, type, Mvir, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		tff0 = np.pi/(20.*np.sqrt(2))*1.

		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)

		ii = 1
		for nRvir in ns:

			Mdiff = Mdiffuse[: , nRvir - 1]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logMvir = np.log10(Mvir[w])
			logMdiff = np.log10(Mdiff[w])
			print 'plotting Mvir vs Mdiffuse freefall within', nRvir, 'Rvir for', len(w), 'galaxies'

			tff = np.sqrt(nRvir**3/(1. + Mdiff[w]/Mvir[w]))*tff0

			col = cm.spectral((nRvir-1)/7.)
			label = '< ' + str(nRvir) + rv
			x_label = log10 + '(' + mv + '/' + msun + ')'
			y_label = r'$t_{ff}^{\mathrm{unbound}}/t_{\mathrm{H}(z)}$'

			ax = fig.add_subplot(1,len(ns),ii)
			plt.axis([9.5,14.1,0.1,1.5])
	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)




			ax.scatter(logMvir, tff, c = col, edgecolor = col, alpha = 0.3, label = label)

		

			ax.set_xlabel(x_label, fontsize=34) 
			if (ii == 1) : ax.set_ylabel(y_label, fontsize=34) 

			if ii == 1 : ax.plot(0., 0., marker = 'None', ls = 'None', label = r'$z = $' + zed)

			if ii == 3 : H_label = 'Hubble Time'
			else : H_label = ''
			ax.axhline(y = 1., c = 'k', lw = 4, ls = ':', label = H_label)

			loc = 2
			if ii == 3 : loc = 4
			ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
			ii = ii + 1


		outputFile = plot_dir + 'freefall_Mvir_vs_Mdiff_ra_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mvir_vs_Mdiffuse_freefall_aperture(self, snapnum, type, Mvir, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Hz = np.sqrt(0.7 + 0.3*(1 + z[snapnum])**3)
		tff0 = np.pi/(20.*np.sqrt(2))*1.



		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)

		ii = 1
		for r_FA in r_FAs:

			Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logMvir = np.log10(Mvir[w])
			logMdiff = np.log10(Mdiff[w])
			print 'plotting Mvir vs Mdiffuse freefall within', r_FA, 'Mpc for', len(w), 'galaxies'

			R_vir = (4.3e-15*Mvir[w])**(1./3.)
			if space == 'comoving' : R_vir = R_vir*(1. + z[snapnum])
			nRvir = r_FA/R_vir

			tff = np.sqrt(nRvir**3/(1. + Mdiff[w]/Mvir[w]))*tff0



			label = mu + '(<' + str(r_FA) + mpc + ')'
			col = cm.gnuplot((r_FA+2)/8.)
			x_label = log10 + '(' + mv + '/' + msun + ')'
			y_label = r'$t_{ff}^{\mathrm{unbound}}/t_{\mathrm{H}(z)}$'


			ax = fig.add_subplot(1,len(ns),ii)
			plt.axis([9.5,14.1,0.1,5.5])
	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)


			ax.scatter(logMvir, tff, c = col, edgecolor = col, alpha = 0.3, label = label)

		

			ax.set_xlabel(x_label, fontsize=34) 
			if (ii == 1) : ax.set_ylabel(y_label, fontsize=34) 

			if ii == 3 : ax.plot(0., 0., marker = 'None', ls = 'None', label = r'$z = $' + zed)

			if ii == 3 : H_label = 'Hubble Time'
			else : H_label = ''
			ax.axhline(y = 1., c = 'k', lw = 4, ls = ':', label = H_label)

			loc = 2
			if snapnum > 60 : loc = 3
			leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
			leg.get_frame().set_alpha(0.5)
			ii = ii + 1


		outputFile = plot_dir + 'freefall_Mvir_vs_Mdiff_ap_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	


#____________Mstars____________#(13)

	def Mstars_vs_Mdiffuse_radial(self, snapnum, nRvir, type, Mstars, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , nRvir - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logMstars = np.log10(Mstars[w])
		logMdiff = np.log10(Mdiff[w]*f_b)
		print 'plotting Mstars vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'

		points = np.linspace(8, 15, 100)
		fit = np.polyfit(logMstars, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		p1 = np.poly1d(np.polyfit(logMstars, logMdiff,  1))
# 		X1 = chisquare(logMdiff, f_exp=p1(logMstars), ddof=0, axis=0)
# 		print 'X1 =', X1
# 
# 
# 		fit2 = np.polyfit(logMstars, logMdiff, 2)
# 		fit_points2 = points * points * fit2[0] + points * fit2[1] + fit2[2]
# 		p2 = np.poly1d(np.polyfit(logMstars, logMdiff,  2))
# 		X2 = chisquare(logMdiff, f_exp=p2(logMstars), ddof=0, axis=0)
# 		print 'X2 =', X2

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1
# 		fit_label = log10 + mu + r'$\times f_b$' + ' = ' + "\n\t" + fit_0 + ' x ' + log10 + ms + ' + ' + fit_1 


		label = mu + '(<' + str(nRvir) + rv + ')'
		col = cm.spectral((nRvir-1)/7.)
		x_label = log10 + '(' + ms + '/' + msun + ')'
		y_label = log10 + '(' + r'$f_b \times$ ' + mu  + '/' + msun + ')'

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([8.5,13.1,8.5,13.1])

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)

		ax.scatter(logMstars, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
		

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=4)
		plt.text(8.7, 13, '$z = $' + zed, size = 40, color = 'k', 
			verticalalignment='top', horizontalalignment='left')

		outputFile = plot_dir + 'Mstars_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_aperture(self, snapnum, r_FA, type, Mstars, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		

		rindex = int(r_FA*2) - 1
		Mdiff = Mdiffuse[: , rindex]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logMstars = np.log10(Mstars[w])
		logMdiff = np.log10(Mdiff[w]*f_b)
		print 'plotting Mstars vs Mdiffuse within', r_FA, 'Mpc/h for', len(w), 'galaxies'

		points = np.linspace(8, 15, 100)
		fit = np.polyfit(logMstars, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		p1 = np.poly1d(np.polyfit(logMstars, logMdiff,  1))
		X1 = chisquare(logMdiff, f_exp=p1(logMstars), ddof=0, axis=0)
		print 'X1 =', X1

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1


		label = '< ' + str(r_FA) + mpc
		col = cm.gnuplot((r_FA+2)/8.)
		x_label = log10 + '(' + ms + '/' + msun + ')'
		y_label = log10 + '(' + r'$f_b \times$ ' + mu  + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([8.5,12.1,10.1,14.6])

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)



		ax.scatter(logMstars, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)


		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		ax.axhline(y = np.log10(ap_expected_mass), c = 'k', lw = 4, ls = ':', label = 'Expected Mass')



		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=4)
		plt.text(8.7, 14.5, '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='left')

		outputFile = plot_dir + 'Mstars_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_radial_all(self, snapnum, type, Mstars, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		

		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)

		ii = 1
		for nRvir in ns:

			Mdiff = Mdiffuse[: , nRvir - 1]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logMstars = np.log10(Mstars[w])
			logMdiff = np.log10(Mdiff[w]*f_b)
			print 'plotting Mvir vs Mdiffuse freefall within', nRvir, 'Rvir for', len(w), 'galaxies'


			points = np.linspace(8, 15, 100)
			fit = np.polyfit(logMstars, logMdiff, 1)
			fit_points = points * fit[0] + fit[1]
			p1 = np.poly1d(np.polyfit(logMstars, logMdiff,  1))
			X1 = chisquare(logMdiff, f_exp=p1(logMstars), ddof=0, axis=0)
			print 'X1 =', X1

			fit_0 = "%1.2f" % fit[0]
			fit_1 = "%1.2f" % fit[1]
			fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1

			label = '(<' + str(nRvir) + rv + ')'
			col = cm.spectral((nRvir-1)/7.)
			x_label = log10 + '(' + ms + '/' + msun + ')'
			y_label = log10 + '(' + r'$f_b \times$ ' + mu  + '/' + msun + ')'


			ax = fig.add_subplot(1,len(ns),ii)
			plt.axis([8.5,12.1,8.5,13.2])
	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)

			ax.scatter(logMstars, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)

			ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)

			ax.set_xlabel(x_label, fontsize=34) 
			if ii == 1 : ax.set_ylabel(y_label, fontsize=34) 
			ax.xaxis.set_ticks([9, 10, 11, 12])
			ax.yaxis.set_ticks([9, 10, 11, 12, 13])

			loc = 2
			if ii == 3 : loc = 4
			ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
			if ii == 2 : plt.text(11.8, 8.6, '$z = $' + zed, size = 40, color = 'k', 
				verticalalignment='bottom', horizontalalignment='right')

			ii = ii + 1

		outputFile = plot_dir + 'Mstars_vs_Mdiff_ra_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_aperture_all(self, snapnum, type, Mstars, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		

		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)

		ii = 1
		for r_FA in r_FAs:

			Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logMstars = np.log10(Mstars[w])
			logMdiff = np.log10(Mdiff[w]*f_b)
			print 'plotting Mvir vs Mdiffuse freefall within', r_FA, 'Mpc for', len(w), 'galaxies'

			points = np.linspace(8, 15, 100)
			fit = np.polyfit(logMstars, logMdiff, 1)
			fit_points = points * fit[0] + fit[1]
			p1 = np.poly1d(np.polyfit(logMstars, logMdiff,  1))
			X1 = chisquare(logMdiff, f_exp=p1(logMstars), ddof=0, axis=0)
			print 'X1 =', X1

			fit_0 = "%1.2f" % fit[0]
			fit_1 = "%1.2f" % fit[1]
			fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1


			label = '(<' + str(r_FA) + mpc + ')'
			col = cm.gnuplot((r_FA+2)/8.)
			x_label = log10 + '(' + ms + '/' + msun + ')'
			y_label = log10 + '(' + r'$f_b \times$ ' + mu  + '/' + msun + ')'


			ax = fig.add_subplot(1,len(ns),ii)
			plt.axis([8.5,11.6,9.8,13.6])
	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)

			ax.scatter(logMstars, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)

			ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)

			ax.set_xlabel(x_label, fontsize=34) 
			if ii == 1 : ax.set_ylabel(y_label, fontsize=34) 
			ax.xaxis.set_ticks([9, 10, 11, 12])
			ax.yaxis.set_ticks([10, 11, 12, 13])

			loc = 2
			if ii == 3 : loc = 4
			ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)

			if ii == 2 : plt.text(8.6, 9.9, '$z = $' + zed, size = 40, color = 'k', 
				verticalalignment='bottom', horizontalalignment='left')

			ii = ii + 1

		outputFile = plot_dir + 'Mstars_vs_Mdiff_ap_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_radial_tag_FA(self, snapnum, nRvir, type, Mstars, Mdiffuse, rhorhobar_FA) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , nRvir - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logMstars = np.log10(Mstars[w])
		logMdiff = np.log10(Mdiff[w]*f_b)
		logrho = np.log10(rho[w])
		print 'plotting Mstars vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'



		points = np.linspace(8, 15, 100)
		fit = np.polyfit(logMstars, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1


		label = mu + '(<' + str(nRvir) + rv + ')' + r', $z =$ ' + zed
		rhomax = np.max(logrho)
		rhomin = np.min(logrho)
		x_label = log10 + '(' + ms + '/' + msun + ')'
		y_label = log10 + '(' + fb + r'$\times$ ' + mu + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([8.5,13.1,8.5,13.1])
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		ax.scatter(logMstars, logMdiff, cmap = cm.jet, c = logrho, edgecolor = 'None', 
			vmin = -1., vmax = 2., alpha = 0.3, label = label)
		pts = ax.scatter(0, 0, cmap = cm.jet, c = 0, edgecolor = 'None', 
			vmin = -1., vmax = 2.)
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		
		cbar = plt.colorbar(pts)
		cbar.set_label(log10 + '(' + rhob + ') [' + str(r_FA) + mpc + ']' )

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		if nRvir < 3 : 
			loc = 2
			yloc = 4.1
		else: 
			loc = 4
			yloc = 0
		leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		leg.get_frame().set_alpha(0.5)


		outputFile = plot_dir + 'FA_tag_Mstars_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_aperture_tag_FA(self, snapnum, r_FA, type, Mstars, Mdiffuse, rhorhobar_FA) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logMstars = np.log10(Mstars[w])
		logMdiff = np.log10(Mdiff[w]*f_b)
		logrho = np.log10(rho[w])
		print 'plotting Mstars vs Mdiffuse within', r_FA, 'Mpc for', len(w), 'galaxies'

		points = np.linspace(0, 10000)*3.
		fit = np.polyfit(logMstars, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1



		label = mu + '(<' + str(r_FA) + mpc + ')' + r', $z =$ ' + zed
		rhomax = np.max(logrho)
		rhomin = np.min(logrho)

		x_label = log10 + '(' + ms + '/' + msun + ')'
		y_label = log10 + '(' + fb + r'$\times$ ' + mu + '/' + msun + ')'

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([8.5,13.1,10.0,14.6])
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)



		ax.scatter(logMstars, logMdiff, cmap = cm.jet, c = logrho, edgecolor = 'None', 
			vmin = -1., vmax = 2., alpha = 0.3, label = label)
		pts = ax.scatter(0, 0, cmap = cm.jet, c = 0, edgecolor = 'None', 
			vmin = -1., vmax = 2.)
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		
		cbar = plt.colorbar(pts)
		cbar.set_label(log10 + '(' + rhob + ') [' + str(r_FA) + mpc + ']' )

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		if ((r_FA > 3.) & (snapnum < 40)) : loc = 4
		else : loc = 2
		leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		leg.get_frame().set_alpha(0.5)


		outputFile = plot_dir + 'FA_tag_Mstars_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_radial_tag_ZZ(self, snapnum, nRvir, type, Mstars, Mdiffuse, GasMetals) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , nRvir - 1]
		ZZ = GasMetals/0.02
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logMstars = np.log10(Mstars[w])
		logMdiff = np.log10(Mdiff[w]*f_b)
		logZZ = np.log10(ZZ[w])
		print 'plotting Mstars vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'


		points = np.linspace(0, 10000)*3.
		fit = np.polyfit(logMstars, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = log10 + mu + ' = ' + fit_0 + ' x ' + log10 + ms + ' + ' + fit_1 

		label = mu + '(<' + str(nRvir) + rv + ')' + r', $z =$ ' + zed
		x_label = log10 + '(' + ms + '/' + msun + ')'
		y_label = log10 + '(' + fb + r'$\times$ ' + mu + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([8.5,12.1,8.5,13.1])


	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		cmap = cm.hot
		ax.scatter(logMstars, logMdiff, cmap = cmap, c = logZZ, edgecolor = 'None', 
			vmin = -2., vmax = 0.5, alpha = 0.3)
		pts = ax.scatter(0, 0, cmap = cmap, c = 0, edgecolor = 'None', 
			vmin = -2., vmax = 0.5)
		ax.plot(0, 0, c = cmap(0.5), label = label, mec = 'None', marker = 'o', ls = 'None')

		ww = np.where(logZZ < -1.)[0]
		ax.scatter(logMstars[ww], logMdiff[ww], cmap = cmap, c = logZZ[ww], edgecolor = 'None', 
			vmin = -2., vmax = 0.5, alpha = 1, marker = '*', s = 100)
		ax.plot(0, 0, c = cmap(0.2), label = 'Low Metallicity Galaxies', marker = '*', 
			markersize = 10, ls = 'None', mec = 'None')
		
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		
		cbar = plt.colorbar(pts, ticks = [-2, -1.5, -1, -0.5, 0, 0.5])
		cbar.set_label(log10 + '(' + zz + ')')

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		if nRvir < 3 : 
			loc = 2
			yloc = 4.1
		else: 
			loc = 4
			yloc = 0
		leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		leg.get_frame().set_alpha(0.5)


		outputFile = plot_dir + 'ZZ_tag_Mstars_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_aperture_tag_ZZ(self, snapnum, r_FA, type, Mstars, Mdiffuse, GasMetals) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
		ZZ = GasMetals/0.02
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logMstars = np.log10(Mstars[w])
		logMdiff = np.log10(Mdiff[w]*f_b)
		logZZ = np.log10(ZZ[w])
		print 'plotting Mstars vs Mdiffuse within', r_FA, 'Mpc for', len(w), 'galaxies'

		points = np.linspace(0, 10000)*3.
		fit = np.polyfit(logMstars, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = log10 + mu + ' = ' + fit_0 + ' x ' + log10 + vd + ' + ' + fit_1 


		label = mu + '(<' + str(r_FA) + mpc + ')' + r', $z =$ ' + zed
		x_label = log10 + '(' + ms + '/' + msun + ')'
		y_label = log10 + '(' + fb + r'$\times$ ' + mu + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([8.5,13.1,10.0,14.6])
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)



		cmap = cm.hot
		ax.scatter(logMstars, logMdiff, cmap = cmap, c = logZZ, edgecolor = 'None', 
			vmin = -2., vmax = 0.5, alpha = 0.3)
		pts = ax.scatter(0, 0, cmap = cmap, c = 0, edgecolor = 'None', 
			vmin = -2., vmax = 0.5)
		ax.plot(0, 0, c = cmap(0.5), label = label, mec = 'None', marker = 'o', ls = 'None')
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)

		ww = np.where(logZZ < -1.)[0]
		ax.scatter(logMstars[ww], logMdiff[ww], cmap = cmap, c = logZZ[ww], edgecolor = 'None', 
			vmin = -2., vmax = 0.5, alpha = 1, marker = '*', s = 100)
		ax.plot(0, 0, c = cmap(0.2), label = 'Low Metallicity Galaxies', marker = '*', 
			markersize = 10, ls = 'None', mec = 'None')
		
		cbar = plt.colorbar(pts, ticks = [-2, -1.5, -1, -0.5, 0, 0.5])
		cbar.set_label(log10 + '(' + zz + ')')

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		if ((r_FA > 3.) & (snapnum < 40)) : loc = 4
		else : loc = 2
		leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		leg.get_frame().set_alpha(0.5)


		outputFile = plot_dir + 'ZZ_tag_Mstars_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_freefall_radial(self, snapnum, type, Mvir, Mstars, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		tff0 = np.pi/(20.*np.sqrt(2))*1.

		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)

		ii = 1
		for nRvir in ns:

			Mdiff = Mdiffuse[: , nRvir - 1]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logMstars = np.log10(Mstars[w])
			logMdiff = np.log10(Mdiff[w])
			print 'plotting Mvir vs Mdiffuse freefall within', nRvir, 'Rvir for', len(w), 'galaxies'

			label = mu + '(<' + str(nRvir) + rv + ')'
			x_label = log10 + '(' + ms + '/' + msun + ')'
			y_label = r'$t_{ff}^{\mathrm{unbound}}/t_{\mathrm{H}(z)}$'




			ax = fig.add_subplot(1,len(ns),ii)
			plt.axis([8.5,13.1,0.1,1.5])
	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)

			col = cm.spectral((nRvir-1)/7.)

			tff = np.sqrt(nRvir**3/(1. + Mdiff[w]/Mvir[w]))*tff0
			wt = np.where(tff < 1.)[0]
			print float(len(wt))/float(len(logMdiff))*100., '% of galaxies can accrete all of the matter out to', nRvir, 'Rvir' 


			ax.scatter(logMstars, tff, c = col, edgecolor = col, alpha = 0.3, label = label)


			ax.set_xlabel(x_label, fontsize=34) 
			if (ii == 1) : ax.set_ylabel(y_label, fontsize=34) 

			if ii == 1 : ax.plot(0., 0., marker = 'None', ls = 'None', label = r'$z = $' + zed)

			if ii == 3 : H_label = 'Hubble Time'
			else : H_label = ''
			ax.axhline(y = 1., c = 'k', lw = 4, ls = ':', label = H_label)

			loc = 2
			if ii == 3 : loc = 4
			ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
			ii = ii + 1



		outputFile = plot_dir + 'freefall_Mstars_vs_Mdiff_ra_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_freefall_aperture(self, snapnum, type, Mvir, Mstars, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Hz = np.sqrt(0.7 + 0.3*(1 + z[snapnum])**3)
		tff0 = np.pi/(20.*np.sqrt(2))*1.



		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)

		ii = 1
		for r_FA in r_FAs:

			Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logMstars = np.log10(Mstars[w])
			logMdiff = np.log10(Mdiff[w])
			print 'plotting Mvir vs Mdiffuse freefall within', r_FA, 'Mpc for', len(w), 'galaxies'

			R_vir = (4.3e-15*Mvir[w])**(1./3.)
			
			if space == 'comoving' : R_vir = R_vir*(1. + z[snapnum])
			nRvir = r_FA/R_vir
			tff = np.sqrt(nRvir**3/(1. + Mdiff[w]/Mvir[w]))*tff0
			wt = np.where(tff < 1.)[0]
			print float(len(wt))/float(len(logMdiff))*100., 'percent of galaxies can accrete all of the matter out to', r_FA, 'Mpc' 


			label = mu + '(<' + str(r_FA) + mpc + ')'
			col = cm.gnuplot((r_FA+2)/8.)
			x_label = log10 + '(' + ms + '/' + msun + ')'
			y_label = r'$t_{ff}^{\mathrm{unbound}}/t_{\mathrm{H}(z)}$'

			ax = fig.add_subplot(1,len(ns),ii)
			plt.axis([8.5,13.1,0.1,5.5])
	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)



			ax.scatter(logMstars, tff, c = col, edgecolor = col, alpha = 0.3, label = label)


			ax.set_xlabel(x_label, fontsize=34) 
			if (ii == 1) : ax.set_ylabel(y_label, fontsize=34) 

			if ii == 1 : ax.plot(0., 0., marker = 'None', ls = 'None', label = r'$z = $' + zed)

			if ii == 1 : H_label = 'Hubble Time'
			else : H_label = ''
			ax.axhline(y = 1., c = 'k', lw = 4, ls = ':', label = H_label)

			loc = 2
			if ((snapnum > 60) & (ii > 1)): 
 				loc = 3
			leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
			leg.get_frame().set_alpha(0.5)
			ii = ii + 1



		outputFile = plot_dir + 'freefall_Mstars_vs_Mdiff_ap_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_radial_tag_fg(self, snapnum, nRvir, type, Mstars, Mdiffuse, ColdGas) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , nRvir - 1]
		fgas = ColdGas/(Mstars + ColdGas)
		w = np.where((type == 0) & (Mdiff > 0) & (ColdGas > 0.))[0]
		logMstars = np.log10(Mstars[w])
		logMdiff = np.log10(Mdiff[w]*f_b)
		logfgas = np.log10(fgas[w])
		print 'plotting Mstars vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'

		print 'log(f_gas) = ', np.min(logfgas), '-', np.max(logfgas)

		points = np.linspace(1, 15, 100)
		fit = np.polyfit(logMstars, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = log10 + mu + ' = ' + fit_0 + ' x ' + log10 + ms + ' + ' + fit_1 

		label = '(<' + str(nRvir) + rv + ')' + r', $z =$ ' + zed
		x_label = log10 + '(' + ms + '/' + msun + ')'
		y_label = log10 + '(' + fb + r'$\times$ ' + mu + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([8.5,13.1,8.5,13.1])



	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		cmap = cm.cool
		ax.scatter(logMstars, logMdiff, cmap = cmap, c = logfgas, edgecolor = 'None', 
			vmin = -4., vmax = 0.0, alpha = 0.3)
		pts = ax.scatter(0, 0, cmap = cmap, c = 0, edgecolor = 'None', 
			vmin = -4., vmax = 0.0)
		ax.plot(0, 0, c = cmap(1.), label = label, mec = 'None', marker = 'o', ls = 'None')

		ww = np.where(logfgas < -2.)[0]
		ax.scatter(logMstars[ww], logMdiff[ww], cmap = cmap, c = logfgas[ww], edgecolor = 'None', 
			vmin = -4., vmax = 0.0, alpha = 1, marker = '*', s = 100)
		ax.plot(0, 0, c = cmap(0.2), label = r'$f_{\mathrm{gas}} < 0.01$', marker = '*', 
			markersize = 10, ls = 'None', mec = 'None')
		
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = 'fit to data') #fit_label)
		
		cbar = plt.colorbar(pts, ticks = [-4, -3, -2, -1, 0,])
		cbar.set_label(log10 + '(' + r'$f_{\mathrm{gas}}$' + ')')

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 

		if nRvir < 3 : 
			loc = 2
			yloc = 4.1
		else: 
			loc = 4
			yloc = 0
		leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		leg.get_frame().set_alpha(0.3)

		outputFile = plot_dir + 'fgas_tag_Mstars_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_vs_Mdiffuse_aperture_tag_fg(self, snapnum, r_FA, type, Mstars, Mdiffuse, ColdGas) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
		fgas = ColdGas/(Mstars + ColdGas)
		w = np.where((type == 0) & (Mdiff > 0) & (ColdGas > 0.))[0]
		logMstars = np.log10(Mstars[w])
		logMdiff = np.log10(Mdiff[w]*f_b)
		logfgas = np.log10(fgas[w])
		print 'plotting Mstars vs Mdiffuse within', r_FA, 'Mpc for', len(w), 'galaxies'


		points = np.linspace(0, 10000)*3.
		fit = np.polyfit(logMstars, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit
		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = log10 + mu + ' = ' + fit_0 + ' x ' + log10 + vd + ' + ' + fit_1 


		label = mu + '(<' + str(r_FA) + mpc + ')' + r', $z =$ ' + zed
		x_label = log10 + '(' + ms + '/' + msun + ')'
		y_label = log10 + '(' + fb + r'$\times$ ' + mu + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([8.5,13.1,10.0,14.6])
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)



		cmap = cm.cool
		ax.scatter(logMstars, logMdiff, cmap = cmap, c = logfgas, edgecolor = 'None', 
			vmin = -4., vmax = 0.0, alpha = 0.3)
		pts = ax.scatter(0, 0, cmap = cmap, c = 0, edgecolor = 'None', 
			vmin = -4., vmax = 0.0)
		ax.plot(0, 0, c = cmap(1.), label = label, mec = 'None', marker = 'o', ls = 'None')

		ww = np.where(logfgas < -2.)[0]
		ax.scatter(logMstars[ww], logMdiff[ww], cmap = cmap, c = logfgas[ww], edgecolor = 'None', 
			vmin = -4., vmax = 0.0, alpha = 1, marker = '*', s = 100)
		ax.plot(0, 0, c = cmap(0.2), label = r'$f_{\mathrm{gas}} < 0.01$', marker = '*', 
			markersize = 10, ls = 'None', mec = 'None')
		
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = 'fit to data') #fit_label)
		
		cbar = plt.colorbar(pts, ticks = [-4, -3, -2, -1, 0,])
		cbar.set_label(log10 + '(' + r'$f_{\mathrm{gas}}$' + ')')

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 

		if ((r_FA > 3.) & (snapnum < 40)) : loc = 4
		else : loc = 2
		leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		leg.get_frame().set_alpha(0.5)

		outputFile = plot_dir + 'fgas_tag_Mstars_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Mstars_Mcold_vs_Mdiffuse_radial_all(self, snapnum, type, Mstars, ColdGas, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		

		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)

		ii = 1
		for nRvir in ns:

			Mdiff = Mdiffuse[: , nRvir - 1]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logMstars = np.log10(Mstars[w] + ColdGas[w])
			logMdiff = np.log10(Mdiff[w]*f_b)
			print 'plotting Mvir vs Mdiffuse freefall within', nRvir, 'Rvir for', len(w), 'galaxies'


			points = np.linspace(8, 15, 100)
			fit = np.polyfit(logMstars, logMdiff, 1)
			fit_points = points * fit[0] + fit[1]
			p1 = np.poly1d(np.polyfit(logMstars, logMdiff,  1))
			X1 = chisquare(logMdiff, f_exp=p1(logMstars), ddof=0, axis=0)
			print 'X1 =', X1

			fit_0 = "%1.2f" % fit[0]
			fit_1 = "%1.2f" % fit[1]
			fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1

			label = '(<' + str(nRvir) + rv + ')'
			col = cm.spectral((nRvir-1)/7.)
			x_label = (log10 + '((' + ms + '+' + cg + ')/' + msun + ')'
			y_label = log10 + '(' + r'$f_b \times$ ' + mu + '/' + msun + ')'


			ax = fig.add_subplot(1,len(ns),ii)
			plt.axis([8.5,12.1,8.5,13.2])
	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)


			ax.scatter(logMstars, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
			ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)

			ax.set_xlabelx_label, fontsize=28) 
			if ii == 1 : ax.set_ylabel(y_label, fontsize=34) 
			ax.xaxis.set_ticks([9, 10, 11, 12])
			ax.yaxis.set_ticks([9, 10, 11, 12, 13])

			loc = 2
			if ii == 3 : loc = 4
			ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)

			if ii == 3 : plt.text(8.7, 14.5, '$z = $' + zed, size = 40, color = 'k', 
				verticalalignment='top', horizontalalignment='left')

			ii = ii + 1

		outputFile = plot_dir + 'Mstars_Mcold_vs_Mdiff_ra_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  
		print 'Saved file to', outputFile
		plt.close()	

	
#____________rho_FA____________#(2)


	def rho_vs_Mdiffuse_radial(self, snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , nRvir - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logrho = np.log10(rho[w])
		logMdiff = np.log10(Mdiff[w])
		print 'plotting Mvir vs Mdiffuse within', nRvir, 'Rvir and with a', r_FA, 'Mp/h aperture'

		label = mu + '(<' + str(nRvir) + rv + ')'
		col = cm.spectral((nRvir-1)/7.)
		x_label = log10 + '(' + rhob + ')' + '[' + str(r_FA) + mpc + ']'
		y_label = log10 + '(' + mu + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([-2.1,2.1,9.5,14.1])
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		ax.scatter(logrho, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
		

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=2)
		plt.text(2., 14., '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='right')

		outputFile = plot_dir + 'rho_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  
		print 'Saved file to', outputFile
		plt.close()	

	def rho_vs_Mdiffuse_aperture(self, snapnum, r_FA, type, rhorhobar_FA, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logrho = np.log10(rho[w])
		logMdiff = np.log10(Mdiff[w])
		print 'plotting Mvir vs Mdiffuse within', r_FA, 'Mpc/h for', len(w), 'galaxies'

		label = mu + '(<' + str(r_FA) + mpc + ')'
		col = cm.gnuplot((r_FA+2)/8.)
		x_label = log10 + '(' + rhob + ')' + '[' + str(r_FA) + mpc + ']'
		y_label = log10 + '(' + mu + '/' + msun + ')'

		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([-1.6,2.6,10.5,15.1])

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		ax.scatter(logrho, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
	

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=3)
		plt.text(2.5, 15., '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='right')

		outputFile = plot_dir + 'rho_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  
		print 'Saved file to', outputFile
		plt.close()	


#____________Vdisp____________#(9)

	def Vdisp_vs_Mdiffuse_radial(self, snapnum, nRvir, type, Vdisp, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , nRvir - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logVdisp = np.log10(Vdisp[w])
		logMdiff = np.log10(Mdiff[w])
		print 'plotting Vdisp vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'

		v_points = np.logspace(1, 3, 100)
		Hz = np.sqrt(0.7 + 0.3*(1 + z[snapnum])**3)
		print Hz

		ub_pcls = float(N_unbound[snapnum])
		rho_mass = ub_pcls*pcl_mass/(box_side/(1. + z[snapnum]))**3
		radius = v_points/(10. * 100.*Hz)

		rad_volume = 4./3.*np.pi*(nRvir*radius)**3
		rad_expected_mass = rho_mass*rad_volume


		points = np.linspace(0, 10000)*3.
		fit = np.polyfit(logVdisp, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1


		label = mu + '(<' + str(nRvir) + rv + ')'
		col = cm.spectral((nRvir-1)/7.)
		x_label = log10 + '(' + vd + '/' + kms + ')'
		y_label = log10 + '(' + mu + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([1.2,3.0,9.,14.1])
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		ax.scatter(logVdisp, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		ax.plot(np.log10(v_points), np.log10(rad_expected_mass), c = 'k', lw = 4, ls = ':', label = 'Expected Mass')
		

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=2)
		plt.text(2.9, 9.1, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='right')

		outputFile = plot_dir + 'Vdisp_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + snap + OutputFormat
		plt.savefig(outputFile) 
		print 'Saved file to', outputFile
		plt.close()	

	def Vdisp_vs_Mdiffuse_aperture(self, snapnum, r_FA, type, Vdisp, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		rindex = int(r_FA*2) - 1
		Mdiff = Mdiffuse[: , rindex]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logVdisp = np.log10(Vdisp[w])
		logMdiff = np.log10(Mdiff[w])
		print 'plotting Vdisp vs Mdiffuse within', r_FA, 'Mpc for', len(w), 'galaxies'


		ub_pcls = float(N_unbound[snapnum])
		rho_mass = ub_pcls*pcl_mass/box_side**3
		radius = r_FA
		if space == 'physical' : radius = radius*(1. + z[snapnum])
		ap_volume = 4./3.*np.pi*radius**3
		ap_expected_mass = rho_mass*ap_volume
		
		
		points = np.linspace(8, 15, 100)
		Hz = np.sqrt(0.7 + 0.3*(1 + z[snapnum])**3)
		print Hz
		Rpoints = (10**points / (2.32443891e14 * Hz**2))**(1./3.)


		points = np.linspace(0, 10000)*3.
		fit = np.polyfit(logVdisp, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit
		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = log10 + mu + ' = ' + fit_0 + ' x ' + log10 + vd + ' + ' + fit_1 

		label = mu + '(<' + str(r_FA) + mpc + ')'
		col = cm.gnuplot((r_FA+2)/8.)
		x_label = log10 + '(' + vd + '/' + kms + ')'
		y_label = log10 + '(' + mu + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([1.2,3.0,10.5,16.1])

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		ax.scatter(logVdisp, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		ax.axhline(y = np.log10(ap_expected_mass), c = 'k', lw = 4, ls = ':', label = 'Expected Mass')
		

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 

		if ((r_FA > 3.) & (snapnum < 40)) : loc = 4
		else : loc = 2
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)

		plt.text(2.9, 16., '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='right')

		outputFile = plot_dir + 'Vdisp_vs_Mdiff_ap_' + str(int(r_FA)) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Vdisp_vs_Mdiffuse_radial_all(self, snapnum, type, Vdisp, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)

		ii = 1
		for nRvir in ns:
			Mdiff = Mdiffuse[: , nRvir - 1]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logVdisp = np.log10(Vdisp[w])
			logMdiff = np.log10(Mdiff[w])
			print 'plotting Vdisp vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'

			v_points = np.logspace(1, 3, 100)
			Hz = np.sqrt(0.7 + 0.3*(1 + z[snapnum])**3)
			print Hz

			ub_pcls = float(N_unbound[snapnum])
			rho_mass = ub_pcls*pcl_mass/(box_side/(1. + z[snapnum]))**3
			radius = v_points/(10. * 100.*Hz)

			rad_volume = 4./3.*np.pi*(nRvir*radius)**3
			rad_expected_mass = rho_mass*rad_volume


			points = np.linspace(0, 3)
			fit = np.polyfit(logVdisp, logMdiff, 1)
			fit_points = points * fit[0] + fit[1]
			print fit

			fit_0 = "%1.2f" % fit[0]
			fit_1 = "%1.2f" % fit[1]
			fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1

			label = '(<' + str(nRvir) + rv + ')'
			col = cm.spectral((nRvir-1)/7.)
			x_label = log10 + '(' + vd + '/' + kms + ')'
			y_label = log10 + '(' + mu + '/' + msun + ')'



			ax = fig.add_subplot(1,len(ns),ii)
			plt.axis([1.2,3.0,9.1,14.1])

	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)



			ax.scatter(logVdisp, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
			ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
			ax.plot(np.log10(v_points), np.log10(rad_expected_mass), c = 'k', lw = 4, ls = ':', label = 'Expected Mass')
		

			ax.set_xlabel(x_label, fontsize=34) 
			if ii == 1 : ax.set_ylabel(y_label, fontsize=34) 

			loc = 2
			if ii == 3 : loc = 4
			leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
			leg.get_frame().set_alpha(0.5)

			if ii == 1 : 
				plt.text(2.9, 9.1, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='right')
			
			ii = ii + 1

		outputFile = plot_dir + 'Vdisp_vs_Mdiff_ra_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	
		
	def Vdisp_vs_Mdiffuse_aperture_all(self, snapnum, type, Vdisp, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		fig = plt.figure(figsize=(18.,8))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .15, left = .08, right = .98)

		ii = 1
		for r_FA in r_FAs:
			rindex = int(r_FA*2) - 1
			Mdiff = Mdiffuse[: , rindex]
			w = np.where((type == 0) & (Mdiff > 0))[0]
			logVdisp = np.log10(Vdisp[w])
			logMdiff = np.log10(Mdiff[w])
			print 'plotting Vdisp vs Mdiffuse within', r_FA, 'Mpc for', len(w), 'galaxies'

			ub_pcls = float(N_unbound[snapnum])
			rho_mass = ub_pcls*pcl_mass/box_side**3
			radius = r_FA
			if space == 'physical' : radius = radius*(1. + z[snapnum])
			ap_volume = 4./3.*np.pi*radius**3
			ap_expected_mass = rho_mass*ap_volume
		
		
			points = np.linspace(0, 3, 100)
			fit = np.polyfit(logVdisp, logMdiff, 1)
			fit_points = points * fit[0] + fit[1]
			print fit
			fit_0 = "%1.2f" % fit[0]
			fit_1 = "%1.2f" % fit[1]
			fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1

			label = '(<' + str(r_FA) + mpc + ')'
			col = cm.gnuplot((r_FA+2)/8.)
			x_label = log10 + '(' + vd + '/' + kms + ')'
			y_label = log10 + '(' + mu + '/' + msun + ')'


			ax = fig.add_subplot(1,len(ns),ii)
			plt.axis([1.2,3.0,10.5,14.5])

	
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(30)
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(30)


			ax.scatter(logVdisp, logMdiff, c = col, edgecolor = col, alpha = 0.3, label = label)
			ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
			ax.axhline(y = np.log10(ap_expected_mass), c = 'k', lw = 4, ls = ':', label = 'Expected Mass')
		

			ax.set_xlabel(x_label, fontsize=34) 
			if ii == 1 : ax.set_ylabel(y_label, fontsize=34) 
#			ax.xaxis.set_ticks([9, 10, 11, 12])
			ax.yaxis.set_ticks([11, 12, 13, 14])

			loc = 2
			if ii != 1 : loc = 4
			leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
			leg.get_frame().set_alpha(0.5)
			if ii == 2 : 
				plt.text(2.1, 14.4, '$z = $' + zed, size = 40, color = 'k', 
					verticalalignment='top', horizontalalignment='center')


			
			ii = ii + 1

		outputFile = plot_dir + 'Vdisp_vs_Mdiff_ap_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	
		
	def Vdisp_vs_Mdiffuse_radial_tag_FA(self, snapnum, nRvir, type, Vdisp, Mdiffuse, rhorhobar_FA) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , nRvir - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logVdisp = np.log10(Vdisp[w])
		logMdiff = np.log10(Mdiff[w])
		logrho = np.log10(rho[w])
		print 'plotting Vdisp vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'


		v_points = np.logspace(1, 3, 100)
		Hz = np.sqrt(0.7 + 0.3*(1 + z[snapnum])**3)
		print Hz

		ub_pcls = float(N_unbound[snapnum])
		rho_mass = ub_pcls*pcl_mass/(box_side/(1. + z[snapnum]))**3
		radius = v_points/(10. * 100.*Hz)

		rad_volume = 4./3.*np.pi*(nRvir*radius)**3
		rad_expected_mass = rho_mass*rad_volume


		points = np.linspace(0, 10000)*3.
		fit = np.polyfit(logVdisp, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = log10 + mu + ' = ' + fit_0 + ' x ' + log10 + vd + ' + ' + fit_1 

		label = mu + '(<' + str(nRvir) + rv + ')' + r', $z =$ ' + zed
		x_label = log10 + '(' + vd + '/' + kms + ')'
		y_label = log10 + '(' + mu + '/' + msun + ')'

	
		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
		ax = fig.add_subplot(1,1,1)

		plt.axis([1.2,3.0,9.,14.1])

		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		cmap = cm.jet
		ax.scatter(logVdisp, logMdiff, cmap = cmap, c = logrho, edgecolor = 'None', 
			vmin = -1., vmax = 2., alpha = 0.3, label = label)
		pts = ax.scatter(0, 0, cmap = cmap, c = 0, edgecolor = 'None', 
			vmin = -1., vmax = 2.)
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		ax.plot(np.log10(v_points), np.log10(rad_expected_mass), c = 'k', lw = 4, ls = ':', label = 'Expected Mass')
		
		cbar = plt.colorbar(pts)
		cbar.set_label(log10 + '(' + rhob + ') [' + str(r_FA) + mpc + ']' )

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 
		leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=2)
		leg.get_frame().set_alpha(0.5)

		outputFile = plot_dir + 'FA_tag_Vdisp_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  
		print 'Saved file to', outputFile
		plt.close()	

	def Vdisp_vs_Mdiffuse_aperture_tag_FA(self, snapnum, r_FA, type, Vdisp, Mdiffuse, rhorhobar_FA) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logVdisp = np.log10(Vdisp[w])
		logMdiff = np.log10(Mdiff[w])
		logrho = np.log10(rho[w])
		print 'plotting Vdisp vs Mdiffuse within', r_FA, 'Mpc for', len(w), 'galaxies'


		v_points = np.logspace(1, 3, 100)
		Hz = np.sqrt(0.7 + 0.3*(1 + z[snapnum])**3)
		print Hz

		ub_pcls = float(N_unbound[snapnum])
		rho_mass = ub_pcls*pcl_mass/box_side**3
		radius = r_FA
		if space == 'physical' : radius = radius*(1. + z[snapnum])
		ap_volume = 4./3.*np.pi*radius**3
		ap_expected_mass = rho_mass*ap_volume


		points = np.linspace(0, 10000)*3.
		fit = np.polyfit(logVdisp, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = log10 + mu + ' = ' + fit_0 + ' x ' + log10 + vd + ' + ' + fit_1 

		label = mu + '(<' + str(r_FA) + mpc + ')' + r', $z =$ ' + zed
		x_label = log10 + '(' + vd + '/' + kms + ')'
		y_label = log10 + '(' + mu + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([1.2,3.0,10.5,16.1])

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		cmap = cm.jet
		ax.scatter(logVdisp, logMdiff, cmap = cmap, c = logrho, edgecolor = 'None', 
			vmin = -1., vmax = 2., alpha = 0.3, label = label)
		pts = ax.scatter(0, 0, cmap = cmap, c = 0, edgecolor = 'None', 
			vmin = -1., vmax = 2.)
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		ax.axhline(y = np.log10(ap_expected_mass), c = 'k', lw = 4, ls = ':', label = 'Expected Mass')
		
		cbar = plt.colorbar(pts)
		cbar.set_label(log10 + '(' + rhob + ') [' + str(r_FA) + mpc + ']' )

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 

		if ((r_FA > 3.) & (snapnum < 40)) : loc = 4
		else : loc = 2
		leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		leg.get_frame().set_alpha(0.5)


		outputFile = plot_dir + 'FA_tag_Vdisp_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Vdisp_vs_rho(self, snapnum, r_FA, type, Vdisp, rhorhobar_FA, Mdiffuse) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
		Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logVdisp = np.log10(Vdisp[w])
		logrho = np.log10(rho[w])
		logMdiff = np.log10(Mdiff[w])
		print 'plotting Mvir vs Mdiffuse within', r_FA, 'Mpc'

		label = mu + '(<' + str(r_FA) + mpc + ')' + r', $z =$ ' + zed
		col = cm.gnuplot((r_FA+2)/8.)


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([1.2, 3.0, -2.1,2.5])

	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)

		ax.scatter(logVdisp, logrho, c = col, edgecolor = col, alpha = 0.3, label = label)
		

		ax.set_xlabel(log10 + '(' + vd + '/' + kms + ')', fontsize=34) 
		ax.set_ylabel(log10 + '(' + rhob + ')' + '[' + str(r_FA) + mpc + ']', fontsize=34) 
		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=4)
		plt.text(2., 14., '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='right')

		outputFile = plot_dir + 'Vdisp_vs_rho_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  
		print 'Saved file to', outputFile
		plt.close()	

	def Vdisp_vs_Mdiffuse_radial_tag_ZZ(self, snapnum, nRvir, type, Vdisp, Mdiffuse, GasMetals) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , nRvir - 1]
		ZZ = GasMetals/0.02
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logVdisp = np.log10(Vdisp[w])
		logMdiff = np.log10(Mdiff[w])
		logZZ = np.log10(ZZ[w])
		print 'plotting Mstars vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'


		points = np.linspace(1, 4, 100)
		fit = np.polyfit(logVdisp, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1

		label = '(<' + str(nRvir) + rv + ')' + r', $z =$ ' + zed
		x_label = log10 + '(' + vd + '/' + kms + ')'
		y_label = log10 + '(' + fb + r'$\times$ ' + mu + '/' + msun + ')'



		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([1.2,3.0,9.,14.1])
	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		cmap = cm.hot
		ax.scatter(logVdisp, logMdiff, cmap = cmap, c = logZZ, edgecolor = 'None', 
			vmin = -2., vmax = 0.5, alpha = 0.3)
		pts = ax.scatter(0, 0, cmap = cmap, c = 0, edgecolor = 'None', 
			vmin = -2., vmax = 0.5)
		ax.plot(0, 0, c = cmap(0.5), label = label, mec = 'None', marker = 'o', ls = 'None')

		ww = np.where(logZZ < -1.)[0]
		ax.scatter(logVdisp[ww], logMdiff[ww], cmap = cmap, c = logZZ[ww], edgecolor = 'None', 
			vmin = -2., vmax = 0.5, alpha = 1, marker = '*', s = 100)
		ax.plot(0, 0, c = cmap(0.2), label = '[' + zz + '] < -1', marker = '*', 
			markersize = 10, ls = 'None', mec = 'None')
		
		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		
		cbar = plt.colorbar(pts, ticks = [-2, -1.5, -1, -0.5, 0, 0.5])
		cbar.set_label(log10 + '(' + zz + ')')

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 

		if nRvir < 3 : 
			loc = 2
			yloc = 4.1
		else: 
			loc = 4
			yloc = 0
		leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		leg.get_frame().set_alpha(0.5)

		outputFile = plot_dir + 'ZZ_tag_Vdisp_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  # Save the figure
		print 'Saved file to', outputFile
		plt.close()	

	def Vdisp_vs_Mdiffuse_radial_tag_sSFR(self, snapnum, nRvir, type, Vdisp, Mdiffuse, SFR) :
		snap = "%03d" % snapnum
		zed = "%1.2f" % z[snapnum]
		
		Mdiff = Mdiffuse[: , nRvir - 1]
		w = np.where((type == 0) & (Mdiff > 0))[0]
		logVdisp = np.log10(Vdisp[w])
		logMdiff = np.log10(Mdiff[w])
		logSFR = np.log10(SFR[w]) + 10.
		print 'plotting Mstars vs Mdiffuse within', nRvir, 'Rvir for', len(w), 'galaxies'
		
		ww = np.where(SFR[w] > 0.)[0]
		print 'min, max sSFR = ', np.min(logSFR[ww]), np.max(logSFR)

		points = np.linspace(1, 4, 100)
		fit = np.polyfit(logVdisp, logMdiff, 1)
		fit_points = points * fit[0] + fit[1]
		print fit

		fit_0 = "%1.2f" % fit[0]
		fit_1 = "%1.2f" % fit[1]
		fit_label = r'$\alpha =$ ' + fit_0 + r', $\beta = $ ' + fit_1

		label = '(<' + str(nRvir) + rv + ')' + r', $z =$ ' + zed
		x_label = log10 + '(' + vd + '/' + kms + ')'
		y_label = log10 + '(' + fb + r'$\times$ ' + mu + '/' + msun + ')'


		fig = plt.figure(figsize=(12.,10))	
		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)

		ax = fig.add_subplot(1,1,1)
		plt.axis([1.2,3.0,9.,14.1])


	
		for tick in ax.xaxis.get_major_ticks():
			tick.label1.set_fontsize(30)
		for tick in ax.yaxis.get_major_ticks():
			tick.label1.set_fontsize(30)


		cmap = rvb
		ax.scatter(logVdisp, logMdiff, cmap = cmap, c = logSFR, edgecolor = 'None', 
			vmin = -1., vmax = 1., alpha = 0.3)
		pts = ax.scatter(0, 0, cmap = cmap, c = 0, edgecolor = 'None', 
			vmin = -1., vmax = 1.)
		ax.plot(0, 0, c = cmap(0.5), label = label, mec = 'None', marker = 'o', ls = 'None')

		ax.plot(points, fit_points, c = 'k', lw = 4, ls = '-', label = fit_label)
		
		cbar = plt.colorbar(pts, ticks = [-1, -0.5, 0, 0.5, 1])
		cbar.set_label(log10 + '(s' + sfr + '/' + hpy + ')')

		ax.set_xlabel(x_label, fontsize=34) 
		ax.set_ylabel(y_label, fontsize=34) 

		if nRvir < 3 : 
			loc = 2
			yloc = 4.1
		else: 
			loc = 4
			yloc = 0
		leg = ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=loc)
		leg.get_frame().set_alpha(0.5)

		outputFile = plot_dir + 'sSFR_tag_Vdisp_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + space + '_' + snap + OutputFormat
		plt.savefig(outputFile)  
		print 'Saved file to', outputFile
		plt.close()	




		
# 	def Find_Galaxies_to_Evolve(self, snaps) :
# 		Type = []
# 		Mvir = []
# 		IDs = []
# 		for zz in snaps :
# 			type, mvir, _ , _, ids = res.read_gals(zz)
# 			print 'max Mvir', np.max(mvir)
# 			ww = np.where(type == 0)[0]
# 			if (zz == 63) : ww = np.where((type == 0) & (1.0e11 < mvir) & (mvir < 1.5e11))[0]
# 			Type.append(type[ww])
# 			Mvir.append(mvir[ww])
# 			IDs.append(ids[ww])
# 			print len(ww), 'galaxies in snap', zz
# 			
# 		IDs01 = set(IDs[0]).intersection(IDs[1])
# 		print len(IDs01), 'galaxies in snapshots 0 and 1'
# 		IDs23 = set(IDs[2]).intersection(IDs[3])
# 		print len(IDs23), 'galaxies in snapshots 2 and 3'
# 
# 		IDs_all = set(IDs01).intersection(IDs23)
# 		
# 		print len(IDs_all), 'galaxies in all 4 snapshots in mass range'
# #		print IDs_all
# 		
# 		f = open(halo_dir + 'IDs_evolution_11.dat','w')
# 		for id in IDs_all : 
# 			f.write(str(id) + '\n')
# 		f.close() 
# 		
# 	def Galaxy_Window(self, galnum, snapnum, mass_range, nRvir, Rvir, xpos, ypos, zpos, bound) :
# 		snap = "%03d" % snapnum
# 		zed = "%1.2f" % z[snapnum]
# 		gal = "%03d" % galnum
# 		
# 		print np.sum(bound), '= bound - unbound'
# 		
# 		ww = np.where((-1. < zpos) & (zpos < 1.))[0]
# 		print len(ww), 'particles shown out of', len(xpos)
# 		xpos = xpos[ww]
# 		ypos = ypos[ww]
# 		zpos = zpos[ww]
# 		bound = bound[ww]
# 		
# 
# # 		xpos = np.where((-box_side/2. < xpos) & (xpos < box_side/2.), xpos, xpos - box_side/2.)
# # 		ypos = np.where((-box_side/2. < ypos) & (ypos < box_side/2.), ypos, ypos - box_side/2.)
# # 		zpos = np.where((-box_side/2. < zpos) & (zpos < box_side/2.), zpos, zpos - box_side/2.)
# 
# 		fig = plt.figure(figsize=(12.,10))	
# 		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .13, right = .94)
# 		ax = fig.add_subplot(1,1,1, aspect = 'equal')
# 
# 		plt.axis([-8.*Rvir,8.*Rvir,-8.*Rvir,8.*Rvir])
# #		ax.set_xticks([11,12,13,14])
# 
# 		for tick in ax.xaxis.get_major_ticks():
# 			tick.label1.set_fontsize(30)
# 		for tick in ax.yaxis.get_major_ticks():
# 			tick.label1.set_fontsize(30)
# 
# 		ax.scatter(0., 0., c = cm.brg(2.5/2.7), 
# 			label = r'$\mathrm{Bound}\ \mathrm{Particles}$', edgecolor = 'none')
# 		ax.scatter(0., 0., c = cm.brg(0.5/2.7),
# 			label = r'$\mathrm{Unbound}\ \mathrm{Particles}$', edgecolor = 'none')
# 		pts = ax.scatter(xpos, ypos, cmap = cm.brg, c = bound, vmin = -1.5, vmax = 1.2, 
# 			edgecolor = 'none', alpha = 0.7, s = 10)
# 
# 		Rvir1=plt.Circle((0,0),Rvir,color='k',fill=False, lw = 3)
# #			, label = r'$1\ \mathrm{R}_{\mathrm{vir}}$')
# 		Rvir6=plt.Circle((0,0),(nRvir + 1)*Rvir,color='k',fill=False, lw = 5, ls = 'dashed')
# #			, label = r'$6\ \mathrm{R}_{\mathrm{vir}}$')
# 		ax.add_patch(Rvir1)
# 		ax.add_patch(Rvir6)
# 
# #		cax = divider.append_axes("right", size="5%", pad=0.1)
# #		cbar = plt.colorbar(pts)
# #		cbar.set_label(r'$\mathrm{log}_{10}(\rho/\bar{\rho})$', fontsize=34) #54 for square
# 
# 
# 		
# 		text1 = r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}(z=0)/h^{-1} \mathrm{M}_{\odot}) = $' + str(mass_range)
# 		text2 = r'$z = $' + zed
# 		ax.set_xlabel(r'$\mathrm{X}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) 
# 		ax.set_ylabel(r'$\mathrm{Y}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) #54 for square
# 		ax.legend(prop = matplotlib.font_manager.FontProperties(size=22),fancybox=True,loc=0)
# 		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$11$' + '\n' + text2, size = 34, color = 'k', 
# 			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
# #		plt.text(12.5, -0.4, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')
# 
# 		outputFile = plot_dir + 'Galaxy_Window_' + str(mass_range) + '_' + gal + '_' + snap + OutputFormat
# 		plt.savefig(outputFile)  # Save the figure
# 		print 'Saved file to', outputFile
# 		plt.close()
# 
# 	def Galaxy_Window_contours(self, galnum, snapnum, mass_range, nRvir, Rvir, xpos, ypos, zpos, bound) :
# 		snap = "%03d" % snapnum
# 		zed = "%1.2f" % z[snapnum]
# 		gal = "%03d" % galnum
# 		
# 		print np.sum(bound), '= bound - unbound'
# 		
# 		ww = np.where((-1. < zpos) & (zpos < 1.))[0]
# 		print len(ww), 'particles shown out of', len(xpos)
# 		xpos = xpos[ww]
# 		ypos = ypos[ww]
# 		zpos = zpos[ww]
# 		bound = bound[ww]
# 		
# 		wb = np.where(bound > 0.)[0]
# 		xpos_b = xpos[wb]
# 		ypos_b = ypos[wb]
# 		zpos_b = zpos[wb]
# 
# 		wu = np.where(bound < 0.)[0]
# 		xpos_u = xpos[wu]
# 		ypos_u = ypos[wu]
# 		zpos_u = zpos[wu]
# 		
# 		axis_range = [-8.*Rvir,8.*Rvir,-8.*Rvir,8.*Rvir]
# 		
# 		n_hist_bins = 20
# 		hist_bins = np.linspace(-8.,8., n_hist_bins + 1)*Rvir
# 		H_a, xedges, yedges = np.histogram2d(ypos, xpos, bins = hist_bins)
# 		H_ab, xedges, yedges = np.histogram2d(ypos, xpos, bins = hist_bins, weights = bound)
# 		H_u, xedges, yedges = np.histogram2d(ypos_u, xpos_u, bins = hist_bins)
# 		H_b, xedges, yedges = np.histogram2d(ypos_b, xpos_b, bins = hist_bins)
# 		dumX = hist_bins[:n_hist_bins] + (hist_bins[1] - hist_bins[0])/2.
# 		X, Y = np.meshgrid(dumX, dumX)
# 		
# 		bin_side = (axis_range[1] - axis_range[0])/float(n_hist_bins)
# 		H_a = H_a * (1. / bin_side**3)/(270./box_side)**3
# 		H_ab = H_ab * (1. / bin_side**3)/(270./box_side)**3
# 		H_u = H_u * (1. / bin_side**3)/(270./box_side)**3
# 		H_b = H_b * (1. / bin_side**3)/(270./box_side)**3
# 		min_rho_per_bin = (1. / bin_side**3)/(270./box_side)**3
# 
# 		print np.size(X), np.size(Y), np.size(H_b)
# 		
# 		logH_u = np.log10(H_u)
# 		logH_u = np.where(H_u > 0, np.log10(H_u), np.log10(min_rho_per_bin))
# 		logH_b = np.log10(H_b)
# 		logH_bb = np.where(H_b > 0, np.log10(H_b), np.log10(min_rho_per_bin))
# 		logH_a = np.log10(H_a)
# 
# # 		xpos = np.where((-box_side/2. < xpos) & (xpos < box_side/2.), xpos, xpos - box_side/2.)
# # 		ypos = np.where((-box_side/2. < ypos) & (ypos < box_side/2.), ypos, ypos - box_side/2.)
# # 		zpos = np.where((-box_side/2. < zpos) & (zpos < box_side/2.), zpos, zpos - box_side/2.)
# 
# 		fig = plt.figure(figsize=(12.,10))	
# 		fig.subplots_adjust(wspace = .0,top = .9, bottom = .12, left = .13, right = .9)
# 		ax = fig.add_subplot(1,1,1, aspect = 'equal')
# 		ax.xaxis.set_label_position('top') 
# 		ax.xaxis.tick_top()
# 
# 		plt.axis(axis_range)
# #		ax.set_xticks([11,12,13,14])
# 
# 		for tick in ax.xaxis.get_major_ticks():
# 			tick.label1.set_fontsize(30)
# 		for tick in ax.yaxis.get_major_ticks():
# 			tick.label1.set_fontsize(30)
# 
# 		plt.axis(axis_range)
# 
# 
# #		cmap_b = colors.ListedColormap(['Greens'])
# 
# 		boundedness = (H_ab/H_a + 1.)/2.
# 		density_b = logH_bb
# 		norm = colors.Normalize(density_b.min(), density_b.max())
# 		bound_array = plt.get_cmap('Greens')(norm(density_b)*0.9)
# 		bound_array[..., 3] = boundedness*0.9  # <- some alpha values between 0.5-1
# #		print red_array
# 
# 		
# 		im2 = plt.imshow(logH_b, interpolation='sinc', cmap=cm.Greens,
#                 origin='lower', extent=axis_range,
#                 vmax=logH_b.max(), vmin=logH_u.min(), alpha = 0.7)
# 		im = plt.imshow(logH_u, interpolation='sinc', cmap=cm.Purples,
#                 origin='lower', extent=axis_range,
#                 vmax=logH_u.max(), vmin=logH_u.min())
# 		plt.imshow(bound_array, interpolation='spline36', 
#                 origin='lower', extent=axis_range)
# #		ax.scatter(xpos_u, ypos_u, s = 1, alpha = 0.1, edgecolor = 'none')
# 
# 		levels = [1., 1.5, 2.0, 2.5, 3.0, 3.5]
# 		if (np.max(logH_b) < 3.5) : levels = [1., 1.5, 2.0, 2.5, 3.0]
# 		
# 		cs = plt.contour(X, Y, logH_b, levels = levels, cmap = cm.YlGn)
# 
# 		Rvir1=plt.Circle((0,0),Rvir,color='k',fill=False, lw = 3)
# #			, label = r'$1\ \mathrm{R}_{\mathrm{vir}}$')
# 		Rvir6=plt.Circle((0,0),(nRvir + 1)*Rvir,color='k',fill=False, lw = 5, ls = 'dashed')
# #			, label = r'$6\ \mathrm{R}_{\mathrm{vir}}$')
# 		ax.add_patch(Rvir1)
# 		ax.add_patch(Rvir6)
# 
# 		text1 = r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}(z=0)/h^{-1} \mathrm{M}_{\odot}) = $' + str(mass_range)
# 		text2 = r'$z = $' + zed
# 		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$13$' + '\n' + text2, size = 34, color = 'k', 
# 			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
# 
# 
# 		divider = make_axes_locatable(ax)
# 		cax = divider.append_axes("right", size="5%", pad=0.1)
# 		cax2 = divider.append_axes("bottom", size="5%", pad=0.1)
# 		cbar = plt.colorbar(im, cax = cax, ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# 		cbar2 = plt.colorbar(im2, cax = cax2, orientation='horizontal', ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
# 		cbar.set_label(r'$\mathrm{log}_{10}(\rho_{\mathrm{Unbound}}/\bar{\rho})$', fontsize=34) #54 for square
# 		cbar2.set_label(r'$\mathrm{log}_{10}(\rho_{\mathrm{Bound}}/\bar{\rho})$', fontsize=34) #54 for square
# 
# 
# 		
# 		ax.set_xlabel(r'$\mathrm{X}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) 
# 		ax.set_ylabel(r'$\mathrm{Y}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) #54 for square
# # 		ax.legend(prop = matplotlib.font_manager.FontProperties(size=22),fancybox=True,loc=0)
# #		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$13$' + '\n' + text2, size = 34, color = 'k', 
# #			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
# #		plt.text(12.5, -0.4, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')
# 
# 		outputFile = plot_dir + 'Galaxy_Window_contours_' + str(mass_range) + '_' + gal + '_' + snap + OutputFormat
# 		plt.savefig(outputFile)  # Save the figure
# 		print 'Saved file to', outputFile
# 		plt.close()
# 
# 	def Galaxy_Window_contours2(self, galnum, snapnum, mass_range, nRvir, Rvir, xpos, ypos, zpos, bound) :
# 		snap = "%03d" % snapnum
# 		zed = "%1.2f" % z[snapnum]
# 		gal = "%03d" % galnum
# 		
# 		print np.sum(bound), '= bound - unbound'
# 		
# 		ww = np.where((-1. < zpos) & (zpos < 1.))[0]
# 		print len(ww), 'particles shown out of', len(xpos)
# 		xpos = xpos[ww]
# 		ypos = ypos[ww]
# 		zpos = zpos[ww]
# 		bound = bound[ww]
# 		
# 		wb = np.where(bound > 0.)[0]
# 		xpos_b = xpos[wb]
# 		ypos_b = ypos[wb]
# 		zpos_b = zpos[wb]
# 
# 		wu = np.where(bound < 0.)[0]
# 		xpos_u = xpos[wu]
# 		ypos_u = ypos[wu]
# 		zpos_u = zpos[wu]
# 		
# 		axis_range = [-8.*Rvir,8.*Rvir,-8.*Rvir,8.*Rvir]
# 		
# 		n_hist_bins = 30
# 		hist_bins = np.linspace(-8.,8., n_hist_bins + 1)*Rvir
# 		H_a, xedges, yedges = np.histogram2d(ypos, xpos, bins = hist_bins)
# 		H_ab, xedges, yedges = np.histogram2d(ypos, xpos, bins = hist_bins, weights = bound)
# 		H_u, xedges, yedges = np.histogram2d(ypos_u, xpos_u, bins = hist_bins)
# 		H_b, xedges, yedges = np.histogram2d(ypos_b, xpos_b, bins = hist_bins)
# 		dumX = hist_bins[:n_hist_bins] + (hist_bins[1] - hist_bins[0])/2.
# 		X, Y = np.meshgrid(dumX, dumX)
# 		
# 		bin_side = (axis_range[1] - axis_range[0])/float(n_hist_bins)
# 		H_a = H_a * (1. / bin_side**3)/(270./box_side)**3
# 		H_ab = H_ab * (1. / bin_side**3)/(270./box_side)**3
# 		H_u = H_u * (1. / bin_side**3)/(270./box_side)**3
# 		H_b = H_b * (1. / bin_side**3)/(270./box_side)**3
# 		min_rho_per_bin = (1. / bin_side**3)/(270./box_side)**3
# 
# 		print np.size(X), np.size(Y), np.size(H_b)
# 		
# 		logH_u = np.log10(H_u)
# 		logH_u = np.where(H_u > 0, np.log10(H_u), np.log10(min_rho_per_bin))
# 		logH_b = np.log10(H_b)
# 		logH_a0 = np.where(H_a > 0, 0., 1.)
# 		logH_a = np.where(H_a > 0, np.log10(H_a), np.log10(min_rho_per_bin))
# 
# 		cdict1 = {'red':   ((0.0, 0.0, 0.0),
# 						   (0.5, 0.0, 0.1),
# 						   (1.0, 1.0, 1.0)),
# 
# 				 'green': ((0.0, 0.0, 0.0),
# 						   (1.0, 0.0, 0.0)),
# 
# 				 'blue':  ((0.0, 0.0, 1.0),
# 						   (0.5, 0.1, 0.0),
# 						   (1.0, 0.0, 0.0))
# 				}
# 
# 
# 		blue_red1 = colors.LinearSegmentedColormap('BlueRed1', cdict1)
# 		plt.register_cmap(cmap=blue_red1)
# 
# 
# # 		xpos = np.where((-box_side/2. < xpos) & (xpos < box_side/2.), xpos, xpos - box_side/2.)
# # 		ypos = np.where((-box_side/2. < ypos) & (ypos < box_side/2.), ypos, ypos - box_side/2.)
# # 		zpos = np.where((-box_side/2. < zpos) & (zpos < box_side/2.), zpos, zpos - box_side/2.)
# 
# 		fig = plt.figure(figsize=(12.,10))	
# 		fig.subplots_adjust(wspace = .0,top = .9, bottom = .12, left = .13, right = .9)
# 		ax = fig.add_subplot(1,1,1, aspect = 'equal')
# # 		ax.xaxis.set_label_position('top') 
# # 		ax.xaxis.tick_top()
# 
# 		plt.axis(axis_range)
# #		ax.set_xticks([11,12,13,14])
# 
# 		for tick in ax.xaxis.get_major_ticks():
# 			tick.label1.set_fontsize(30)
# 		for tick in ax.yaxis.get_major_ticks():
# 			tick.label1.set_fontsize(30)
# 
# 		plt.axis(axis_range)
# 
# 
# 		cmap = colors.ListedColormap(['blue', 'purple', 'red'])
# 		bounds=[0, 0.33, 0.67, 1.0]
# 		cmap_w = colors.ListedColormap(['white'])
# #		cmap = colors.ListedColormap(['blue', 'red'])
# #		bounds=[0, 0.5, 1.0]
# 		norm_b = colors.BoundaryNorm(bounds, cmap.N)
# 
# 
# 		density = logH_a
# 		boundedness = (H_ab/H_a + 1.)/2.
# 		norm = colors.Normalize(density.min(), density.max())
# #		color_array = plt.get_cmap('BlueRed1')(boundedness)
# 		color_array = cmap(boundedness)
# 		white_array = cmap_w(logH_a0)
# 		print color_array.shape
# 		color_array[..., 3] = 0.5 + 0.5*norm(density)  # <- some alpha values between 0.5-1
# 		white_array[..., 3] = logH_a0  # <- some alpha values between 0.5-1
# #		print color_array
# 
# 		white_array
# 		
# 		im = plt.imshow(color_array, interpolation='spline16', 
#                 origin='lower', extent=axis_range)
# 		plt.imshow(white_array, interpolation='spline16',
#                 origin='lower', extent=axis_range, alpha = 0.5)
# 
# 		
# 		cs = plt.contour(X, Y, logH_b, 10, cmap = cm.Reds)
# 
# 		Rvir1=plt.Circle((0,0),Rvir,color='k',fill=False, lw = 3)
# #			, label = r'$1\ \mathrm{R}_{\mathrm{vir}}$')
# 		Rvir6=plt.Circle((0,0),(nRvir + 1)*Rvir,color='k',fill=False, lw = 5, ls = 'dashed')
# #			, label = r'$6\ \mathrm{R}_{\mathrm{vir}}$')
# 		ax.add_patch(Rvir1)
# 		ax.add_patch(Rvir6)
# 
# 		text1 = r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}(z=0)/h^{-1} \mathrm{M}_{\odot}) = $' + str(mass_range)
# 		text2 = r'$z = $' + zed
# 		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$13$' + '\n' + text2, size = 34, color = 'k', 
# 			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
# 
# 
# #		divider = make_axes_locatable(ax)
# #		cax = divider.append_axes("right", size="5%", pad=0.1)
# #		cbar = plt.colorbar(im, cax = cax, ticks = [0.0, 0.5, 1.0], cmap = cm.hot)
# #		cbar.set_label(r'$\mathrm{Bound}\ \mathrm{Fraction}$', fontsize=34) #54 for square
# 
# 
# 		
# 		ax.set_xlabel(r'$\mathrm{X}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) 
# 		ax.set_ylabel(r'$\mathrm{Y}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) #54 for square
# # 		ax.legend(prop = matplotlib.font_manager.FontProperties(size=22),fancybox=True,loc=0)
# #		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$13$' + '\n' + text2, size = 34, color = 'k', 
# #			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
# #		plt.text(12.5, -0.4, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')
# 
# 		outputFile = plot_dir + 'Galaxy_Window_contours_' + str(mass_range) + '_' + gal + '_' + snap + OutputFormat
# 		plt.savefig(outputFile)  # Save the figure
# 		print 'Saved file to', outputFile
# 		plt.close()
# 
# 	def Galaxy_Window_contours_set(self, galnum, snaps, mass_range) :
# 		gal = "%03d" % galnum
# 
# 		Rvir_a = []
# 		logH_aa = []
# 		logH_au = []
# 		logH_ab = []
# 		logH_abb = []
# 
# 		axis_range = [-2., 2., -2., 2.]
# 		
# 		n_hist_bins = 20
# 		hist_bins = np.linspace(-2.,2., n_hist_bins + 1)
# 
# 		win_size = axis_range[1] - axis_range[0]
# 		bin_side = (axis_range[1] - axis_range[0])/float(n_hist_bins)
# 		min_rho_per_bin = (1. / (bin_side**2 * win_size))/(270./box_side)**3
# 		print np.log10(min_rho_per_bin)
# 		dumX = hist_bins[:n_hist_bins] + (hist_bins[1] - hist_bins[0])/2.
# 		X, Y = np.meshgrid(dumX, dumX)
# 
# 
# 		for snapnum in snaps : 
# 			snap = "%03d" % snapnum
# 		
# 			Rvir, xpos, ypos, zpos, bound = res.read_fixed_window(galnum, mass_range, snapnum)
# 			Rvir_a.append(Rvir)
# 
# 
# 			print np.sum(bound), '= bound - unbound'
# 		
# 			ww = np.where((-2. < zpos) & (zpos < 2.))[0]
# 			print len(ww), 'particles shown out of', len(xpos)
# 			xpos = xpos[ww]
# 			ypos = ypos[ww]
# 			zpos = zpos[ww]
# 			bound = bound[ww]
# 		
# 			wb = np.where(bound > 0.)[0]
# 			xpos_b = xpos[wb]
# 			ypos_b = ypos[wb]
# 			zpos_b = zpos[wb]
# 
# 			wu = np.where(bound < 0.)[0]
# 			xpos_u = xpos[wu]
# 			ypos_u = ypos[wu]
# 			zpos_u = zpos[wu]
# 		
# 			H_a, xedges, yedges = np.histogram2d(ypos, xpos, bins = hist_bins)
# 			H_u, xedges, yedges = np.histogram2d(ypos_u, xpos_u, bins = hist_bins)
# 			H_b, xedges, yedges = np.histogram2d(ypos_b, xpos_b, bins = hist_bins)
# 
# 		
# 			H_a = H_a * (1. / (bin_side**2 * win_size))/(270./box_side)**3
# 			H_u = H_u * (1. / (bin_side**2 * win_size))/(270./box_side)**3
# 			H_b = H_b * (1. / (bin_side**2 * win_size))/(270./box_side)**3
# 
# 			logH_a = np.where(H_a > 0, np.log10(H_a), np.log10(min_rho_per_bin))
# 			logH_u = np.where(H_u > 0, np.log10(H_u), np.log10(min_rho_per_bin))
# 			logH_b = np.log10(H_b)
# 			logH_bb = np.where(H_b > 0, np.log10(H_b), np.log10(min_rho_per_bin))
# 
# 			logH_aa.append(logH_a)
# 			logH_au.append(logH_u)
# 			logH_ab.append(logH_b)
# 			logH_abb.append(logH_bb)
# 			
# 			
# 
# 
# 			print 'min, max all:', np.min(logH_a), np.max(logH_a)
# 			print 'min, max unbound:', np.min(logH_u), np.max(logH_u)
# 			print 'min, max bound:', np.min(logH_b), np.max(logH_b)
# 
# 
# 		logH_aa = np.array(logH_aa)
# 		logH_au = np.array(logH_au)
# 		logH_ab = np.array(logH_ab)
# 		logH_abb = np.array(logH_abb)
# 
# 		fig = plt.figure(figsize=(20.,7))	
# 		fig.subplots_adjust(wspace = .0,top = .89, bottom = .12, left = .1, right = .95)
# 
# 		for ii in range(len(snaps)) :
# 			snapnum = snaps[ii]
# 			zed = "%1.2f" % z[snapnum]
# 
# 
# 			ax = fig.add_subplot(1,4,ii + 1, aspect = 'equal')
# 			ax.xaxis.set_label_position('top') 
# 			ax.xaxis.tick_top()
# 
# 			plt.axis(axis_range)
# 			if (ii == 3) : ax.set_xticks([-2, -1, 0, 1, 2])
# 			else : ax.set_xticks([-2, -1, 0, 1])
# 			if (ii == 0) : ax.set_yticks([-2, -1, 0, 1, 2])
# 			else : ax.set_yticks([])
# 
# 
# 
# 			for tick in ax.xaxis.get_major_ticks():
# 				tick.label1.set_fontsize(30)
# 			for tick in ax.yaxis.get_major_ticks():
# 				tick.label1.set_fontsize(30)
# 
# 
# 
# 
# 			boundedness = (10**(logH_ab[ii] - logH_aa[ii]) + 1.)/2.
# #			boundedness = np.where(boundedness > 0., boundedness, 0.)
# 			density_b = logH_abb[ii]
# 			norm = colors.Normalize(-0.5, 2.5) #logH_abb[ii].min(), logH_abb[ii].max())
# 			
# 						
# 			bound_array = plt.get_cmap('Greens')(norm(density_b)*0.9)
# 
# 			print np.size(bound_array[..., 3]), np.size(boundedness)
# 
# 			bound_array[..., 3] = boundedness*0.9  # <- some alpha values between 0.5-1
# 		
# 			im2 = plt.imshow(logH_ab[ii], interpolation='sinc', cmap=cm.Greens,
# 					origin='lower', extent=axis_range,
# 					vmax=logH_ab.max(), vmin=logH_au.min(), alpha = 0.7)
# 			im = plt.imshow(logH_au[ii], interpolation='sinc', cmap=cm.Purples,
# 					origin='lower', extent=axis_range,
# 					vmax=logH_au.max(), vmin=logH_au.min())
# 			plt.imshow(bound_array, interpolation='spline36', 
# 					origin='lower', extent=axis_range)
# 
# 			levels = [1., 1.5, 2.0, 2.5, 3.0, 3.5]
# 			if (np.max(logH_ab) < 3.5) : levels = [1., 1.5, 2.0, 2.5, 3.0]
# 		
# 			cs = plt.contour(X, Y, logH_ab[ii], levels = levels, cmap = cm.YlGn)
# 
# 
# 			print Rvir_a[ii]
# 			Rvir1=plt.Circle((0,0),Rvir_a[ii],color='k',fill=False, lw = 3)
# 	#			, label = r'$1\ \mathrm{R}_{\mathrm{vir}}$')
# 			Rvir6=plt.Circle((0,0),(6.)*Rvir_a[ii],color='k',fill=False, lw = 5, ls = 'dashed')
# 	#			, label = r'$6\ \mathrm{R}_{\mathrm{vir}}$')
# 			ax.add_patch(Rvir1)
# 			ax.add_patch(Rvir6)
# 
# 			text1 = r'$\mathrm{log}_{10}(\mathrm{M}_{\mathrm{vir}}(z=0)/h^{-1} \mathrm{M}_{\odot}) = $' + str(mass_range)
# 			text2 = r'$z = $' + zed
# 			plt.text(-1.9, -1.9, text2, size = 34, color = 'k', 
# 				verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
# 
# 
# 			divider = make_axes_locatable(ax)
# 			cax2 = divider.append_axes("bottom", size="5%", pad=0.1)
# 			if (ii % 2 == 0) : 
# #				cax = divider.append_axes("right", size="5%", pad=0.1)
# 				cbar = plt.colorbar(im, cax = cax2, orientation='horizontal', ticks = [-0.5, 0.0, 0.5, 1, 2.0, 3.0])
# 				cbar.set_label(r'$\mathrm{log}_{10}(\rho_{\mathrm{Unbound}}/\bar{\rho})$', fontsize=34) #54 for square
# 			else :
# 				cbar2 = plt.colorbar(im2, cax = cax2, orientation='horizontal', ticks = [0.0, 1.0, 2.0, 3.0, 4.0])
# 				cbar2.set_label(r'$\mathrm{log}_{10}(\rho_{\mathrm{Bound}}/\bar{\rho})$', fontsize=34) #54 for square
# 
# 
# 		
# 			ax.set_xlabel(r'$\mathrm{X}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) 
# 			if (ii == 0) : ax.set_ylabel(r'$\mathrm{Y}\ \mathrm{[}h^{-1}\mathrm{Mpc]}$', fontsize=34) #54 for square
# 	# 		ax.legend(prop = matplotlib.font_manager.FontProperties(size=22),fancybox=True,loc=0)
# 	#		plt.text(-(nRvir+2.5)*Rvir, -(nRvir+2.5)*Rvir, r'$13$' + '\n' + text2, size = 34, color = 'k', 
# 	#			verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.9))
# 	#		plt.text(12.5, -0.4, r'$\mathrm{r}_{\mathrm{FA}} = 2 h^{-1}\mathrm{Mpc}$', size = 25, color = 'k', verticalalignment='bottom', horizontalalignment='left')
# 
# 		outputFile = plot_dir + 'Evolution_Galaxy_Window_contours_' + str(mass_range) + '_' + gal + OutputFormat
# 		plt.savefig(outputFile)  # Save the figure
# 		print 'Saved file to', outputFile
# 		plt.close()
# 
# 	def plot_pvalues(self, N1, N2, pvalues, r_FA) :
# 		fig = plt.figure(figsize=(12.,10))	
# 		fig.subplots_adjust(wspace = .0,top = .96, bottom = .12, left = .12, right = .95)
# 		ax = fig.add_subplot(1,1,1)
# 
# 		plt.axis([0,2.05,-0.1,1])
# 
# 		for tick in ax.xaxis.get_major_ticks():
# 			tick.label1.set_fontsize(30)
# 		for tick in ax.yaxis.get_major_ticks():
# 			tick.label1.set_fontsize(30)
# 
# 		label1 = ''
# 		label2 = ''
# 		if N1 == 'Mdiff' :
# 			label1 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
# 		if N1 == 'Mvir' :
# 			label1 = r'$\mathrm{M}_{\mathrm{vir}}$'
# 
# 		if N2 == 'Mdiff' :
# 			label2 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
# 		if N2 == str(r_FA) + '.0FA' :
# 			label2 = r'$\mathrm{FA}$'
# 
# 		for rr in range(1, 10) :
# 			label = str(rr + 1) + r'$\mathrm{R}_{\mathrm{vir}}$'
# 			col = cm.jet((rr-1)/8.)
# 			ax.plot(z, pvalues[rr], label = label, c = col)
# 		
# 
# 		ax.set_xlabel(r'$z$', fontsize=34) #54 for square
# 		ax.set_ylabel(r'$\mathrm{Pearson}\ \mathrm{p-value}$ (' + label1 + '$\mathrm{vs}$ ' + label2 + ')', fontsize=34) #54 for square
# 		ax.legend(prop = matplotlib.font_manager.FontProperties(size=25),fancybox=True,loc=0, ncol = 3)
# 
# 		outputFile = plot_dir + N1 + '_vs_' + N2 + '_' + space + '_pvalues' + OutputFormat
# 		plt.savefig(outputFile)  # Save the figure
# 		print 'Saved file to', outputFile
# 		plt.close()
# 
# 	def plot_pvalues_Mbins(self, Mvir, N1, N2, pvalues_bins, r_FA) :
# 
# 		fig = plt.figure(figsize=(12.,10))	
# 		fig.subplots_adjust(wspace = 0.15, hspace = 0.05, top = .96, bottom = .12, left = .12, right = .95)
# 
# 		for mm in range(4) : 
# 			Mmin = 10. #np.min(logMvir)
# 			bin = 1. #(np.max(logMvir) - np.min(logMvir))/4.
# 			Mbin_min = "%2.1f" % (Mmin + mm*bin)
# 			Mbin_max = "%2.1f" % (Mmin + (mm + 1)*bin)
# 
# 			ax = fig.add_subplot(2,2,mm + 1)
# 
# 			plt.axis([0,2.05,-0.1,1])
# 
# 			for tick in ax.xaxis.get_major_ticks():
# 				tick.label1.set_fontsize(18)
# 
# 			for tick in ax.yaxis.get_major_ticks():
# 				tick.label1.set_fontsize(18)
# 				
# 			if mm < 2 : ax.set_xticklabels([])
# #			if mm == 1 : ax.set_yticklabels([])
# #			if mm == 3 : ax.set_yticklabels([])
# 
# 			label1 = ''
# 			label2 = ''
# 			if N1 == 'Mdiff' :
# 				label1 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
# 			if N1 == 'Mvir' :
# 				label1 = r'$\mathrm{M}_{\mathrm{vir}}$'
# 
# 			if N2 == 'Mdiff' :
# 				label2 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
# 			if N2 == str(r_FA) + '.0FA' :
# 				label2 = r'$\mathrm{FA}$'
# 
# 			for rr in range(1, 10) :
# 				label = str(rr + 1) + r'$\mathrm{R}_{\mathrm{vir}}$'
# 				col = cm.jet((rr-1)/8.)
# 				ax.plot(z, pvalues_bins[mm,rr], label = label, c = col)
# 		
# 
# 			ax.set_xlabel(r'$z$', fontsize=34) #54 for square
# #			ax.set_ylabel(r'$\mathrm{Pearson}\ \mathrm{p-value}$ (' + label1 + '$\mathrm{vs}$ ' + label2 + ')', fontsize=34) #54 for square
# 			if mm == 2 : 
# 				ax.legend(prop = matplotlib.font_manager.FontProperties(size=18),
# 					fancybox=True,loc=0, ncol = 3, borderpad = 0.1, columnspacing = 0.1)
# 			log = r'$\mathrm{log}_{10}$'
# 			Msun = r'$\mathrm{M}_{\odot}$'
# 
# 			text2 = Mbin_min + r'$< (\mathrm{M}_{\mathrm{vir}}/ h^{-1} \mathrm{M}_{\odot}) <$' + Mbin_max
# 			plt.text(2.0, 0.96, text2, size = 20, color = 'k', 
# 				verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.9))
# 			ylabel = r'$\mathrm{Pearson}\ \mathrm{p-value}$ (' + label1 + '$\mathrm{vs}$ ' + label2 + ')'
# 		fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', size = 30)
# 
# 		outputfile = plot_dir + N1 + '_vs_' + N2 + '_' + space + '_pvalues_bins' 
# 		if shuffle == 'All' : outputFile = outputfile + '_shuffle'
# 		if shuffle == 'Mbins' : outputFile = outputfile + '_shuffle_Mbins'
# 		outputFile = outputFile + OutputFormat
# 		print 'Saving file to', outputFile
# 		plt.savefig(outputFile)  # Save the figure
# 		print 'Saved file to', outputFile
# 		plt.close()
# 
# 	def plot_pvalues_rhobins(self, Mvir, N1, N2, pvalues_bins, r_FA) :
# 
# 		fig = plt.figure(figsize=(12.,10))	
# 		fig.subplots_adjust(wspace = 0.15, hspace = 0.05, top = .96, bottom = .12, left = .12, right = .95)
# 
# 		for pp in range(4) : 
# 			rhomin = -0.12 #np.min(logMvir)
# 			bin = 0.65 #(np.max(logMvir) - np.min(logMvir))/4.
# 			rhobin_min = "%1.2f" % (rhomin + pp*bin)
# 			rhobin_max = "%1.2f" % (rhomin + (pp + 1)*bin)
# 
# 			ax = fig.add_subplot(2,2,pp + 1)
# 
# 			plt.axis([0,2.05,-0.1,1])
# 
# 			for tick in ax.xaxis.get_major_ticks():
# 				tick.label1.set_fontsize(18)
# 
# 			for tick in ax.yaxis.get_major_ticks():
# 				tick.label1.set_fontsize(18)
# 				
# 			if pp < 2 : ax.set_xticklabels([])
# #			if pp == 1 : ax.set_yticklabels([])
# #			if pp == 3 : ax.set_yticklabels([])
# 
# 			label1 = ''
# 			label2 = ''
# 			if N1 == 'Mdiff' :
# 				label1 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
# 			if N1 == 'Mvir' :
# 				label1 = r'$\mathrm{M}_{\mathrm{vir}}$'
# 
# 			if N2 == 'Mdiff' :
# 				label2 = r'$\mathrm{M}_{\mathrm{diffuse}}$'
# 			if N2 == str(r_FA) + '.0FA' :
# 				label2 = r'$\mathrm{FA}$'
# 
# 			for rr in range(1, 10) :
# 				label = str(rr + 1) + r'$\mathrm{R}_{\mathrm{vir}}$'
# 				col = cm.jet((rr-1)/8.)
# 				ax.plot(z, pvalues_bins[pp,rr], label = label, c = col)
# 		
# 
# 			ax.set_xlabel(r'$z$', fontsize=34) #54 for square
# #			ax.set_ylabel(r'$\mathrm{Pearson}\ \mathrm{p-value}$ (' + label1 + '$\mathrm{vs}$ ' + label2 + ')', fontsize=34) #54 for square
# 			if pp == 0 : 
# 				ax.legend(prop = matplotlib.font_manager.FontProperties(size=18),
# 					fancybox=True,loc=0, ncol = 3, borderpad = 0.1, columnspacing = 0.1)
# 			log = r'$\mathrm{log}_{10}$'
# 			Msun = r'$\mathrm{M}_{\odot}$'
# 
# 			text2 = rhobin_min + r'$< (\rho_{\mathrm{FA}} / \bar{\rho}) <$' + rhobin_max
# 			plt.text(2.0, 0.96, text2, size = 20, color = 'k', 
# 				verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.9))
# 			ylabel = r'$\mathrm{Pearson}\ \mathrm{p-value}$ (' + label1 + '$\mathrm{vs}$ ' + label2 + ')'
# 		fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', size = 30)
# 
# 		outputfile = plot_dir + N1 + '_vs_' + N2 + '_' + space + '_pvalues_bins' 
# 		if shuffle == 'All' : outputFile = outputfile + '_shuffle'
# 		if shuffle == 'Mbins' : outputFile = outputfile + '_shuffle_Mbins'
# 		if shuffle == 'None' : outputFile = outputfile
# 		outputFile = outputFile + OutputFormat
# 		print 'Saving file to', outputFile
# 		plt.savefig(outputFile)  # Save the figure
# 		print 'Saved file to', outputFile
# 		plt.close()
# 	def rho_vs_Mdiffuse_radial_Mstar_bins(self, snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse, Mstars) :
# 		snap = "%03d" % snapnum
# 		zed = "%1.2f" % z[snapnum]
# 
# 		fig = plt.figure(figsize=(12.,10))	
# 		fig.subplots_adjust(wspace = 0.15, hspace = 0.04, top = .96, bottom = .12, left = .12, right = .95)
# 
# 		
# 
# 		Mdiff = Mdiffuse[: , nRvir - 1]
# 		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
# 		w = np.where((type == 0) & (Mdiff > 0))[0]
# 		print 'plotting Mvir vs Mdiffuse within', nRvir, 'Rvir for a', r_FA, 'Mpc aperture'
# 	
# 		label = mu + '(<' + str(nRvir) + rv + ')'
# 		label1 = log10 + '(' + rhob + ')' + '[' + str(r_FA) + mpc + ']'
# 		label2 = log10 + '(' + mu + '/' + msun + ')'
# 		label3 = log10 + '(' + ms + '/' + msun + ')'
# 		col = cm.spectral((nRvir-1)/7.)
# 
# 		logrho = np.log10(rho[w])
# 		logMdiff = np.log10(Mdiff[w])
# 		logMstars = np.log10(Mstars[w])	
# 		print logMstars
# 	
# 		Mmin = np.min(logMstars)
# 		bin = (np.max(logMstars) - np.min(logMstars))/4.
# 		for mm in range(4) : 
# 			mbin_min = (Mmin + mm*bin)
# 			mbin_max = (Mmin + (mm + 1)*bin)
# 			Mbin_min = "%2.1f" % mbin_min
# 			Mbin_max = "%2.1f" % mbin_max
# 			w = np.where((mbin_min <= logMstars) & (logMstars < mbin_max))[0]
# 			print len(w), 'galaxies in mass bin', Mbin_min, '-', Mbin_max
# 			
# 			logrho_w = logrho[w]
# 			logMdiff_w = logMdiff[w]
# 
# 			ax = fig.add_subplot(2,2,mm + 1)
# 
# 			plt.axis([-1.6,2.6,9.5,14.1])
# 
# 			for tick in ax.xaxis.get_major_ticks():
# 				tick.label1.set_fontsize(18)
# 
# 			for tick in ax.yaxis.get_major_ticks():
# 				tick.label1.set_fontsize(18)
# 				
# 			if mm < 2 : ax.set_xticklabels([])
# 
# 			ax.scatter(logrho_w, logMdiff_w, c = col, edgecolor = col, alpha = 0.3, label = label)
# 		
# 
# 			if mm == 2 : 
# 				ax.legend(prop = matplotlib.font_manager.FontProperties(size=18),
# 					fancybox=True,loc=4, ncol = 3, borderpad = 0.1, columnspacing = 0.1)
# 			log = r'$\mathrm{log}_{10}$'
# 			Msun = r'$\mathrm{M}_{\odot}$'
# 
# 			text2 = Mbin_min + '< (' + ms + '/' + msun + ') <' + Mbin_max
# 			plt.text(0.5, 14, text2, size = 20, color = 'k', 
# 				verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.9))
# 			if mm == 3 : plt.text(2.5, 9.6, '$z = $' + zed, size = 40, color = 'k', verticalalignment='bottom', horizontalalignment='right')
# 
# 
# 		fig.text(0.5, 0.04, label1, ha='center', size = 30)
# 		fig.text(0.04, 0.5, label2, va='center', rotation='vertical', size = 30)
# 
# 		outputFile = plot_dir + 'Mstars_bins_rho_vs_Mdiff_ra_' + str(nRvir) + 'Rvir_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
# 		plt.savefig(outputFile)  # Save the figure
# 		print 'Saved file to', outputFile
# 		plt.close()
# 
# 	def rho_vs_Mdiffuse_aperture_Mstar_bins(self, snapnum, r_FA, type, rhorhobar_FA, Mdiffuse, Mstars) :
# 		snap = "%03d" % snapnum
# 		zed = "%1.2f" % z[snapnum]
# 
# 		fig = plt.figure(figsize=(12.,10))	
# 		fig.subplots_adjust(wspace = 0.15, hspace = 0.04, top = .96, bottom = .12, left = .12, right = .95)
# 
# 		
# 
# 		Mdiff = Mdiffuse[: , int(r_FA*2) - 1]
# 		rho = rhorhobar_FA[: , int(r_FA*2) - 1]
# 		w = np.where((type == 0) & (Mdiff > 0))[0]
# 		print 'plotting Mvir vs Mdiffuse within', r_FA, 'Mpc/h for', len(w), 'galaxies'
# 	
# 		label = mu + '(<' + str(r_FA) + mpc + ')'
# 		label1 = log10 + '(' + rhob + ')' + '[' + str(r_FA) + mpc + ']'
# 		label2 = log10 + '(' + mu + '/' + msun + ')'
# 		label3 = log10 + '(' + ms + '/' + msun + ')'
# 
# 		col = cm.gnuplot((r_FA+2)/8.)
# 
# 
# 		logrho = np.log10(rho[w])
# 		logMdiff = np.log10(Mdiff[w])
# 		logMstars = np.log10(Mstars[w])	
# 		print logMstars
# 	
# 		Mmin = np.min(logMstars)
# 		bin = (np.max(logMstars) - np.min(logMstars))/4.
# 		for mm in range(4) : 
# 			mbin_min = (Mmin + mm*bin)
# 			mbin_max = (Mmin + (mm + 1)*bin)
# 			Mbin_min = "%2.1f" % mbin_min
# 			Mbin_max = "%2.1f" % mbin_max
# 			w = np.where((mbin_min <= logMstars) & (logMstars < mbin_max))[0]
# 			print len(w), 'galaxies in mass bin', Mbin_min, '-', Mbin_max
# 			
# 			logrho_w = logrho[w]
# 			logMdiff_w = logMdiff[w]
# 
# 			fit = np.polyfit(logrho_w, logMdiff_w, 1)
# 			print 'log(Mdiff) =', fit[0], '* log(rho/rhobar) +', fit[1]
# 
# 
# 			ax = fig.add_subplot(2,2,mm + 1)
# 
# 			plt.axis([-1.6,2.6,9.5,14.1])
# 
# 			for tick in ax.xaxis.get_major_ticks():
# 				tick.label1.set_fontsize(18)
# 
# 			for tick in ax.yaxis.get_major_ticks():
# 				tick.label1.set_fontsize(18)
# 				
# 			if mm < 2 : ax.set_xticklabels([])
# #			if mm == 1 : ax.set_yticklabels([])
# #			if mm == 3 : ax.set_yticklabels([])
# 
# 			ax.scatter(logrho_w, logMdiff_w, c = col, edgecolor = col, alpha = 0.3, label = label)
# 		
# 
# 			if mm == 2 : 
# 				ax.legend(prop = matplotlib.font_manager.FontProperties(size=18),
# 					fancybox=True,loc=0, ncol = 3, borderpad = 0.1, columnspacing = 0.1)
# 
# 			text2 = Mbin_min + '< (' + ms + '/' + msun + ') <' + Mbin_max
# 			plt.text(0.5, 9.6, text2, size = 20, color = 'k', 
# 				verticalalignment='bottom', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.9))
# 			if mm == 1 : plt.text(2.5, 14, '$z = $' + zed, size = 40, color = 'k', verticalalignment='top', horizontalalignment='right')
# 
# 
# 		fig.text(0.5, 0.04, label1, ha='center', size = 30)
# 		fig.text(0.04, 0.5, label2, va='center', rotation='vertical', size = 30)
# 
# 		outputFile = plot_dir + 'Mstars_bins_rho_vs_Mdiff_ap_' + str(r_FA) + 'Mpc_' + space + '_' + snap + OutputFormat
# 		plt.savefig(outputFile)  # Save the figure
# 		print 'Saved file to', outputFile
# 		plt.close()

		
		


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
		type, Mvir, Mstars, ColdGas, IDs, Vdisp, GasMetals, SFR = rd.read_gals(snapnum)
		print 'highest stellar mass = ', np.log10(np.max(Mstars))
		print "Reading in unbound matter in a fixed aperture." 
		Mdiffuse_FA = rd.read_aperture_distributions(snapnum, len(type))
		print "Reading in unbound matter in a radial aperture." 
		Mdiffuse_Rvir = rd.read_radial_distributions(snapnum, len(type))
		print "Reading in fixed aperture galaxy counts." 
		rhorhobar_FA = rd.read_fixed_aperture(snapnum, len(type))

		res.Mvir_vs_Mdiffuse_radial_all(snapnum, type, Mvir, Mdiffuse_Rvir)			
		res.Mvir_vs_Mdiffuse_aperture_all(snapnum, type, Mvir, Mdiffuse_FA)			
# 		res.Mvir_vs_Mdiffuse_freefall_radial(snapnum, type, Mvir, Mdiffuse_Rvir)
# 		res.Mvir_vs_Mdiffuse_freefall_aperture(snapnum, type, Mvir, Mdiffuse_FA)
# 		res.Mstars_vs_Mdiffuse_freefall_radial(snapnum, type, Mvir, Mstars, Mdiffuse_Rvir)
# 		res.Mstars_vs_Mdiffuse_freefall_aperture(snapnum, type, Mvir, Mstars, Mdiffuse_FA)
# 		res.Mstars_vs_Mdiffuse_radial_all(snapnum, type, Mstars, Mdiffuse_Rvir)
# 		res.Mstars_vs_Mdiffuse_aperture_all(snapnum, type, Mstars, Mdiffuse_FA)
# 		res.Mstars_Mcold_vs_Mdiffuse_radial_all(snapnum, type, Mstars, ColdGas, Mdiffuse_Rvir)
# 		res.Vdisp_vs_Mdiffuse_radial_all(snapnum, type, Vdisp, Mdiffuse_Rvir)
# 		res.Vdisp_vs_Mdiffuse_aperture_all(snapnum, type, Vdisp, Mdiffuse_FA)


# 		for nRvir in ns :
# 			res.Mstars_vs_Mdiffuse_radial(snapnum, nRvir, type, Mstars, Mdiffuse_Rvir)			
# 			res.Mvir_vs_Mdiffuse_radial(snapnum, nRvir, type, Mvir, Mdiffuse_Rvir)			
# 			res.Vdisp_vs_Mdiffuse_radial(snapnum, nRvir, type, Vdisp, Mdiffuse_Rvir)
# 			res.Mstars_vs_Mdiffuse_radial_tag_fg(snapnum, nRvir, type, Mstars, Mdiffuse_Rvir, ColdGas)
# 			res.Mstars_vs_Mdiffuse_radial_tag_ZZ(snapnum, nRvir, type, Mstars, Mdiffuse_Rvir, GasMetals)
# 			res.Vdisp_vs_Mdiffuse_radial_tag_ZZ(snapnum, nRvir, type, Vdisp, Mdiffuse_Rvir, GasMetals)
# 			res.Vdisp_vs_Mdiffuse_radial_tag_sSFR(snapnum, nRvir, type, Vdisp, Mdiffuse_Rvir, SFR/Mstars)
# 			for r_FA in r_FAs :
# 				res.rho_vs_Mdiffuse_radial(snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse_Rvir)
# 				res.rho_vs_Mdiffuse_radial_Mstar_bins(snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse_Rvir, Mstars)
# 				res.Mstar_vs_Mdiffuse_radial_rho_bins(snapnum, r_FA, nRvir, type, rhorhobar_FA, Mdiffuse_Rvir, Mstars)
# 				res.Vdisp_vs_Mdiffuse_radial_tag_FA(snapnum, nRvir, type, Vdisp, Mdiffuse_Rvir, rhorhobar_FA)
# 				res.Mstars_vs_Mdiffuse_radial_tag_FA(snapnum, nRvir, type, Mstars, Mdiffuse_Rvir, rhorhobar_FA)

# 		for r_FA in r_FAs :
# 			res.Vdisp_vs_Mdiffuse_aperture_tag_FA(snapnum, r_FA, type, Vdisp, Mdiffuse_FA, rhorhobar_FA)
# 			res.Mstars_vs_Mdiffuse_aperture(snapnum, r_FA, type, Mstars, Mdiffuse_FA)			
# 			res.Mvir_vs_Mdiffuse_aperture(snapnum, r_FA, type, Mvir, Mdiffuse_FA)			
# 			res.Vdisp_vs_Mdiffuse_aperture(snapnum, r_FA, type, Vdisp, Mdiffuse_FA)
# 			res.rho_vs_Mdiffuse_aperture(snapnum, r_FA, type, rhorhobar_FA, Mdiffuse_FA)
# 			res.rho_vs_Mdiffuse_aperture_Mstar_bins(snapnum, r_FA, type, rhorhobar_FA, Mdiffuse_FA, Mstars)
# 			res.Mstars_vs_Mdiffuse_aperture_tag_FA(snapnum, r_FA, type, Mstars, Mdiffuse_FA, rhorhobar_FA)
# 			res.Mstars_vs_Mdiffuse_aperture_tag_ZZ(snapnum, r_FA, type, Mstars, Mdiffuse_FA, GasMetals)
# 			res.Vdisp_vs_rho(snapnum, r_FA, type, Vdisp, rhorhobar_FA, Mdiffuse_FA)
# 			res.Mstars_vs_Mdiffuse_aperture_tag_fg(snapnum, r_FA, type, Mstars, Mdiffuse_FA, ColdGas)





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
