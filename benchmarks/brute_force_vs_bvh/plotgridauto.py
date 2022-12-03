# Import libraries
import sys
import os
import string
from math import *
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

def procdata(x1,y1,z1,z2):
    "Processes data for plotting"
    lx1= [log2(x1[ii]) for ii in range(len(x1))]
    ly1= [log2(y1[ii]) for ii in range(len(y1))]
    zdiff = [z2[ii] - z1[ii] for ii in range(len(z1))]
    lzdiff = [log10(z2[ii]) - log10(z1[ii]) for ii in range(len(z1))]
    return lx1,ly1,lzdiff

def getlzpn(lx,ly,lz):
    "Partitions data into positive and negative components"
    lxp=[]
    lxn=[]
    lyp=[]
    lyn=[]
    lzp=[]
    lzn=[]
    for ii in range(len(lz)):
        if lz[ii]>0:
           lxp.append(lx[ii])
           lyp.append(ly[ii])
           lzp.append(lz[ii])
        else:
           lxn.append(lx[ii])
           lyn.append(ly[ii])
           lzn.append(lz[ii])
    return lxp,lyp,lzp,lxn,lyn,lzn

def file2buffer(fnm):
    "Reads file into buffer"
    f = open(fnm,'r')
    buf = []
    for line in f.readlines():
        buf.append(line)
    f.close()
    return buf

def extractdatacols_nv(buf):
    "Gets data columns from buffer (format from Nvidia NSight tools)"
    sig=[]
    npts=[]
    time=[]
    for ii in range(len(buf)):
        buftmp=buf[ii].split()
        nmproc=buftmp[0].split('_')
        tmg=buftmp[2].replace(',','')
        sig.append(float(nmproc[1]))
        npts.append(int(nmproc[2]))
        time.append(int(tmg))
    return sig,npts,time

def extractdatacols_kp(buf):
    "Gets data columns from buffer (format from Kokkos tooks kp-connecter)"
    sig=[]
    npts=[]
    time=[]
    for ii in range(len(buf)):
        buftmp=buf[ii].split()
        nmproc=buftmp[0].split('_')
        sig.append(float(nmproc[1]))
        nmproc2=nmproc[2].split('.')
        npts.append(int(nmproc2[0]))
        timeidx=buftmp.index('sec')-1
        time.append(float(buftmp[timeidx]))
    return sig,npts,time

def extractdatacols_verboseout(buf):
    "Gets data columns from buffer (format from DBSCAN with verbose output)"
    sig=[]
    npts=[]
    time=[]
    for ii in range(len(buf)):
        buftmp=buf[ii].split()
        nmproc=buftmp[0].split('_')
        sig.append(float(nmproc[1]))
        nmproc2=nmproc[2].split('.')
        npts.append(int(nmproc2[0]))
        time.append(float(buftmp[-1]))
    return sig,npts,time


def data2file(x,y,z1,z2,run):
    "Writes data to csv file"
    fnm='brute-bvh_perftest'+run+'.csv'
    f = open(fnm,'w')
    bru='t_brute_'+run+' (s)'
    bvh='t_bvh_'+run+' (s)'
    ratio='(t_bvh/t_brute)'
    f.write('epsilon,n_points,%s,%s,%s\n' %(bru,bvh,ratio))
    for ii in range(len(x)):
        f.write('%9.3E,%d,%8.2E,%8.2E,%8.2E\n' %(x[ii],y[ii],z1[ii],z2[ii],z2[ii]/z1[ii]))
    f.close()
    return

def gridify(x,y,z,xg,yg):
    "Generate grid with len(xg) by len(yg) values"
    grid = [[0 for ii in range(len(yg))] for jj in range(len(xg))]
    for ii in range(len(x)):
        xidx=xg.index(x[ii])
        yidx=yg.index(y[ii])
        if xidx > -1 and yidx > -1: grid[xidx][yidx]=z[ii]
    return grid

def trimgrid(xg,yg,z1g,z2g):
    "Remove entries which are zero in either z1g or z2g"
    xr=[]
    yr=[]
    z1r=[]
    z2r=[]
    for ii in range(len(xg)):
        for jj in range(len(yg)):
            if z1g[ii][jj] > 0 and z2g[ii][jj] > 0:
                xr.append(xg[ii])
                yr.append(yg[jj])
                z1r.append(z1g[ii][jj])
                z2r.append(z2g[ii][jj])
    return xr,yr,z1r,z2r

# Interpolate onto fine trid for surface plot
def getfinegrid(lx1,ly1,lzdiff):
    "Generates fine grid for surface plot"
    Nfine=32
    lx1f, ly1f = np.linspace(min(lx1), max(lx1), Nfine), np.linspace(min(ly1), max(ly1), Nfine)
    lx1f, ly1f = np.meshgrid(lx1f, ly1f)
    rbf = scipy.interpolate.Rbf(lx1, ly1, lzdiff, function='linear')
    lz1f = rbf(lx1f, ly1f)
    return lx1f,ly1f,lz1f

# Plot surface
def makedataplot(lx1,ly1,lzdiff,run):
    "Generates surface/scatter plot of data"
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.set_xlabel("log_2 epsilon")
    ax.set_ylabel("log_2 Npoints")
    ax.set_zlabel("log_10 (t_bvh/t_brute)")
    title='BVH - Brute Performance Comparison ('+run+')'
    plt.title(title)
    # Surface contour data
    lx1f,ly1f,lz1f=getfinegrid(lx1, ly1, lzdiff)
    surf = ax.plot_surface(lx1f, ly1f, lz1f,
                       cmap='viridis', 
                       edgecolor='none', 
                       linewidth=0, 
                       alpha=0.25)
    cntr = ax.contour(lx1f, ly1f, lz1f,
           levels=14, 
           linewidths=1, 
           colors='k',
           offset=min(lzdiff))
    cntr1 = ax.contourf(lx1f, ly1f, lz1f,
        levels=14, 
        cmap="viridis",
        alpha=0.8,
        offset=min(lzdiff))
    fig.colorbar(cntr1, shrink=0.5, aspect=5)
    # Plot points from original data
    lxp,lyp,lzp,lxn,lyn,lzn=getlzpn(lx1,ly1,lzdiff)
    ax.scatter3D(lxp, lyp, lzp, s=12, color = "orange")
    ax.scatter3D(lxn, lyn, lzn, s=12, color = "green")
    # Show plot
    plt.gca().view_init(15.0, 135.0)
    plt.savefig('BVH-Brute_'+run+'.png',bbox_inches='tight')
#    plt.show()
    return

def main(run, profiler):
    "Main function"
    brufile='summary_brute_'+run+'.out'
    bvhfile='summary_bvh_'+run+'.out'
    # Read file, put in buffer
    brubuf=file2buffer(brufile)
    bvhbuf=file2buffer(bvhfile)
    # Convert data in buffer to lists
    if profiler == 'nvidia':
       x1,y1,z1=extractdatacols_nv(brubuf)
       x2,y2,z2=extractdatacols_nv(bvhbuf)
    elif profiler == 'kp':
       x1,y1,z1=extractdatacols_kp(brubuf)
       x2,y2,z2=extractdatacols_kp(bvhbuf)
    elif profiler == 'none':
       x1,y1,z1=extractdatacols_verboseout(brubuf)
       x2,y2,z2=extractdatacols_verboseout(bvhbuf)
    # Find list of epsilon, npts values in union of both data sets
    thexs=sorted(list(set(x1+x2)))
    theys=sorted(list(set(y1+y2)))
    # Put timings in grid corresponding to union of data sets
    z1g=gridify(x1,y1,z1,thexs,theys)
    z2g=gridify(x2,y2,z2,thexs,theys)
    # Extract eps, npts, and timing values at intersection of data sets
    x1r,y1r,z1r,z2r=trimgrid(thexs,theys,z1g,z2g)
    # Get log of epsilon, npts, and diming differences for plotting
    lx1,ly1,lzdiff=procdata(x1r,y1r,z1r,z2r)
    for ii in range(len(x1r)):
        print("%6.3f %7d %12.8f %12.8f" %(x1r[ii],y1r[ii],z1r[ii],z2r[ii]))
    title='BVH - Brute Performance Comparison'
    # Plot data
    data2file(x1r,y1r,z1r,z2r,run)
    makedataplot(lx1,ly1,lzdiff,run)
    return

available_profilers = ['nvidia', 'kp', 'none']
if len(sys.argv) == 2:
   profiler = sys.argv[1]
   if profiler not in available_profilers: 
      print("syntax: '$> %s <profiler>'; where <profiler> = %s" %(sys.argv[0],str(available_profilers)))
   else: 
      run='const'
      main(run,profiler)
      run='neigh'
      main(run,profiler)
      run='query'
      main(run,profiler)
else: print("syntax: '$> %s <profiler>'; where <profiler> = %s" %(sys.argv[0],str(available_profilers)))



