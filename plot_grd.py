import numpy as np
import matplotlib.pyplot as plt
from numpy import load
import json
import glob
import numpy.ma as ma


with open('data.txt') as json_file:
	data = json.load(json_file)


def get_f(dat):
   f=[]
   for freq in dat:
      if freq != 'cfreq':
         a='{0:.1f}'.format(float(freq))
         f.append(a)
   return f


#Read data
evid='*1160214001*.npz'
fid=glob.glob(evid)



#plt.figure(figsize=(10,5))
fig, axs = plt.subplots(2, 4,figsize=(10,5))
plt.subplots_adjust(wspace=0.05,hspace=0.19,left=0.07,bottom=0.09,right=0.93,top=0.91)

i=0
k=0

#velocity for g0
v0=3400
for f in fid:
   #read npz files
   data=np.load(f,allow_pickle=True)
   x=data['arr_0'][2][0]
   y=data['arr_0'][2][1]
   z=data['arr_0'][3]

   zm=ma.masked_where(np.isnan(z),z)
   zi=ma.masked_where(np.isinf(zm),zm)
   index=np.unravel_index(np.argmin(zi), zi.shape)
   Qsc=10**x[index]
   Qi=10**y[index]
   cfreq=float(f.split('_')[2].split('H')[0])
   b=2*np.pi*cfreq*Qi
   g0=(2*np.pi*cfreq*Qsc)/(v0)	
   
   
   if i == 4:
      i=0
      k=1

   ax=axs[k,i]
   ax.set_title(f.split('_')[2])
   pcm=ax.pcolormesh(x,y,z,vmin=0,vmax=2)
   ax.scatter(np.log10(Qsc),np.log10(Qi),c='r',s=3.0)

   if i == 0 and k == 0:
      ax.set_ylabel('log$_{10}$(b)')
   if k == 1 and i == 1:
      ax.set_xlabel('B0')
   if i != 0:
      ax.tick_params(labelleft=False)  
   if k != 1:
      ax.tick_params(labelbottom=False)
   i=i+1



cbar=plt.colorbar(pcm,ax=axs[:,3])
cbar.set_label('residual')
plt.savefig('obj_func_'+evid+'.png',dpi=600)
plt.show()




exit()








