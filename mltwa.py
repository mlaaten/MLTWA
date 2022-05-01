from obspy import read, read_inventory, read_events
from obspy.core import UTCDateTime as UTC
import obspy.signal
import numpy as np
import json
import obspy
import scipy
from obspy.geodetics import gps2dist_azimuth
import matplotlib.pyplot as plt
from scipy import optimize
import multiprocessing as mp


#Read data
st=read('waveform/*.gse')
inv=read_inventory('../inventory/*.xml')
evt=read_events('evt/*.evt',format='EVT',inventory=inv)


# Frequency bands:
# step - difference between neighboring central frequencies in octave
# (x octave corresponds to a factor of 2**x between the frequencies)
# width - difference between higher and lower corner frequency in octave
# max - maximal central frequency,
# min - minimal possible central frequency.
# cfreqs - list of central frequencies (min, max and step are ignored)
# fbands - list frequency bands (all other entries are ignored)

#cfreqs={"width": 1, "cfreqs": [3,4.2,6,8.5]}
cfreqs={"fbands": [[1, 2], [2, 4], [4, 8]]}
#cfreqs={"step": 0.5, "width": 1.8, "max": 34, "min": 3}

#maximum distance earthquake - station in meters
dist_limit=90000
#mean density
rho=2700
#free surface correction
fs=4
#window length relative to origin (must contain SNR window!)
twin=(-10,100)
#signal-noise-ratio
snr=2
#SNR window relative to origin
snrwin=(-4,0)
#minimum window length relative to S-pick
twin_min=75
#MLTWA windows relative to S-pick
mltwwin=((0,15),(15,30),(30,45))
#coda normalization window relativ to S-pick
codanormwin=(60,65)
#minimum number of stations for an event
min_stat=4
#Number of samples of the synthetic data 
n_syn=101
#weighting factors of the three time windows
w__=[0.5,1,1]
#bounds for optimization of Qsc and Qi in log(val)
inv_range=([-5, -3], [-4.0, -2.0]) #Qsc Qi
#Number of inversion grid points
n_inv=10
#number of cores for parallel computing
n_kern=4


def get_stream(pick,station,st):
    """Return stream for pick pick"""
    stream=obspy.core.stream.Stream()
    st_station=st.select(station=station)
    for trace in st_station:
        tr_start=UTC(trace.stats.starttime)
        tr_end=UTC(trace.stats.endtime)
        if UTC(pick) >= tr_start and UTC(pick)<=tr_end:
            stream=stream+trace
    return stream

def energy1c(data, rho, df, fs=4):
    """Spectral energy density of one channel"""
    hilb = scipy.fftpack.hilbert(data)
    return rho * (data ** 2 + hilb ** 2) / 2 / df / fs

def get_freqs(max=None, min=None, step=None, width=None, cfreqs=None,
              fbands=None):
    """Determine frequency bands"""
    if cfreqs is None and fbands is None:
        max_exp = int(np.log(max / min) / step / np.log(2))
        exponents = step * np.arange(max_exp + 1)[::-1]
        cfreqs = max / 2 ** exponents
    if fbands is None:
        df = np.array(cfreqs) * (2 ** width - 1) / (2 ** width + 1)
        fbands = [(f, (f - d, f + d)) for d, f in zip(df, cfreqs)]
    else:
        fbands = sorted(fbands)
        cfreqs = [0.5 * (f1 + f2) for f1, f2 in fbands]
        fbands = [(0.5 * (f1 + f2), (f1, f2)) for f1, f2 in fbands]
    return fbands

def get_f(dat):
   f=[]
   for freq in dat:
      if freq != 'cfreq':
         f.append(freq)
   return f


def get_data(freq,data,evid):
   e, c, t1, t2, r, w=[], [], [], [], [], []
   for event in data[freq]:
      if event == evid:
         for station in data[freq][event]:
            dist=station[2]
            t=station[3]
            vel=dist/station[3]
            c_=[vel,vel,vel]
            w_=w__
            t1_=[t+mltwwin[0][0],t+mltwwin[1][0],t+mltwwin[2][0]]
            t2_=[t+mltwwin[0][1],t+mltwwin[1][1],t+mltwwin[2][1]]
            r_=[dist,dist,dist]
            e.append(station[1])
            c.append(c_)
            t1.append(t1_)
            t2.append(t2_)
            r.append(r_)
            w.append(w_)
   w, r, c = np.ravel(w), np.ravel(r), np.ravel(c),
   t1, t2, Eobs = np.ravel(t1), np.ravel(t2), np.ravel(e)
   return Eobs, t1, t2, r, c, w

def rt3d_direct(t, c, g0, var='t'):
    t1 = np.exp(-c * t * g0)
    t2 = (4 * np.pi * c ** 2 * t ** 2)
    return t1 / t2

def _F(x):
    return np.sqrt(1 + 2.026 / x)

def rt3d_coda_reduced(r, t):
    # Coda term for r<t in reduced variables r'=rg0, t'=tg0c  (3d)
    a = 1 - r ** 2 / t ** 2
    t1 = a ** 0.125 / (4 * np.pi * t / 3) ** 1.5
    t2 = np.exp(t * (a ** 0.75 - 1)) * _F(t * a ** 0.75)
    return t1 * t2

def rt3d_coda(r, t, c, g0):
    return rt3d_coda_reduced(r * g0, t * c * g0) * g0 ** 3

def G(r, t, c, g0, type='rt3d', include_direct=True):
    """Full Green's function with direct wave term (optional)"""
    Gcoda = rt3d_coda
    Gdirect = rt3d_direct
   
    t_isarray = isinstance(t, np.ndarray)
    r_isarray = isinstance(r, np.ndarray)
    if not t_isarray and not r_isarray:
        if t - r / c < 0:
            G_ = 0
        else:
            G_ = Gcoda(r, t, c, g0)
    elif t_isarray and r_isarray:
        if len(t) != len(r):
            msg = ('If t and r are numpy arrays,'
                   'they need to have the same length')
            raise ValueError(msg)
        if include_direct:
            msg = 'If t and r are numpy arrays, include_direct not supported'
            raise NotImplementedError(msg)
        G_ = np.zeros(np.shape(t))
        ind = c * t - r >= 0
        G_[ind] = Gcoda(r[ind], t[ind], c, g0)
    elif t_isarray:
        G_ = np.zeros(len(t))
        eps = float(t[1] - t[0])
        i = np.count_nonzero(c * t - r < 0)
        G_[i+1:] = Gcoda(r, t[i+1:], c, g0)
        if include_direct and 0 < i < len(G_):
            # factor 1 / c due to conversion of Dirac delta from
            # delta(r - c * t) to delta(t - r / c)
            G_[i] = Gdirect(r / c, c, g0) / eps / c
    elif r_isarray:
        G_ = np.zeros(len(r))
        eps = float(r[1] - r[0])
        i = -np.count_nonzero(c * t - r < 0)
        if i == 0:
            i = len(r)
        G_[:i] = Gcoda(r[:i], t, c, g0)
        if include_direct and i != len(G_):
            G_[i] = Gdirect(t, c, g0) / eps
    return G_




#Select stations and picks
sta_pick=[]
for event in evt:
    arrivals=event.origins[0].arrivals
    evid=str(event.resource_id).split('/')[-1]
    origin=event.origins[0].time
    evt_lat=event.origins[0].latitude
    evt_long=event.origins[0].longitude
    evt_depth=event.origins[0].depth
    for arrival in arrivals: 
        p_id=arrival.pick_id
        phase=arrival.phase
        if phase == 'Sg':
            p=arrival.pick_id.get_referred_object()
            pick_t=p.time
            sta=p.waveform_id.station_code
            try:
                sta_lat=inv.select(station=sta)[0][0].latitude
                sta_long=inv.select(station=sta)[0][0].longitude
                sta_depth=inv.select(station=sta)[0][0].elevation
            except Exception:
                continue
            hdist=gps2dist_azimuth(sta_lat, sta_long, evt_lat, evt_long)[0]
            vdist=(evt_depth+sta_depth)
            dist=np.sqrt(hdist ** 2 + vdist ** 2)
            stationpick=[evid, sta, pick_t, dist, origin]
            sta_pick.append(stationpick)


#Get frequencies
cfreqs_=get_freqs(**cfreqs)            

#Create empty dictonary
results={}	
for cfreq in cfreqs_:
    results[cfreq[0]]={}
    for pick in sta_pick:
        evid=pick[0]
        results[cfreq[0]][evid]=[]




#Read, processes data and prepare MLTWA input
energie=[]
distance=[]
for pick in sta_pick:
    st1=get_stream(pick[2],pick[1],st)
    if len(st1) > 0:
        for cfreq in cfreqs_:
            stream=st1.copy()
            station_q={}
            evid=pick[0]
            sta=pick[1]
            dist=pick[3]
            if dist >= dist_limit:
                print('Skip Station {} at Event {}, {} km distance is exceeded'.format(sta,evid,dist_limit))
                continue
            origin=pick[4]
            spick=pick[2]
            onset=spick-origin
            print('Read station {} at event {} at cfreq {}'.format(sta,evid,cfreq[0]))
        #Trim data
            stream_snr=stream.copy()
            stream.trim(origin+twin[0],origin+twin[1])
        #Preprocessing
            stream.detrend("linear")
            stream.taper(max_percentage=0.05, type="hann")
            fmin=cfreq[1][0]
            fmax=cfreq[1][1]
            df=fmax-fmin
            stream_filt=stream.filter('bandpass',freqmin=fmin,freqmax=fmax)

        #Data Meta stats
            npts=stream_filt[0].stats.npts
            samprate=stream_filt[0].stats.sampling_rate
            t=np.arange(0,npts/samprate,1/samprate)

        #Calc envelope and smooth
            data = [energy1c(tr.data, rho, df, fs=fs) for tr in stream_filt]
            trenv = np.sum(data, axis=0)
            
        #Determine window index
            win1_s=int((spick-stream_filt[0].stats.starttime+mltwwin[0][0])*samprate)
            win1_e=int((spick-stream_filt[0].stats.starttime+mltwwin[0][1])*samprate)
            win2_s=int((spick-stream_filt[0].stats.starttime+mltwwin[1][0])*samprate)
            win2_e=int((spick-stream_filt[0].stats.starttime+mltwwin[1][1])*samprate)
            win3_s=int((spick-stream_filt[0].stats.starttime+mltwwin[2][0])*samprate)
            win3_e=int((spick-stream_filt[0].stats.starttime+mltwwin[2][1])*samprate)

        #Calc SNR
            win_snr_s=int((origin-stream_filt[0].stats.starttime+snrwin[0])*samprate)
            win_snr_e=int((origin-stream_filt[0].stats.starttime+snrwin[1])*samprate)
            noise_level=np.mean(trenv[win_snr_s:win_snr_e])

        #Calc noise level
            i1=int((origin-stream_filt[0].stats.starttime+codanormwin[0])*samprate)
            i2=int((origin-stream_filt[0].stats.starttime+codanormwin[1])*samprate)
            data_level_cm=np.mean(trenv[i1:i2])
            data_level_e3=np.mean(trenv[win3_s:win3_e])


            data_snr=trenv[win1_s:len(trenv)]
            try:
                index = np.where(data_snr < snr * noise_level)[0][0]
            except IndexError:
                index = len(data_snr)
            if index == 0:
                index=1
            t_snr = spick + index * stream_filt[0].stats.delta - origin

            if t_snr <= twin_min:
            #if data_level_e3 <= snr*noise_level or data_level_cm <= snr*noise_level:
                print("Skip, short window")
                continue

        #Calc normalize by station and source
            data_norm=trenv / data_level_cm

            """
        #Plot time windows
            plt.plot(t,data_norm)
            plt.plot(t[win1_s:win1_e],data_norm[win1_s:win1_e],'r')
            plt.plot(t[win2_s:win2_e],data_norm[win2_s:win2_e],'g')
            plt.plot(t[win3_s:win3_e],data_norm[win3_s:win3_e],'b')
            plt.yscale('log')
            plt.show()
            """

        #Calc energie		
            e1=np.mean(data_norm[win1_s:win1_e])
            e2=np.mean(data_norm[win2_s:win2_e])
            e3=np.mean(data_norm[win3_s:win3_e])
 
            ene=[e1,e2,e3]
            ene_update=[sta,ene,dist,onset,str(spick)]
            results[cfreq[0]][evid].append(ene_update)

#Identify events with less than min_stat
del_evt=[]
for res in results:
    for re in results[res]:
        if len(results[res][re]) < min_stat:
            del_evt.append([res,re])
        
#Remove events with less than min_stat
if len(del_evt) > 0:
    for devt in del_evt:
        del results[devt[0]][devt[1]]


results['cfreq']=[]
for cfreq in cfreqs_:
    results['cfreq'].append(cfreq[0])
     
with open('data.txt','w') as outfile: 
	json.dump(results,outfile) 

data=results


def G__(r, t, c, g0, b):
   G_=G(r,t,c,g0) * np.exp(-b*t)
   return G_

def intG(tstart, tend, r, c, Qsc, Qi, f):
   sumGall=[]
   for t1, t2, r_, c_ in zip(tstart, tend, r, c,):
   
        f=float(f)
        Qsc_=10**Qsc
        Qi_=10**Qi

        b_=2*np.pi*f*Qi_
        g0_=(2*np.pi*f*Qsc_)/(c_)

        t_ = np.linspace(codanormwin[0], codanormwin[1], n_syn) 
        sum_=np.mean(G__(r_, t_,c_,g0_,b_))

        t = np.linspace(t1-0.000001, t2, n_syn)
        Gsum = np.mean(G__(r_,t,c_,g0_,b_))/sum_
        sumGall.append(Gsum)
   return sumGall


def invert(z, Eobs, t1, t2, r, c, w, f):
    Qsc, Qi = z
    def error(Qsc, Qi):
        resid = w*(np.log10(4*np.pi*r**2*Eobs) - np.log10(4*np.pi*r**2*intG(t1, t2, r, c, Qsc, Qi, f)))
        err = np.sum(resid**2)
        return err
    val=error(Qsc,Qi)
    return val

def inv(evid, freq):
    print(evid,freq)
    rranges = inv_range
    Eobs, t1, t2, r, c, w = get_data(float(freq),data,evid)
    params = Eobs, t1, t2, r, c, w, float(freq)
    print('Start inversion for '+evid+' at frequency: '+freq)
    resbrute = optimize.brute(invert, rranges, args=params, Ns=n_inv, full_output=True, finish=None)
    freq='{0:.1f}'.format(float(freq))
    np.savez('inv_grid_'+freq+'Hz_'+evid,resbrute,allow_pickle=True)

def multi_run_wrapper(args):
   return inv(*args)

cfreq=get_f(data)


events=[]
flist=[]
for f_ in cfreq:
    evids=data[f_].keys()
    if len(evids) > 0:
        for ev in evids:
            events.append(ev)
            flist.append(str(f_))




pool = mp.Pool(n_kern)
pool.starmap(inv, zip(events,flist))


exit()

"""
#Read MLTWA results
def get_mltwa(cfreq,v0):
	data=np.load(mltwa_dir+'/inv_grid_'+str(cfreq)+'Hz_'+str(evid)+'.npz',allow_pickle=True)
	x=data['arr_0'][2][0]
	y=data['arr_0'][2][1]
	z=data['arr_0'][3]
	zm=ma.masked_where(np.isnan(z),z)
	zi=ma.masked_where(np.isinf(zm),zm)
	index=np.unravel_index(np.argmin(zi), zi.shape)
	Qsc=10**x[index]
	Qi=10**y[index]
	b=2*np.pi*float(cfreq)*Qi
	g0=(2*np.pi*float(cfreq)*Qsc)/(v0)	
	return g0, b
"""


