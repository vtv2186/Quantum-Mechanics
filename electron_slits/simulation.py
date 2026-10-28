"""Nonrelativistic TDSE: length unit a=4 hbar/p, time unit m a²/hbar."""
from dataclasses import dataclass, asdict
import csv, json
from pathlib import Path
import numpy as np
from scipy.constants import hbar, m_e, e
from scipy.fft import fft2, ifft2, fft, ifft

@dataclass
class Config:
    voltage: float = 100.
    separation_nm: float = .982
    width_nm: float = .246
    dx: float = .25
    dt: float = .02
    steps: int = 1500
    closed: bool = False
    barrier: bool = True
    absorb: bool = True
    @property
    def length_m(self): return 4*hbar/np.sqrt(2*m_e*e*self.voltage)
    def validate(self):
        if not 0 < self.voltage <= 5000: raise ValueError('Require 0 < voltage <= 5000 V (nonrelativistic)')
        if self.dx <= 0 or self.dt <= 0 or self.steps < 1: raise ValueError('Positive grid and time parameters required')
        d,w=np.array([self.separation_nm,self.width_nm])*1e-9/self.length_m
        if w < 4*self.dx or d <= w or d+w >= 100 or 4*self.dx >= np.pi:
            raise ValueError('Resolve width with >=4 cells, width < separation, apertures inside domain, momentum below Nyquist')
        return d,w

def run(c=None, snapshot_every=100):
    c=c or Config(); d,w=c.validate(); dx=c.dx
    x=np.arange(-48,88,dx); y=np.arange(-64,64,dx)
    X,Y=x[:,None],y[None,:]
    psi=np.exp(-(X+22)**2/36-Y**2/1600+4j*X)
    psi/=np.sqrt(np.sum(abs(psi)**2)*dx**2)
    openings=abs(Y-d/2)<w/2
    if not c.closed: openings= openings | (abs(Y+d/2)<w/2)
    V=80*((abs(X)<.5)&~openings) if c.barrier else np.zeros_like(psi.real)
    W=3*(np.maximum((-X-36)/12,0)**4+np.maximum((X-70)/18,0)**4+np.maximum((abs(Y)-50)/14,0)**4)
    if not c.absorb: W=np.zeros_like(W)
    half=np.exp((-1j*V-W)*c.dt/2)
    kx=2*np.pi*np.fft.fftfreq(len(x),dx)[:,None]; ky=2*np.pi*np.fft.fftfreq(len(y),dx)[None,:]
    kinetic=np.exp(-.5j*(kx*kx+ky*ky)*c.dt)
    s=np.argmin(abs(x-60)); flux=np.zeros(len(y)); lost=0.; history=[]; snapshots=[]; times=[]
    for n in range(c.steps):
        before=np.sum(abs(psi)**2)*dx**2; psi*=half
        lost+=before-np.sum(abs(psi)**2)*dx**2
        psi=ifft2(fft2(psi)*kinetic)
        before=np.sum(abs(psi)**2)*dx**2; psi*=half
        lost+=before-np.sum(abs(psi)**2)*dx**2
        derivative=ifft(1j*kx*fft(psi,axis=0),axis=0)[s]
        flux+=np.imag(psi[s].conj()*derivative)*c.dt
        if n%snapshot_every==0 or n==c.steps-1:
            norm=np.sum(abs(psi)**2)*dx**2
            history.append(((n+1)*c.dt,norm,lost,norm+lost))
            snapshots.append(psi.astype(np.complex64)); times.append((n+1)*c.dt)
    positive=np.maximum(flux,0); transmission=positive.sum()*dx
    probability=positive/positive.sum() if transmission>1e-12 else np.zeros_like(positive)
    return dict(x=x,y=y,psi=psi,snapshots=np.array(snapshots),times=np.array(times),history=np.array(history),
                flux=flux,probability=probability,transmission=transmission,length_m=c.length_m,
                time_s=m_e*c.length_m**2/hbar,config=asdict(c))

def detections(result,count=1000,rate=100.,seed=7):
    """Conditional transmitted-electron Born samples; rate is observed counts/second."""
    if count<1 or rate<=0: raise ValueError('Positive count and rate required')
    if not np.isclose(result['probability'].sum(),1): raise ValueError('No screen arrival; propagate longer')
    rng=np.random.default_rng(seed); i=rng.choice(len(result['y']),count,p=result['probability'])
    dy=result['y'][1]-result['y'][0]
    positions=(result['y'][i]+rng.uniform(-dy/2,dy/2,count))*result['length_m']
    return np.cumsum(rng.exponential(1/rate,count)),positions

def save(result,directory='artifacts',count=1000,rate=100.):
    p=Path(directory); p.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p/'wavefunction.npz',**{k:v for k,v in result.items() if k!='config'})
    (p/'config.json').write_text(json.dumps(result['config'],indent=2))
    t,y=detections(result,count,rate)
    with (p/'detections.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['observation_time_s','screen_y_m']); writer.writerows(zip(t,y))
