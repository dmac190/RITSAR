##############################################################################
#                                                                            #
#  This library contains the basic signal processing functions to be used    #
#  with the PySAR module                                                     #
#                                                                            #
##############################################################################


import numpy as np
from numpy import pi, arccosh, sqrt, cos
from scipy.fftpack import fftshift, fft2, ifft2, fft, ifft
from scipy.signal import firwin, filtfilt, resample

#all FT's assumed to be centered at the origin
def ft(f, ax=-1):
    F = fftshift(fft(fftshift(f), axis = ax))
    
    return F
    
def ift(F, ax = -1):
    f = fftshift(ifft(fftshift(F), axis = ax))
    
    return f

def ft2(f, delta=1):
    F = fftshift(fft2(fftshift(f)))*delta**2
    
    return(F)

def ift2(F, delta=1):
    N = F.shape[0]
    f = fftshift(ifft2(fftshift(F)))*(delta*N)**2
    
    return(f)

def RECT(t,T):
    f = np.zeros(len(t))
    f[(t/T<0.5) & (t/T >-0.5)] = 1
    
    return f
    
def taylor(nsamples, S_L=43):
    xi = np.linspace(-0.5, 0.5, nsamples)
    A = 1.0/pi*arccosh(10**(S_L*1.0/20))
    n_bar = int(2*A**2+0.5)+1
    sigma_p = n_bar/sqrt(A**2+(n_bar-0.5)**2)
    
    #Compute F_m
    m = np.arange(1,n_bar)
    n = np.arange(1,n_bar)
    F_m = np.zeros(n_bar-1)
    for i in m:
        num = 1
        den = 1
        for j in n:
            num = num*\
            (-1)**(i+1)*(1-i**2*1.0/sigma_p**2/(\
                            A**2+(j-0.5)**2))
            if i!=j:
                den = den*(1-i**2*1.0/j**2)
            
        F_m[i-1] = num/den
    
    w = np.ones(nsamples)
    for i in m:
        w += F_m[i-1]*cos(2*pi*i*xi)
    
    w = w/w.max()          
    return(w)
    
def upsample(f,size):
    x_pad = size[1]-f.shape[1]
    y_pad = size[0]-f.shape[0]
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
        
    if np.mod(y_pad,2):
        y_off = 1
    else:
        y_off = 0
    
    F = ft2(f)
    F_pad = np.pad(F, ((y_pad/2,y_pad/2+y_off),(x_pad/2, x_pad/2+x_off)),
                   mode = 'constant')
    f_up = ift2(F_pad)
    
    return(f_up)
    
def upsample1D(f, size):
    x_pad = size-f.size
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
    
    F = ft(f)
    F_pad = np.pad(F, (x_pad/2, x_pad/2+x_off),
                   mode = 'constant')
    f_up = ift(F_pad)
    
    return(f_up)
    
def pad1D(f, size):
    x_pad = size-f.size
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
    
    
    f_pad = np.pad(f, (x_pad/2, x_pad/2+x_off),
                   mode = 'constant')
    
    return(f_pad)

def pad(f, size):
    x_pad = size[1]-f.shape[1]
    y_pad = size[0]-f.shape[0]
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
        
    if np.mod(y_pad,2):
        y_off = 1
    else:
        y_off = 0
    
    f_pad = np.pad(f, ((y_pad//2,y_pad//2+y_off),(x_pad//2, x_pad//2+x_off)),
                   mode = 'constant')
    
    return(f_pad)
    
def cart2sph(cart):
    x = np.array([cart[:,0]]).T
    y = np.array([cart[:,1]]).T
    z = np.array([cart[:,2]]).T
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    sph = np.hstack([azimuth, elevation, r])
    return sph
    
def sph2cart(sph):
    azimuth     = np.array([sph[:,0]]).T
    elevation   = np.array([sph[:,1]]).T
    r           = np.array([sph[:,2]]).T
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    cart = np.hstack([x,y,z])
    return cart
    
def decimate(x, q, n=None, axis=-1, beta = None, cutoff = 'nyq'):
    if not isinstance(q, int):
        raise TypeError("q must be an integer")
        
    if n == None:
        n = int(np.log2(x.shape[axis]))
        
    if x.shape[axis] < n:
        n = x.shape[axis]-1
    
    if beta == None:
        beta = 1.*n/8
    
    padlen = n/2
    
    if cutoff == 'nyq':
        eps = np.finfo(np.float).eps
        cutoff = 1.-eps
    
    window = ('kaiser', beta)
    a = 1.
    
    b = firwin(n,  cutoff/ q, window=window)
    y = filtfilt(b, [a], x, axis=axis, padlen = padlen)
    
    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, None, q)
    return y[sl]

def polyphase_interp (x, xp, yp, n_taps=15, n_phases=10000, cutoff = 0.95):
        
    # Compute input and output sample spacing
    dxp = np.diff(xp).min()
    dx = np.diff(x).min()
    
    # Assume uniformly spaced input
    #if dx > (1.001*np.diff(x)).max() or dx < (0.999*np.diff(x)).min():
    #        raise ValueError('Output sample spacing not uniform')
    
    # Input centered convolution - scale output sample spacing to 1
    offset = x.min()
    G = ((x-offset)/dx).astype(int)
    Gp = ((xp-offset)/dx)
    
    # Create prototype polyphase filter
    if not n_taps%2:
        raise ValueError('Filter should have odd number of taps')
    
    if dx > dxp:                    # Downsampling
        f_cutoff = cutoff           # Align filter nulls with output which has a sample spacing of 1 by definition
    else:                           # Upsampling
        f_cutoff = cutoff * dx/dxp  # Align filter nulls with input which has a normalized sample spacing of dxp/dx
    
    filt_proto = firwin(n_taps, f_cutoff, fs=2)
    #filter_length = n_taps + 1
    #locs = np.linspace(-filter_length/2, filter_length/2, n_taps)
    #filt_proto = np.sinc(locs/ds_factor) #Use sinc for prototype filter
    
    # Create polyphase filter
    filt_poly = resample(filt_proto, n_taps*n_phases)
    
    # Pad input for convolution
    pad_left = max(G[0] - int(np.floor(Gp[0] - (n_taps-1)/2)), 0)
    pad_right = max(int(np.ceil(Gp[-1] + (n_taps-1)/2)) - G[-1], 0)
    
    # Calculate output
    y_pad = np.zeros(x.size + pad_left + pad_right)
    
    for i in range(xp.size):
        V_current = yp[i]
        G_current = Gp[i] + pad_left
        G_left = G_current - (n_taps-1)/2
        G_start = int(np.ceil(G_left))
        G_right = G_current + (n_taps-1)/2
        G_end = int(np.floor(G_right))
        
        # Input samples may not be evenly spaced so comput a local scale factor
        if i < xp.size - 1:
            local_scale = Gp[i+1] - Gp[i]
        
        filt = filt_poly[int((G_start-G_left)*n_phases):int((G_end-G_left)*n_phases)+1:n_phases]*local_scale
        y_pad[G_start:G_end+1] += V_current*filt
      
    if pad_right > 0:
        return y_pad[pad_left:-pad_right]
    else:
        return y_pad[pad_left:]