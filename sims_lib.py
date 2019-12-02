# coding: utf-8



from astropy import convolution as astro_conv
from scipy.stats import vonmises, expon
from scipy import signal
from scipy import ndimage
from scipy.special import iv
from math import pi#, pow #this doesn't support elementwise operations
from numpy import zeros_like, linspace, roll, exp, sqrt, where
import psrchive as psr

# TODO make it possible to set nbin or get it from data
nbin = 1024

def sum_of_gaussians(sigma, amp, centre, nbin):
    ''' Produces a sum of Gaussians on a periodic domain.
    
        sigma, amp, centre are lists of equal length
        nbin is a positive integer
        
        returns two arrays, one of x values and one of the model'''
    kappa = []
    mu = []
    for i in range(len(sigma)):
        kappa.append(1.0 / (sigma[i]**2) * 2.35428 * 2.35428 * nbin**2 / (4 * pi**2))
        mu.append(centre[i]/nbin)
        
    x = linspace(0.0, 1.0, nbin)
    xbins = linspace(1, nbin, num=nbin)
    
    data = zeros_like(xbins)
    
    for i in range(len(sigma)):
        model = vonmises.pdf(x*2*pi-pi, kappa[i], loc=mu[i]*2*pi - pi, scale=1)
        model *= amp[i] / max(model)
        data += model
        
    return xbins, data

def filteredGaussians(sigma, amp, centre, nbin, _filter):
    '''
    params:'''
    xbins = linspace(1,nbin,nbin)
    return signal.convolve( sum_of_gaussians(sigma, amp, centre, nbin), _filter, mode='full') / sum(_filter)

def truncateScreen(xbins, _filter, cutoff):
    return where(xbins < cutoff, _filter, [0]*len(xbins))

def infiniteScreenFilter(xbins, tau):
    '''
    produces a PBF for a infinite screen halfway between obs and source
    g_{thin}=\exp(-t/\tau_{s}) (Rusul et al. 2012, original Cronyn 1970, Williamson 1972)
    Based on Cronyn, W. M. 1970, Science 1968, 1453
    params:
    '''
    return expon.pdf(xbins, scale=tau)

def infiniteScreenFilterTruncated(xbins, tau, cutoff):
    return truncateScreen(xbins, infiniteScreenFilter(xbins, tau), cutoff)


def thickScreenNear(xbins, tau):
    '''
    TODO: Is this screan near observer or source? Scalings are different, but are they ultimately just hidden away in tau?
    produces a PBF for a infinite screen halfway between obs and source
    g_{thin}=g_{thick}=(\frac{\pi\tau_{s}}{4t^3})^{1/2}\exp(\frac{-\pi^2\tau_{s}}{16t})
    (Rusul et al. 2012, original Cronyn 1970, Williamson 1972)
    params:
    '''
    xzero = xbins[0]
    xbins[0] = 1.0
    filter_g  = sqrt(pi * tau / 4. / pow(xbins, 3)) * exp(-pow(pi, 2) * tau / 16. / xbins )
    filter_g[0] = 0.0
    xbins[0] = xzero
    return filter_g

def extendedScreen(xbins, tau):
    '''
    A screen spreading uniformly between the observer and the source.
    g_{extend}=(\frac{\pi^5\tau_{s}^3}{8t^5})^{1/2}\exp(\frac{-\pi^2\tau_{s}}{4t}) (Rusul et al. 2012, originally, Williamson 1972)
    '''
    xzero = xbins[0]
    xbins[0] = 1.0
    filter_g = sqrt(pow(pi, 5) * pow(tau, 3) / 8. / pow(xbins, 5)) * exp(-pow(pi, 2) * tau / 4. / xbins )
    filter_g[0] = 0.0
    xbins[0] = xzero
    return filter_g

def anisotropicScreen(xbins, tau_x, tau_y):
    '''
    A screen with different scattering properties in two dimensions:
    g_{anisotropic} = \frac{1}{\sqrt{\tau_x \tau_y}} \exp\left(-t/2\left(\frac{1}{\tau_x}+\frac{1}{\tau_y}\right)\right) I\left(0, t/2\left(\frac{1}{\tau_x}-\frac{1}{\tau_y}\right)\right)
    where I is the modified Bessel function of the first kind.
    
    From Marisa Geyer's talk at PWG. She calculated it using Jim Cordes' eq. 9 from his 2001 paper "Anomalous radio-wave scattering..."
    '''
    exp_ = exp(-xbins/2. * (1. / tau_x + 1. / tau_y))
    Iv = iv(0, xbins/2. * ( 1. / tau_x - 1. / tau_y))
    scale =  1. / sqrt(tau_x * tau_y) 
    return scale * exp_ * Iv

def anisotropicScreenTruncated(xbins, tau_x, tau_y, cutoff):
    '''
    '''
    return truncateScreen(xbins, anisotropicScreen(xbins, tau_x, tau_y), cutoff)

def scatterProfileReturnAsIs(profile, filter_, mode, boundary):
    '''
    This function returns a profile convolved with the filter.
    It will return the result exactly as returned by the convolving function.
    This may be different than what is expected, e.g., for signal.convolve
    the length of output will be different than for the input
    '''
    if mode == 'signal':
        filtered = signal.convolve( data, filter_, mode='full') / sum(filter_)
    elif mode == 'astropy':
        filtered = astro_conv.convolve_fft(profile, roll(filter_,
            len(profile)/2), boundary=boundary)
    else:
        raise ValueError("Mode " + mode + " is not supported. Only 'astropy' and 'signal' are available")
    return filtered
        
def scatterProfile(profile, filter_, mode='astropy', boundary='wrap'):
    '''
    This function returns profile convolved with filter.
    It ensures that the returned profile is the same length as input data
    '''
    return scatterProfileReturnAsIs(profile, filter_, mode, boundary)[0:nbin] #range needed if mode is signal

def getTemplateFromData(input, smooth=False):
    ''' params:
    input: File used as a template
    smooth: Apply Savitzky-Golay filter to the data
    '''
    ar = psr.Archive_load(input)
    ar.remove_baseline()
    ar.pscrunch()
    ar.fscrunch()
    template = ar.get_data().squeeze()
    if smooth:
        from scipy.signal import savgol_filter
        template_sg = savgol_filter(template, 17, 5)
        # TODO:
        # J2051 specific 550 and 180 below is the off-pulse region I want to blank out
        # for simulations
        template_sg[:180] = 0.0
        template_sg[550:] = 0.0
        # ensure smooth join with the zeroed part
        template = savgol_filter(template_sg, 17, 5)

    return template


def scattered_gaussian_residual(params, data=None, errs=None):
    #prms = np.array([param.value for param in params.itervalues()])
    return (data - scattered_gaussian(params, nbin)) / errs

