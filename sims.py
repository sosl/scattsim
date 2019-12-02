import numpy
import argparse
from os import remove

from sims_lib import *
import psrchive as psr

parser = argparse.ArgumentParser(description='''
        This code will produce simulated data based on an input template.
        It can add various types of propagation effects, such as scattering
        by a variety of screens and extra dispersion measure. Three types of
        output are produced:
            -- template with white noise only
            -- template with white noise and chosen scattering
            -- template with white noise, scattering, and random DM change
        ''')

parser.add_argument('-t', '--template', dest='template', nargs=1,
        help="Load template archive from this file")
parser.add_argument('--smooth', dest='smooth', action='store_true',
        help="Smooth the template with a Savitzky-Golay filter (17, 5).")
parser.add_argument('-d', '--data', dest='data', nargs=1,
        help="Load the data to be used as input from this file")
parser.add_argument('-O', '--output-path', dest='outpath', default = "./",
        nargs=1, help="Unload the simulated data in this directory")
parser.add_argument('-p', '--postfix', dest='postfix', nargs=1,
        help="Use this postfix in the naming of output files")
parser.add_argument('-s', '--screen', dest='screen_type', nargs=1,
        help="Use this type of screen to scatter data.\n"
             "Possible values:\n"
             "  thin        - thin screen\n"
             "  thick       - thick screen\n"
             "  extended    - extended (uniform) screen\n"
             "  anisotropic - anisotropic screen\n\n"
             "To use a truncated screen add '_truncated'.")
parser.add_argument('--tau-mean',  dest='tau_means', nargs='+', type=float,
        help="The mean scattering time. For anisotropic screen provide 2 values")
parser.add_argument('--tau-std', dest='tau_stds', nargs='+', type=float,
        help="The std dev of scattering time. For anisotropic screen provide 2 values")
parser.add_argument('--DM-mean',  dest='DM_mean', nargs='+', type=float,
        default=0., help="The mean DM change.")
parser.add_argument('--DM-std', dest='DM_std', nargs='+', type=float,
        default=0., help="The std dev of DM change.")
parser.add_argument('--noise-std', dest='noise_std', nargs=1, type=float,
        help="The white noise rms.")
parser.add_argument('--cutoff-mean',  dest='cutoff_mean', nargs=1, type=float,
        help="The mean cutoff value.")
parser.add_argument('--cutoff-std', dest='cutoff_std', nargs=1, type=float,
        help="The std dev of cutoff value.")

args = parser.parse_args()

#load template
template = getTemplateFromData(args.template[0], smooth=args.smooth)

# Load in archive to replace with simulated profiles
# profile: subint, pol, chan
data = psr.Archive_load(args.data[0])
desired_channels = 4

data.remove_baseline()
data.fscrunch_to_nchan(desired_channels)
data.pscrunch()
# Ensure we are producing not-dedispersed data suitable for timing. 
data.dedisperse()
nbin = data.get_nbin()
xbins = numpy.linspace(0, nbin-1, nbin)
noise_only = data.clone()
scattered_thin = data.clone()
scattered_thin_DM = data.clone() # see Util/fft/shift.C or use rotate_phase and convert time to phase via folding period

nsubint = data.get_nsubint()

tau = []
for i in xrange(len(args.tau_means)):
    tau.append(numpy.random.normal(args.tau_means[i], args.tau_stds[i], nsubint))

if args.cutoff_mean and args.cutoff_std:
    cutoff = numpy.random.normal(args.cutoff_mean, args.cutoff_std, nsubint)

# DM offsets:
DMconst = 4148.808 # to get delays in s, see handbook eq. 4.7
DM = numpy.random.normal(args.DM_mean, args.DM_std, nsubint)

# basic preparation of the archive

noise_std= args.noise_std
# scattering times and stddev at 1 GHz, in bins:
if len(args.tau_means) != len(args.tau_stds):
    print 'You must provide as many tau-std as tau-mean'
    exit(-1)

# Store the injected tau and DM, tau in units of ms at 1 GHz, DM in natural units
inject_data = numpy.c_[tau[0] / nbin, DM]# * data.get_first_Integration().get_folding_period()*1000, DM]
if "anisotropic" in args.screen_type[0]:
    inject_data = numpy.c_[inject_data, tau[1] / nbin]
if "truncated" in args.screen_type[0]:
    inject_data = numpy.c_[inject_data, cutoff / nbin]

#save the input values used to generate data
try:
    remove(args.outpath[0] + "/Injected_Tau_" + args.postfix[0] + ".txt")
except OSError:
    pass

#loop through the subints and apply the requested filtering
for subint in xrange(nsubint):
    period = data.get_Integration(subint).get_folding_period()
    # convert tau_1 to ms:
    inject_data[subint][0] *= period*1000.
    # if present, convert tau_2 as well:
    if "anisotropic" in args.screen_type[0]:
        inject_data[subint][2] *= period*1000.
    # if present, convert cutoff as well:
    if "truncated" in args.screen_type[0]:
        inject_data[subint][-1] *= period*1000.
    # loop through channels:
    for chan in xrange(data.get_nchan()):
        prof_noise_only = noise_only.get_Profile(subint, 0, chan)
        prof_scattered = scattered_thin.get_Profile(subint, 0, chan)
        prof_scattered_DM = scattered_thin_DM.get_Profile(subint, 0, chan)
        
        cfreq = prof_noise_only.get_centre_frequency()
        filter_g = linspace(0, nbin-1, nbin)
        if args.screen_type[0] == "thin":
            filter_g = infiniteScreenFilter(xbins, tau[0][subint]*pow(cfreq/1000., -4))
        elif args.screen_type[0] == "thin_truncated":
            filter_g = infiniteScreenFilterTruncated(xbins, tau[0][subint]*pow(cfreq/1000., -4), cutoff[subint])
        elif args.screen_type[0] == "thick":
            filter_g = thickScreenNear(xbins, tau[0][subint]*pow(cfreq/1000., -4))
        elif args.screen_type[0] == "extended":
            filter_g = extendedScreen(xbins, tau[0][subint]*pow(cfreq/1000., -4))
        elif args.screen_type[0] == "anisotropic":
            filter_g = anisotropicScreen(xbins, tau[0][subint]*pow(cfreq/1000., -4),
                    tau[1][subint]*pow(cfreq/1000., -4))
        elif args.screen_type[0] == "anisotropic_truncated":
            filter_g = anisotropicScreenTruncated(xbins, tau[0][subint]*pow(cfreq/1000., -4),
                                                  tau[1][subint]*pow(cfreq/1000., -4), cutoff[subint])
        else:
            raise Error
        
        noise = numpy.random.normal(0, args.noise_std, 1024)
        amps_noise = prof_noise_only.get_amps()
        amps_scattered = prof_scattered.get_amps()
        amps_scattered_DM = prof_scattered_DM.get_amps()

        scattered = scatterProfile(template, filter_g)
        for i in xrange(nbin):
            amps_noise[i] = template[i] + noise[i]
            amps_scattered[i] = scattered[i] + noise[i]
            amps_scattered_DM[i] = amps_scattered[i]
        prof_scattered_DM.rotate_phase(DM[subint] * DMconst / period * pow(cfreq, -2))

with open(args.outpath[0] + "/Injected_Tau_" + args.postfix[0] + ".txt", "ab") as fh:
    fh.write("# template   : " + args.template[0] + "\n")
    fh.write("# smoothed?  : " + str(args.smooth) + "\n")
    fh.write("# data       : " + args.data[0] + "\n") 
    fh.write("# postfix    : " + args.postfix[0] + "\n")
    fh.write("# screen     : " + args.screen_type[0] + "\n")
    fh.write("# noise rms  : " + " ".join(map(str, args.noise_std)) + "\n")
    fh.write("# tau means  : " + " ".join(map(str, args.tau_means)) + "\n")
    fh.write("# tau stds   : " + " ".join(map(str, args.tau_stds)) + "\n")
    fh.write("# DM mean    : " + " ".join(map(str, args.DM_mean)) + "\n")
    fh.write("# DM std     : " + " ".join(map(str, args.DM_std)) + "\n")
    if args.cutoff_mean:
        fh.write("# C mean     : " + " ".join(map(str, args.cutoff_mean)) + "\n")
    if args.cutoff_std:
        fh.write("# C std      : " + " ".join(map(str, args.cutoff_std)) + "\n")
    numpy.savetxt(fh, inject_data)

# ### unload the data

noise_only.dededisperse()
noise_only.unload(args.outpath[0] + "/noise_" + args.postfix[0] + ".ar")
scattered_thin.dededisperse()
scattered_thin.unload(args.outpath[0] + "/scattered_" + args.postfix[0] + ".ar")
scattered_thin_DM.dededisperse()
scattered_thin_DM.unload(args.outpath[0] + "/scattered_" + args.postfix[0] + "_DM.ar")

