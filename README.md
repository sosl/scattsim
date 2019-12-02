# scattsim
Simulate scattering

Example usage:
python sims.py -t template.ar O output -s thin -d data.ar --tau-mean 2.5 --tau-std 0.5

Scattering is units of bins, nbin (nbin in sims_lib.py) and output nchan (desired_chans in sims.py) are hardcoded to 1024 and 4, respectively.

Importantly, also getTemplateFromData has hardcoded phase range for template, never meant to make this code public!
