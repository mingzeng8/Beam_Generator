import beam_gen
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from numpy.lib.scimath import sqrt

def piecewise_linear(x, xarr, yarr):
    '''
    Return piecewise linear function of x. The piecewise linear points are defined by xarr and yarr.
    xarr and yarr should have the same size. xarr should be ascending.
    x should be an ascending array.
    '''
    y = np.zeros_like(x)
    for i_start in range(len(x)):
        if x[i_start] >= xarr[0]: break
    # Pointer for xarr
    p_arr = 1
    for i in range(i_start, len(x)):
        while p_arr < len(xarr) and xarr[p_arr] <= x[i]:
            p_arr += 1
        if p_arr >= len(xarr): break
        y[i] = yarr[p_arr-1] + (yarr[p_arr] - yarr[p_arr-1])/(xarr[p_arr] - xarr[p_arr-1]) * (x[i] - xarr[p_arr-1])
    return y

def driver_profile(x):
    #n = np.piecewise(x, [x<5., x>12.8],\
    #                    [  0.,     0., lambda x: (1./(13.-x)-x+13.)/8.])
    # the last value is the default value, when no condition is satisfied
    z = np.array([0.0,0.2026,0.4052,0.6754,0.8779999999,0.878,0.8781,1.0807,1.4184,1.6885,1.9587,2.499,3.5121,5.1331,8.1049])
    fz= np.array([0.2667,0.1834,0.1364,0.1101,0.107711,0.107711,0.1078,0.115,0.1403,0.1674,0.1981,0.2646,0.3959,0.6089,1.0])
    z = np.flip(12.9-z)
    fz= np.flip(fz)
    return piecewise_linear(x, z, fz)

def trailer_profile(z):
    n = np.piecewise(z, [z<1.8558, z>2.8689],\
                        [      0.,       0., lambda z: 0.4+0.6/1.0131*(z-1.8558)])
    # the last value is the default value, when no condition is satisfied
    return(n)

def driver_gen():
    n0_per_cc = 5.0334e15
    N = int(1e6)
    z = np.linspace(5., 13., 128)
    current = driver_profile(z)
    samp_generator = beam_gen.DistGen(z, current)
    sample_z = samp_generator.sample_gen_direct(N*1.01)
    hist, z_bins = np.histogram(sample_z, bins = len(z))
    hist = hist*(np.max(current)/hist.max())
    plt.plot(z_bins[:-1], hist, 'k.-')

    k0 = np.sqrt(4*constants.pi*constants.physical_constants['classical electron radius'][0]*n0_per_cc*1e6)
    sample_z = sample_z/k0 # Transform to meter

    beam_generator = beam_gen.BeamGen(sig_x=6.e-6, sig_y=6.e-6, n_emit_x=20.e-6, n_emit_y=20.e-6, gamma0=19569.5, sig_gamma=10., n_macroparticles=N, zf_x=0., zf_y=0., z_array=sample_z)
    beam_generator.beam_gen()
    # varying beam size
    z_max = np.max(beam_generator.z_array)
    z_min = np.min(beam_generator.z_array)
    ratio = z_max - z_min
    #beam_generator.x_array *= (z_max - beam_generator.z_array)/ratio +1.
    #beam_generator.y_array *= (z_max - beam_generator.z_array)/ratio +1.
    beam_generator.h5_file_name = './driver.h5'
    beam_generator.save_sample_h5(sim_bound = [[0.,13.], [-6., 6.], [-6., 6.]], nx = [512, 512, 512], n0_per_cc = n0_per_cc, Q_beam = -6.e-9)
    beam_generator.plot_hist2D(xaxis='z', yaxis='x')
    plt.show()

def var_driver_gen():
    # Driver with varing emittance
    n0_per_cc = 5.0334e15
    N = int(2e7)
    z = np.linspace(5., 13., 128)
    current = driver_profile(z)
    samp_generator = beam_gen.DistGen(z, current)
    sample_z = samp_generator.sample_gen_direct(N*1.01)
    sample_z = sample_z[:N] # discard extra particles
    hist, z_bins = np.histogram(sample_z, bins = len(z))
    hist = hist*(np.max(current)/hist.max())
    plt.plot(z_bins[:-1], hist, 'k.-')

    n_emit0 = 20.e-6 # emittance at beam head in SI units
    # varying emittance
    z_max = sample_z.max()
    z_min = sample_z.min()
    #n_emit = ((z_max - sample_z)/(z_max - z_min)*99.+1.)*n_emit0
    z_trans = z_max - (z_max-z_min)*0.05
    n_emit = np.piecewise(sample_z, [sample_z<z_trans], [lambda sample_z:((z_trans - sample_z)/(z_trans - z_min)*99.+1.)*n_emit0, n_emit0])

    k0 = np.sqrt(4*constants.pi*constants.physical_constants['classical electron radius'][0]*n0_per_cc*1e6)
    gamma_beam = 19569.5
    # Matched beam size
    sig_r = sqrt(sqrt(2./gamma_beam)/k0)*np.sqrt(n_emit)

    sample_z = sample_z/k0 # Transform to meter

    beam_generator = beam_gen.BeamGen(sig_x=sig_r, sig_y=sig_r, n_emit_x=n_emit, n_emit_y=n_emit, gamma0=gamma_beam, sig_gamma=1., n_macroparticles=N, zf_x=0., zf_y=0., z_array=sample_z)
    beam_generator.beam_gen()
    #beam_generator.x_array *= (z_max - beam_generator.z_array)/ratio +1.
    #beam_generator.y_array *= (z_max - beam_generator.z_array)/ratio +1.
    #beam_generator.beam_symmetrization()
    beam_generator.h5_file_name = './driver.h5'
    #beam_generator.h5_file_name = './driver_symmetric.h5'
    beam_generator.save_sample_h5(sim_bound = [[0.,13.], [-6., 6.], [-6., 6.]], nx = [512, 512, 512], n0_per_cc = n0_per_cc, Q_beam = -6.e-9)
    beam_generator.plot_hist2D(xaxis='z', yaxis='x')
    plt.show()

def trailer_gen():
    n0_per_cc = 5.0334e15
    N = int(1e6)
    z = np.linspace(1.8, 2.9, 64)
    current = trailer_profile(z)
    samp_generator = beam_gen.DistGen(z, current)
    sample_z = samp_generator.sample_gen_direct(N)
    hist, z_bins = np.histogram(sample_z, bins = len(z))
    hist = hist*(np.max(current)/hist.max())
    plt.plot(z_bins[:-1], hist, 'r--')

    k0 = np.sqrt(4*constants.pi*constants.physical_constants['classical electron radius'][0]*n0_per_cc*1e6)
    sample_z = sample_z/k0 # Transform to meter

    beam_generator = beam_gen.BeamGen(sig_x=8.8-6, sig_y=8.8-6, n_emit_x=100.e-6, n_emit_y=100.e-6, gamma0=19569.5, sig_gamma=10., n_macroparticles=N, zf_x=0., zf_y=0., z_array=sample_z)
    beam_generator.beam_gen(save_h5_name='./trailer.h5', sim_bound = [[0.,13.], [-6., 6.], [-6., 6.]], nx = [512, 256, 256], n0_per_cc = n0_per_cc, Q_beam = -0.87e-9)
    beam_generator.plot_hist2D(xaxis='z', yaxis='x')
    plt.show()

if '__main__' == __name__:
    #driver_gen()
    var_driver_gen()
    #trailer_gen()
