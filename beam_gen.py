# Gernerate beam for 6D OSIRIS input
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os.path
from scipy import constants
import warnings
from matplotlib.colors import LogNorm

class DistGen:
    def __init__(self, x_bins, counts):
        '''
        Gernerate 1D distribution based on a histogram.
        
        Parameters
        ----------
        x_bins : 1d numpy array of floats
            The positions of the bins. Should be in ascending order.
        counts : 1d numpy array of floats
            The counts in each of the bins.
        '''
        self.x_bins = np.array(x_bins)
        # Normalized to maximum of 1
        self.counts = np.array(counts)/np.max(counts)
        if len(self.x_bins) != len(self.counts): raise RuntimeError('The lengths of x_bins and counts are different!')

    def sample_gen_direct(self, n):
        '''
        Gernerate n samples directly.
        This method is not good if n is not much biger than length of self.x_bins.
        
        Parameters
        ----------
        n : int
            The number of samples.

        Return
        ----------
        x_array : 1d numpy array of floats
            The position of the samples.
        '''
        # The number in each bin
        # np.around will return the same type as input
        n_in_bins = np.around(n/np.sum(self.counts)*self.counts).astype(int)
        x_array = np.array([])
        for b_ind in range(len(self.x_bins)-1):
           # Generate and append a random array with length n_in_bins[b_ind] in the range [self.x_bins[b_ind], self.x_bins[b_ind+1])
           x_array = np.append(x_array, np.random.random_sample((n_in_bins[b_ind],))*(self.x_bins[b_ind+1]-self.x_bins[b_ind])+self.x_bins[b_ind])
        # Special dealing the last bin
        x_array = np.append(x_array, np.random.random_sample((n_in_bins[-1],))*(self.x_bins[-1]-self.x_bins[-2])+self.x_bins[-1])
        # Shuffle before return
        np.random.shuffle(x_array)
        return x_array

class BeamGen:
    def __init__(self, sig_x, sig_y, n_emit_x, n_emit_y, gamma0, sig_gamma, n_macroparticles, zf_x, zf_y, z_array, x0=0., y0=0., q_particle=-1):
        '''
        Gernerate 6D information of macroparicles for the OSIRIS 6D particle input..
        
        Parameters
        ----------
        ascii_file_name : a string
            The file name of saving the 6D information.
        sig_x : float (in meters)
            The transverse RMS bunch size in x direction (horizontal).
        sig_y : float (in meters)
            The transverse RMS bunch size in y direction (vertical).
        sig_z : float (in meters)
            The longitudinal RMS bunch size.
        n_emit_x : float (in meters)
            The normalized emittance of the bunch in x direction (horizontal).
        n_emit_y : float (in meters)
            The normalized emittance of the bunch in y direction (vertical).
        gamma0 : float
            The Lorentz factor of the electrons.
        sig_gamma : float
            The absolute energy spread of the bunch.
        n_macroparticles : int
            The number of macroparticles the bunch should consist of.
        zf_x : float (in meters)
            z position of the focus in x direction (horizontal).
        zf_y : float (in meters)
            z position of the focus in y direction (vertical).
        z_array : 1d numpy array of floats
            z Position of the sample particles. This can be generated by a <class DistGen> object.
        x0 : float (in meters), optional
            Beam center in x direction (horizontal).
        y0 : float (in meters), optional
            Beam center in y direction (vertical).
        q_particle : int, optional
            1 for positrons, and -1 for electrons.
        '''
        self.n_macroparticles = n_macroparticles
        self.gamma0 = gamma0
        self.sig_x = sig_x
        self.sig_y = sig_y
        self.n_emit_x = n_emit_x
        self.n_emit_y = n_emit_y
        self.sig_gamma = sig_gamma
        self.zf_x = zf_x
        self.zf_y = zf_y
        self.z_array = z_array
        self.x0=x0
        self.y0=y0
        self.q_particle=q_particle

    def get_file_name(self, value, file_ext='part'):
        '''If value is a existing folder, return 6D.<file_ext>, or 6D1.<file_ext> if the former already exits, or 6D2.<file_ext> and so on. If value is a existing file, add a number at the end of the file name (but before the extension).'''
        tmp_path = os.path.abspath(value)
        if os.path.isdir(tmp_path):
            i=1
            final_file = os.path.join(tmp_path, '6D.{}'.format(file_ext))
            while os.path.exists(final_file):
                final_file = os.path.join(tmp_path, '6D{}.{}'.format(i, file_ext))
                i+=1
            warnings.warn('Warning: \'{0}\' is a folder. Use \'{1}\' as the file name instead!'.format(tmp_path, final_file))
        elif os.path.isfile(tmp_path):
            root, ext = os.path.splitext(tmp_path)
            i=1
            final_file = '{0}{1}{2}'.format(root, i, ext)
            while os.path.exists(final_file):
                i+=1
                final_file = '{0}{1}{2}'.format(root, i, ext)
            warnings.warn('Warning: file \'{0}\' already exist. Use \'{1}\' instead!'.format(tmp_path, final_file))
        else:
            head, tail = os.path.split(tmp_path)
            if os.path.isdir(head):
                final_file = tmp_path
            else:
                raise IOError('Directory \'{0}\' do not exist!'.format(head))
        return final_file

################################property ascii_file_name################################
    def get_ascii_file_name(self):
        return self._ascii_file_name

    def set_ascii_file_name(self, value):
        self._ascii_file_name = self.get_file_name(value, file_ext='osi')

    ascii_file_name = property(get_ascii_file_name, set_ascii_file_name)

################################property h5_file_name################################
    def get_h5_file_name(self):
        return self._h5_file_name

    def set_h5_file_name(self, value):
        self._h5_file_name = self.get_file_name(value, file_ext='h5')

    h5_file_name = property(get_h5_file_name, set_h5_file_name)

################################property n_macroparticles################################
    def get_n_macroparticles(self):
        return self._n_macroparticles

    def set_n_macroparticles(self, value):
        if value<=0: raise ValueError('n_macroparticles = {} is not a positive number!'.format(value))
        value = int(value)
        # If self._n_macroparticles already exists, call the warning
        try:
            self._n_macroparticles
            warnings.warn("Setting n_macroparticles. You may have to reset all the beam parameters to refresh the sample. \n")
        except: pass
        self._n_macroparticles = value

    n_macroparticles = property(get_n_macroparticles, set_n_macroparticles)

################################property sig_gamma################################
    def get_sig_gamma(self):
        return self._sig_gamma

    def set_sig_gamma(self, value):
        if value<0:
            warnings.warn("Negative energy spread sig_gamma detected. sig_gamma will be set to zero. \n")
            self._sig_gamma = 0.
        else: self._sig_gamma = value

    sig_gamma = property(get_sig_gamma, set_sig_gamma)

################################method beam_gen################################
    def beam_gen(self, save_ascii_name = None, save_h5_name = None, sim_bound = None, nx = None, n0_per_cc = 1.e16, Q_beam = -1.e-12):
        '''
        Gernerate 6D information of macroparicles for the OSIRIS 6D particle input.
        Gernating method based on the method of FBPIC, at https://github.com/fbpic/fbpic/blob/dev/fbpic/lpa_utils/bunch.py (function add_particle_bunch_gaussian()).
        '''
        if self.sig_gamma > 0.: gamma = np.random.normal(self.gamma0, self.sig_gamma, self.n_macroparticles)
        else: gamma = np.full(self.n_macroparticles, self.gamma0)# Zero energy spread beam
        # Get Gaussian particle distribution in x,y
        self.x_array = self.sig_x * np.random.normal(0., 1., self.n_macroparticles)
        self.y_array = self.sig_y * np.random.normal(0., 1., self.n_macroparticles)

        # Define sigma of ux and uy based on normalized emittance
        sig_ux = (self.n_emit_x / self.sig_x)
        sig_uy = (self.n_emit_y / self.sig_y)
        # Get Gaussian distribution of transverse normalized momenta ux, uy
        self.ux_array = sig_ux * np.random.normal(0., 1., self.n_macroparticles)
        self.uy_array = sig_uy * np.random.normal(0., 1., self.n_macroparticles)

        # Finally we calculate the uz of each particle
        # from the gamma and the transverse momenta ux, uy
        uz_sqr = (gamma ** 2 - 1) - self.ux_array ** 2 - self.uy_array ** 2

        # Check for unphysical particles with uz**2 < 0
        mask = uz_sqr >= 0
        N_new = np.count_nonzero(mask)
        if N_new < self.n_macroparticles:
            warnings.warn(
              "Particles with uz**2<0 detected."
              " %d Particles will be removed from the beam. \n"
              "This will truncate the distribution of the beam"
              " at gamma ~= 1. \n"
              "However, the charge will be kept constant. \n"%(self.n_macroparticles
                                                               - N_new))
            # Remove unphysical particles with uz**2 < 0
            self.x_array = self.x_array[mask]
            self.y_array = self.y_array[mask]
            self.ux_array = self.ux_array[mask]
            self.uy_array = self.uy_array[mask]
            uz_sqr = uz_sqr[mask]
        if len(self.z_array)<N_new: raise RuntimeError('Sample z_array has fewer number than n_macroparticles! Set longer z_array.')
        elif len(self.z_array)>N_new: self.z_array = self.z_array[:N_new] # Truncate self.z_array to the length of uy_array
        # Calculate longitudinal momentum of the bunch
        self.uz_array = np.sqrt(uz_sqr)
        # Propagate distribution to an out-of-focus position tf.
        # (without taking space charge effects into account)
        distance_after_f_x = self.z_array - self.zf_x
        distance_after_f_y = self.z_array - self.zf_y
        self.x_array = self.x_array + self.ux_array / self.uz_array * distance_after_f_x
        self.y_array = self.y_array + self.uy_array / self.uz_array * distance_after_f_y
        if save_ascii_name is not None:
            self.ascii_file_name = save_ascii_name
            self.save_sample_ascii()
        if save_h5_name is not None:
            self.h5_file_name = save_h5_name
            self.save_sample_h5(sim_bound = sim_bound, nx = nx, n0_per_cc = n0_per_cc, Q_beam = Q_beam)

################################method beam_symmetrization################################
    def beam_symmetrization(self, ratio=1):
        '''
        Cylindrically symmetrization the beam.
        ratio is the ratio of particles to be symmetrized.
        '''
        N=len(self.x_array)
        R=int(N*ratio)
        # R should not be larger than N
        R = R if R<N else N
        if R>0:
            self.x_array = np.append(self.x_array[:R], -self.x_array[:R])
            self.y_array = np.append(self.y_array[:R], -self.y_array[:R])
            self.z_array = np.append(self.z_array[:R], self.z_array[:R])
            self.ux_array = np.append(self.ux_array[:R], -self.ux_array[:R])
            self.uy_array = np.append(self.uy_array[:R], -self.uy_array[:R])
            self.uz_array = np.append(self.uz_array[:R], self.uz_array[:R])

################################method save_sample_ascii################################
    def save_sample_ascii(self):
        '''
        Save self.z_array [m], self.x_array [m], self.y_array [m], self.uz_array [mc], self.ux_array [mc], self.uy_array [mc], self.q_particle to ascii file.
        '''
        with open(self.ascii_file_name, 'w') as fid:
            for i in range(len(self.z_array)):
                fid.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(self.z_array[i],self.x_array[i],self.y_array[i],self.uz_array[i],self.ux_array[i],self.uy_array[i],self.q_particle))

################################method save_sample_h5################################
    def save_sample_h5(self, sim_bound = None, nx = None, n0_per_cc = 1.e16, Q_beam = -1.e-12):
        '''
        Transform self.z_array [m], self.x_array [m], self.y_array [m], self.uz_array [mc], self.ux_array [mc], self.uy_array [mc], self.q_particle to normalized units and save to h5 file.
        sim_bound : simulation boundaries, [[x1min, x1max], [x2min, x2max], [x3min, x3max]]
        nx : number of cells in 3 dimensions, [nx1, nx2, nx3]
        n0_per_cc : density for normalization, in unit of per cc.
        Q_beam : beam charge, in unit of Coulomb
        '''
        with h5py.File(self.h5_file_name, 'w') as fid:
            sim_bound = np.transpose(sim_bound)
            fid.attrs.create('TIME', (0.,))
            fid.attrs.create('XMIN', sim_bound[0])
            fid.attrs.create('XMAX', sim_bound[1])
            fid.attrs.create('NX', nx)
            # k0 in unit of m^-1
            k0 = np.sqrt(4*constants.pi*constants.physical_constants['classical electron radius'][0]*n0_per_cc*1e6)
            fid.create_dataset("x1", data=k0*self.z_array)
            fid.create_dataset("x2", data=k0*self.x_array)
            fid.create_dataset("x3", data=k0*self.y_array)
            fid.create_dataset("p1", data=self.uz_array)
            fid.create_dataset("p2", data=self.ux_array)
            fid.create_dataset("p3", data=self.uy_array)
            n_part = len(self.x_array)
            cell_volume_norm = 1.
            for i in range(3): cell_volume_norm *= (sim_bound[1,i]-sim_bound[0,i])/nx[i]
            q = Q_beam/n_part/constants.elementary_charge/cell_volume_norm*k0**3/n0_per_cc/1.e6
            fid.create_dataset("q", data=np.full(n_part, q))

################################method plot_hist2D################################
    def plot_hist2D(self, xaxis='z', yaxis='x', bins=64):
        '''
        plot 2D histogram.
        '''
        axis_dic={'z':self.z_array, 'x':self.x_array, 'y':self.y_array, 'pz':self.uz_array, 'px':self.ux_array, 'py':self.uy_array}
        H, xedges, yedges = np.histogram2d(axis_dic[xaxis], axis_dic[yaxis], bins=bins)
        plt.figure()
        plt.pcolormesh(xedges[:-1], yedges[:-1], np.transpose(H), norm=LogNorm())
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.tight_layout()

if __name__ == '__main__':
    current_filename = './ZeroCrossing1.txt'
    data = np.loadtxt(current_filename,delimiter=',')
    z = (data[0,:]*constants.c*1.e-9-26710.)*1e-6 # Original unit fs; transform to m; shift to simulation box position
    current = data[1,:]
    plt.plot(z,current,'k-')
    samp_generator = DistGen(z, current)
    sample_z = samp_generator.sample_gen_direct(1e5)
    print(len(sample_z))
    hist, z_bins = np.histogram(sample_z, bins = 128)
    hist = hist*(np.max(current)/hist.max())
    plt.plot(z_bins[:-1], hist, 'r-')
    beam_generator = BeamGen(sig_x=19.6e-6, sig_y=17.1e-6, n_emit_x=14.04e-6, n_emit_y=5.32e-6, gamma0=2184., sig_gamma=10., n_macroparticles=1e5, zf_x=2.9e-2, zf_y=-5e-3, z_array=sample_z)
    beam_generator.beam_gen(save_h5_name='./', sim_bound = [[0.,10.], [-4., 4.], [-4., 4.]], nx = [512, 256, 256], n0_per_cc = 1.e16, Q_beam = -300.e-12)
    beam_generator.plot_hist2D(xaxis='z', yaxis='x')
    plt.show()
