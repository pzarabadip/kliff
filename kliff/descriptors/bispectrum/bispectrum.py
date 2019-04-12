import sys
import logging
import numpy as np
import kliff
from kliff.descriptors.descriptor import Descriptor
from kliff.descriptors.descriptor import generate_full_cutoff, generate_species_code
from kliff.neighbor import NeighborList
from kliff.error import InputError
from . import bs

logger = kliff.logger.get_logger(__name__)


class Bispectrum(Descriptor):
    """Bispectrum descriptor."""

    def __init__(self, cut_name, cut_values, hyperparams=None, normalize=True,
                 dtype=np.float32):
        super(Bispectrum, self).__init__(cut_name, cut_values, hyperparams,
                                         normalize, dtype)

        self.update_hyperparams(self.hyperparams)

        # init cdesc
        rfac0 = self.hyperparams['rfac0']
        jmax = self.hyperparams['jmax']
        diagonalstyle = self.hyperparams['diagonalstyle']
        use_shared_arrays = self.hyperparams['use_shared_arrays']
        rmin0 = self.hyperparams['rmin0']
        switch_flag = self.hyperparams['switch_flag']
        bzero_flag = self.hyperparams['bzero_flag']
        self._cdesc = bs.Bispectrum(rfac0, 2*jmax, diagonalstyle, use_shared_arrays,
                                    rmin0, switch_flag, bzero_flag)

        self._set_cutoff()
        self._set_hyperparams()

    def transform(self, conf, grad=False):

        # neighbor list
        infl_dist = max(self.cutoff.values())
        nei = NeighborList(conf, infl_dist, padding_need_neigh=False)

        coords = nei.coords
        image = nei.image
        species = np.asarray([self.species_code[i] for i in nei.species],
                             dtype=np.intc)
        numneigh, neighlist = nei.get_numneigh_and_neighlist_1D()

        Natoms = len(coords)
        Ncontrib = conf.get_number_of_atoms()
        Ndesc = self.get_size()

        grad = True
        if grad:
            zeta, dzeta_dr = self._cdesc.compute_B(
                coords, species, neighlist, numneigh, image, Natoms, Ncontrib, Ndesc)
            # reshape to 4D array
            dzeta_dr = dzeta_dr.reshape(Ncontrib, Ndesc, Ncontrib, 3)
        else:
            zeta = self._cdesc.compute_B(
                coords, species, neighlist, numneigh, image, Natoms, Ncontrib, Ndesc)
            dzeta_dr = None

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug(
                '\n'+'='*25 + 'descriptor values (no normalization)' + '='*25)
            logger.debug('\nconfiguration name: %s', conf.get_identifier())
            logger.debug('\natom id    descriptor values ...')
            for i, line in enumerate(zeta):
                s = '\n{}    '.format(i)
                for j in line:
                    s += '{:.15g} '.format(j)
                logger.debug(s)

        return zeta, dzeta_dr

    def update_hyperparams(self, params):
        """Update the hyperparameters based on the input at initialization.
        """
        default_hyperparams = {
            'jmax': 3,
            'rfac0': 0.99363,
            'diagonalstyle': 3,
            'use_shared_arrays': 0,
            'rmin0': 0,
            'switch_flag': 1,
            'bzero_flag': 0,
            'rcutfac': 0.5, }

        if params is not None:
            for key, value in params.items():
                if key not in default_hyperparams:
                    name = self.__class__.__name__
                    raise InputError(
                        'Hyperparameter "{}" not supported by descirptor "{}".'
                        .format(key, name))
                else:
                    default_hyperparams[key] = value
        self.hyperparams = default_hyperparams

    def _set_cutoff(self):

        # check cutoff support
        if self.cut_name not in ['cos', 'exp']:
            raise SupportError("Cutoff type `{}' unsupported.".format(self.cut_name))

        self.cutoff = generate_full_cutoff(self.cut_values)
        self.species_code = generate_species_code(self.cut_values)
        num_species = len(self.species_code)

        rcutsym = np.zeros([num_species, num_species], dtype=np.double)
        for si, i in self.species_code.items():
            for sj, j in self.species_code.items():
                rcutsym[i][j] = self.cutoff[si+'-'+sj]

        rcutfac = self.hyperparams['rcutfac']
        self._cdesc.set_cutoff(self.cut_name, rcutsym, rcutfac)

    def _set_hyperparams(self):

        weight = np.array([1., 1.], dtype=np.double)
        self._cdesc.set_weight(weight)

        radius = np.array([5., 5.], dtype=np.double)
        self._cdesc.set_radius(radius)

    def get_size(self):
        """Return the size of descriptor.
        """
        diagonal = self.hyperparams['diagonalstyle']
        twojmax = int(2*self.hyperparams['jmax'])

        N = 0
        for j1 in range(0, twojmax+1):
            if(diagonal == 2):
                N += 1
            elif(diagonal == 1):
                for j in range(0, min(twojmax, 2*j1)+1, 2):
                    N += 1
            elif(diagonal == 0):
                for j2 in range(0, j1+1):
                    for j in range(j1-j2, min(twojmax, j1+j2)+1, 2):
                        N += 1
            elif(diagonal == 3):
                for j2 in range(0, j1+1):
                    for j in range(j1-j2, min(twojmax, j1+j2)+1, 2):
                        if (j >= j1):
                            N += 1
        return N
