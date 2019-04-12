import numpy as np
import sys
import logging
from collections import OrderedDict
import kliff
from kliff.descriptors.descriptor import Descriptor
from kliff.descriptors.descriptor import generate_full_cutoff, generate_species_code
from kliff.neighbor import NeighborList
from kliff.error import InputError, SupportError
from . import sf

logger = kliff.logger.get_logger(__name__)


class SymmetryFunction(Descriptor):
    """Atom-centered symmetry functions descriptor as discussed in [Behler2011]_.

    Parameters
    ----------
    cut_name: str
        Name of the cutoff, such as ``cos``, ``exp``.

    cut_values: dict
        Values for the cutoff, with key of the form ``A-B`` where ``A`` and ``B``
        are atomic species, and value should be a float.

    hyperparams: dict
        A dictionary of the hyperparams of the descriptor.

    normalize: bool (optional)
        If ``True``, the fingerprints is centered and normalized according to:
        ``zeta = (zeta - mean(zeta)) / stdev(zeta)``

    dtype: np.dtype (optional)
        Data type for the generated fingerprints, such as ``np.float32`` and
        ``np.float64``.

    Example
    -------
    >>> cut_name = 'cos'
    >>> cut_values = {'C-C': 3.5, 'C-H': 3.0, 'H-H': 1.0}
    >>> hyperparams = {'g1': None,
    >>>                'g2': [{'eta':0.1, 'Rs':0.2}, {'eta':0.3, 'Rs':0.4}],
    >>>                'g3': [{'kappa':0.1}, {'kappa':0.2}, {'kappa':0.3}]}
    >>> desc = SymmetryFunction(cut_name, cut_values, hyperparams)

    References
    ----------
    .. [Behler2011] J. Behler, "Atom-centered symmetry functions for constructing
       high-dimensional neural network potentials," J. Chem. Phys. 134, 074106
       (2011).
    """

    def __init__(self, cut_name, cut_values, hyperparams, normalize=True,
                 dtype=np.float32):
        super(SymmetryFunction, self).__init__(cut_name, cut_values, hyperparams,
                                               normalize, dtype)

        self._desc = OrderedDict()

        self._cdesc = sf.Descriptor()
        self._set_cutoff()
        self._set_hyperparams()

        logger.info('"SymmetryFunction" descriptor initialized.')

    def transform(self, conf, grad=False):
        """Transform atomic coords to atomic enviroment descriptor values.

        Parameters
        ----------
        conf: :class:`~kliff.dataset.Configuration` object
            A configuration of atoms.


        grad: bool (optional)
            Whether to compute the gradient of descriptor values w.r.t. atomic
            coordinates.

        Returns
        -------
        zeta: 2D array
            Descriptor values, each row for one atom.
            zeta has shape (num_atoms, num_descriptors), where num_atoms is the
            number of atoms in the configuration, and num_descriptors is the size
            of the descriptor vector (depending on the the choice of hyper-parameters).

        dzeta_dr: 4D array if grad is ``True``, otherwise ``None``
            Gradient of descriptor values w.r.t. atomic coordinates.
            dzeta_dr has shape (num_atoms, num_descriptors, num_atoms, DIM), where
            num_atoms and num_descriptors has the same meanings as described in zeta.
            DIM = 3 denotes three Cartesian coordinates.
        """

        # create neighbor list
        infl_dist = max(self.cutoff.values())
        nei = NeighborList(conf, infl_dist, padding_need_neigh=False)

        coords = nei.coords
        image = nei.image
        species = np.asarray([self.species_code[i] for i in nei.species],
                             dtype=np.intc)
        numneigh, neighlist = nei.get_numneigh_and_neighlist_1D()

        Natoms = len(coords)
        Ncontrib = conf.get_number_of_atoms()
        Ndesc = len(self)

        if grad:
            zeta, dzeta_dr = self._cdesc.get_gen_coords_and_deri(
                coords, species, neighlist, numneigh, image, Natoms, Ncontrib, Ndesc)
            # reshape 3D array to 4D array
            dzeta_dr = dzeta_dr.reshape(Ncontrib, Ndesc, Ncontrib, 3)
        else:
            zeta = self._cdesc.get_gen_coords(
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
        self._cdesc.set_cutoff(self.cut_name, rcutsym)

    def _set_hyperparams(self):

        # hyperparams of descriptors
        for key, values in self.hyperparams.items():
            if key.lower() not in ['g1', 'g2', 'g3', 'g4', 'g5']:
                raise SupportError(
                    "Symmetry function `{}' unsupported.".format(key))

            # g1 needs no hyperparams, put a placeholder
            name = key.lower()
            if name == 'g1':
                # it has no hyperparams, zeros([1,1]) for placeholder
                params = np.zeros([1, 1], dtype=np.double)
            else:
                rows = len(values)
                cols = len(values[0])
                params = np.zeros([rows, cols], dtype=np.double)
                for i, line in enumerate(values):
                    if name == 'g2':
                        params[i][0] = line['eta']
                        params[i][1] = line['Rs']
                    elif name == 'g3':
                        params[i][0] = line['kappa']
                    elif key == 'g4':
                        params[i][0] = line['zeta']
                        params[i][1] = line['lambda']
                        params[i][2] = line['eta']
                    elif key == 'g5':
                        params[i][0] = line['zeta']
                        params[i][1] = line['lambda']
                        params[i][2] = line['eta']

            # store cutoff values in both this python and cpp class
            self._desc[name] = params
            self._cdesc.add_descriptor(name, params)

    def __len__(self):
        N = 0
        for key in self._desc:
            N += len(self._desc[key])
        return N

    def get_size(self):
        return len(self)


class Set51(SymmetryFunction):
    """ Symmetry function descriptor with the hyperparameters from:
    Artrith and Behler, PRB, 85, 045439 (2012)

    Parameters
    ----------

    cutname: str
      cutoff function name, e.g. `cos`

    cutvalue: dict
      cutoff values based on species.

    Example
    -------
        cutvalue = {'C-C': 3.5, 'C-H': 3.0, 'H-H': 1.0}
    """

    def __init__(self, cutvalue, cutname='cos', *args, **kwargs):

        params = OrderedDict()

        params['g2'] = [
            {'eta': 0.001, 'Rs': 0.},
            {'eta': 0.01,   'Rs': 0.},
            {'eta': 0.02,   'Rs': 0.},
            {'eta': 0.035,  'Rs': 0.},
            {'eta': 0.06,   'Rs': 0.},
            {'eta': 0.1,    'Rs': 0.},
            {'eta': 0.2,    'Rs': 0.},
            {'eta': 0.4,    'Rs': 0.}
        ]

        params['g4'] = [
            {'zeta': 1,  'lambda': -1, 'eta': 0.0001},
            {'zeta': 1,  'lambda': 1,  'eta': 0.0001},
            {'zeta': 2,  'lambda': -1, 'eta': 0.0001},
            {'zeta': 2,  'lambda': 1,  'eta': 0.0001},
            {'zeta': 1,  'lambda': -1, 'eta': 0.003},
            {'zeta': 1,  'lambda': 1,  'eta': 0.003},
            {'zeta': 2,  'lambda': -1, 'eta': 0.003},
            {'zeta': 2,  'lambda': 1,  'eta': 0.003},
            {'zeta': 1,  'lambda': -1,  'eta': 0.008},
            {'zeta': 1,  'lambda': 1,  'eta': 0.008},
            {'zeta': 2,  'lambda': -1,  'eta': 0.008},
            {'zeta': 2,  'lambda': 1,  'eta': 0.008},
            {'zeta': 1,  'lambda': -1,  'eta': 0.015},
            {'zeta': 1,  'lambda': 1,  'eta': 0.015},
            {'zeta': 2,  'lambda': -1,  'eta': 0.015},
            {'zeta': 2,  'lambda': 1,  'eta': 0.015},
            {'zeta': 4,  'lambda': -1,  'eta': 0.015},
            {'zeta': 4,  'lambda': 1,  'eta': 0.015},
            {'zeta': 16,  'lambda': -1,  'eta': 0.015},
            {'zeta': 16,  'lambda': 1,  'eta': 0.015},
            {'zeta': 1,  'lambda': -1,  'eta': 0.025},
            {'zeta': 1,  'lambda': 1,  'eta': 0.025},
            {'zeta': 2,  'lambda': -1,  'eta': 0.025},
            {'zeta': 2,  'lambda': 1,  'eta': 0.025},
            {'zeta': 4,  'lambda': -1,  'eta': 0.025},
            {'zeta': 4,  'lambda': 1,  'eta': 0.025},
            {'zeta': 16,  'lambda': -1,  'eta': 0.025},
            {'zeta': 16,  'lambda': 1,  'eta': 0.025},
            {'zeta': 1,  'lambda': -1,  'eta': 0.045},
            {'zeta': 1,  'lambda': 1,  'eta': 0.045},
            {'zeta': 2,  'lambda': -1,  'eta': 0.045},
            {'zeta': 2,  'lambda': 1,  'eta': 0.045},
            {'zeta': 4,  'lambda': -1,  'eta': 0.045},
            {'zeta': 4,  'lambda': 1,  'eta': 0.045},
            {'zeta': 16,  'lambda': -1,  'eta': 0.045},
            {'zeta': 16,  'lambda': 1,  'eta': 0.045},
            {'zeta': 1,  'lambda': -1,  'eta': 0.08},
            {'zeta': 1,  'lambda': 1,  'eta': 0.08},
            {'zeta': 2,  'lambda': -1,  'eta': 0.08},
            {'zeta': 2,  'lambda': 1,  'eta': 0.08},
            {'zeta': 4,  'lambda': -1,  'eta': 0.08},
            {'zeta': 4,  'lambda': 1,  'eta': 0.08},
            # {'zeta':16,  'lambda':-1,  'eta':0.08 },
            {'zeta': 16,  'lambda': 1,  'eta': 0.08}
        ]

        # tranfer units from bohr to angstrom
        bhor2ang = 0.529177
        for key, values in params.items():
            for val in values:
                if key == 'g2':
                    val['eta'] /= bhor2ang**2
                elif key == 'g4':
                    val['eta'] /= bhor2ang**2

        super(Set51, self).__init__(cutname, cutvalue, params, *args, **kwargs)


class Set30(SymmetryFunction):
    """ Symmetry function descriptor with the hyperparameters from:
    Artrith and Behler, PRB, 85, 045439 (2012)

    Parameters
    ----------

    cutname: string
      cutoff function name, e.g. `cos`

    cutvalue: dict
      cutoff values based on species.

      Example
      -------
      cutvalue = {'C-C': 3.5, 'C-H': 3.0, 'H-H': 1.0}
    """

    def __init__(self, cutvalue, cutname='cos'):

        params = OrderedDict()

        params['g2'] = [
            {'eta': 0.0009, 'Rs': 0.},
            {'eta': 0.01,   'Rs': 0.},
            {'eta': 0.02,   'Rs': 0.},
            {'eta': 0.035,  'Rs': 0.},
            {'eta': 0.06,   'Rs': 0.},
            {'eta': 0.1,    'Rs': 0.},
            {'eta': 0.2,    'Rs': 0.},
            {'eta': 0.4,    'Rs': 0.}
        ]

        params['g4'] = [
            {'zeta': 1,  'lambda': -1, 'eta': 0.0001},
            {'zeta': 1,  'lambda': 1,  'eta': 0.0001},
            {'zeta': 2,  'lambda': -1, 'eta': 0.0001},
            {'zeta': 2,  'lambda': 1,  'eta': 0.0001},
            {'zeta': 1,  'lambda': -1, 'eta': 0.003},
            {'zeta': 1,  'lambda': 1,  'eta': 0.003},
            {'zeta': 2,  'lambda': -1, 'eta': 0.003},
            {'zeta': 2,  'lambda': 1,  'eta': 0.003},
            {'zeta': 1,  'lambda': 1,  'eta': 0.008},
            {'zeta': 2,  'lambda': 1,  'eta': 0.008},
            {'zeta': 1,  'lambda': 1,  'eta': 0.015},
            {'zeta': 2,  'lambda': 1,  'eta': 0.015},
            {'zeta': 4,  'lambda': 1,  'eta': 0.015},
            {'zeta': 16,  'lambda': 1,  'eta': 0.015},
            {'zeta': 1,  'lambda': 1,  'eta': 0.025},
            {'zeta': 2,  'lambda': 1,  'eta': 0.025},
            {'zeta': 4,  'lambda': 1,  'eta': 0.025},
            {'zeta': 16,  'lambda': 1,  'eta': 0.025},
            {'zeta': 1,  'lambda': 1,  'eta': 0.045},
            {'zeta': 2,  'lambda': 1,  'eta': 0.045},
            {'zeta': 4,  'lambda': 1,  'eta': 0.045},
            {'zeta': 16,  'lambda': 1,  'eta': 0.045}
        ]

        # tranfer units from bohr to angstrom
        bhor2ang = 0.529177
        for key, values in params.items():
            for val in values:
                if key == 'g2':
                    val['eta'] /= bhor2ang**2
                elif key == 'g4':
                    val['eta'] /= bhor2ang**2

        super(Set30, self).__init__(cutname, cutvalue, params, *args, **kwargs)
