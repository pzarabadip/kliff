import edn_format
import numpy as np
import collections

class WrapCalculator:
    """ Wrapper over user-defined predictor function.

    Parameters
    ----------

    runner, function
        User defined function to generate prediction for a material property.
        For example, using LAMMPS to compute the elastic constant.

    outname, str
        Name of file that stores the results generated by 'runner'. The file should
        be in EDN format.

    keys: list of str
        Keywords in the 'outname' EDN file, whose value will be parsed as the prediction.
    """

    def __init__(self, runner, outname, keys):
        self.runner = runner
        self.outname = outname
        self.keys = keys

    def get_prediction(self):
        """
        Return 1D array of floats.
        """
        self.runner()
        return self._parse_edn(self.outname, self.keys)


    def update_params(self):
        pass

    def _parse_edn(self, fname, keys):
        """ Wrapper to use end_format to parse output file of 'runner' in edn format.

        Parameters
        ----------

        fname: str
            Name of the output file of OpenKIM test.

        keys: list of str
            Keyword in the edn format(think it as a dictionary), whose value will be returned.
        """

        with open(fname, 'r') as fin:
            lines = fin.read()
        parsed = edn_format.loads(lines)
        values = []
        for k in keys:
            try:
                v = parsed[k]['source-value']
            except KeyError:
                raise KeyError('Keyword "{}" not found in {}.'.format(k, self.outname))
            # make it a 1D array
            # we expect v as a python built-in object (if numpy object, this will fail)
            if isinstance(v, collections.Sequence):
                v = np.array(v).flatten()
            else:
                v = [v]
            values.append(v)
        return np.concatenate(values)




