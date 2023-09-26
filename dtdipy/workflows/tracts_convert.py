#!/usr/bin/env python

import logging
import numpy as np

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.workflows.workflow import Workflow
from dipy.io.image import load_nifti
from dipy.tracking.streamline import transform_streamlines


class TractsConvertFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'tracts_convert'

    def run(self, input_files, out_tracts='out_tracts.trk', reference='same', vox=False, out_dir=''):

        """ Workflow for converting a tract file.

        Parameters
        ----------
        input_files : string
            Path to a tract file.
        out_tracts : string, optional
            Name of the output tract file.

        reference : string, optional
            Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
            trk.header (dict), or 'same' if the input is a trk file.
            Reference that provides the spatial attribute to override spatial attribute in input.
            Typically a nifti-related object from the native diffusion used for
            streamlines generation
        vox : boolean, optional
            transform streamlines from ras to vox space. Use it carefully.
        out_dir : string, optional
            Output directory. (default current directory)

        References
        ----------
        dmritool-dipy (https://github.com/DiffusionMRITool/dtdipy)
        """

        io_it = self.get_io_iterator()

        for input_path, out_tract_path in io_it:

            logging.info('Convert tracts of {0}'.format(input_path))

            tracts = load_tractogram(input_path, reference, bbox_valid_check=False)

            if reference!='same' and vox:
                _, affine = load_nifti(reference)
                tracts.streamlines = transform_streamlines(tracts.streamlines, np.linalg.inv(affine))

            save_tractogram(tracts, out_tracts, bbox_valid_check=False)

            logging.info('Tracts saved at {0}'.format(out_tract_path))

