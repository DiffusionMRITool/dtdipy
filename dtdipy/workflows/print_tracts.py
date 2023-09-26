#!/usr/bin/env python

import os
import logging
import numpy as np
import nibabel as nib

from dipy.io.streamline import load_tractogram, Origin
from dipy.io.vtk import load_vtk_streamlines
from dipy.io.dpy import Dpy
from dipy.io.image import load_nifti
from dipy.tracking.streamline import transform_streamlines

from dipy.workflows.workflow import Workflow


class PrintTracts(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'print_tracts'

    def run(self, input_files, reference='same', space='ras', origin='center'):

        """ Workflow for converting a tract file.

        Parameters
        ----------
        input_files : string
            Path to tract file.

        reference : string, optional
            Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
            trk.header (dict), or 'same' if the input is a trk file.
            Reference that provides the spatial attribute to override spatial attribute in input.
            Typically a nifti-related object from the native diffusion used for
            streamlines generation
        space : string, optional
            'ras': ras space,
            'vox': voxel space.
        origin : string, optional
            Current origin in which the streamlines are (center or corner).
            After loading with nibabel the origin is CENTER.
            'center': NIFTI.
            'corner': TRACKVIS.

        References
        ----------
        dmritool-dipy (https://github.com/DiffusionMRITool/dtdipy)
        """

        io_it = self.get_io_iterator()

        for input_path in io_it:

            if space.lower() == 'vox':
                logging.info('Print tract of {0} in the VOX space'.format(input_path))
            elif space.lower() == 'ras':
                logging.info('Print tract of {0} in the RAS space'.format(input_path))
            else:
                raise('wrong space')

            affine=None

            _, extension = os.path.splitext(input_path)
            if extension == '.trk':
                tg = load_tractogram(input_path, 'same', to_origin=Origin(origin.lower()), bbox_valid_check=False)
                streamlines = tg.streamlines
                affine = tg.affine
                print('trk header:\n', tg)
            elif extension == '.tck':
                tractogram_obj = nib.streamlines.load(input_path).tractogram
                streamlines = tractogram_obj.streamlines
            elif extension in ['.vtk', '.fib']:
                streamlines = load_vtk_streamlines(input_path)
            elif extension in ['.dpy']:
                dpy_obj = Dpy(input_path, mode='r')
                streamlines = list(dpy_obj.read_tracks())
                dpy_obj.close()

            if reference!='same' and affine is None:
                _, affine = load_nifti(reference)

            if affine is not None and space.lower() =='vox':
                streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))
            elif affine is None and space.lower() =='vox':
                raise('need to set reference')

            print(streamlines)



