#!/usr/bin/env python

import logging

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.workflows.workflow import Workflow
import fury


class TrackConvertFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'track_convert'

    def run(self, input_files, out_track='out_track.trk', reference='same', out_dir=''):

        """ Workflow for creating a binary mask

        Parameters
        ----------
        input_files : string
           Path to track file.
        out_track : string, optional
           Name of the output track file.

        reference : string, optional
            Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
            trk.header (dict), or 'same' if the input is a trk file.
            Reference that provides the spatial attribute to override spatial attribute in input.
            Typically a nifti-related object from the native diffusion used for
            streamlines generation
        out_dir : string, optional
           Output directory. (default current directory)
        """

        io_it = self.get_io_iterator()

        for input_path, out_track_path in io_it:

            logging.info('Convert track of {0}'.format(input_path))

            track = load_tractogram(input_path, reference, bbox_valid_check=False)
            save_tractogram(track, out_track, bbox_valid_check=False)

            logging.info('Track saved at {0}'.format(out_track_path))

