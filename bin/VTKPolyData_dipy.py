#!/usr/bin/env python3
"""
Description: Render a list of VTK data, track data, a nifti image, then view or save PNG or save WebGL.
The code uses dipy and fury.

Usage:
  VTKPolyData_dipy.py [--vtk f1,f2] [--vtk2 f1,f2] [--image <nifti_file>] [--track f1,f2] [--sh sh_file] [--tensor tensor_file] [--axes x,y,z] [--box x0,x1,y0,y1,z0,z1] [--image-opacity opa] [--sh-scale scale] [--sh-opacity opa] [--tensor-scale scale] [--tensor-opacity opa] [--size s1,s2] [--wc] [--frame] [--scalar-range r1,r2] [--png pngfile] [--zoom zoom] [--bgcolor r,g,b] [-v] [--no-normal] [--ni]
  VTKPolyData_dipy.py (-h | --help)

Options:

  --vtk f1,f2              Input VTK PolyData (.vtk, .vtp, etc), multiple inputs (file1,file2,...). Use HueRange=(0.6667,0).
  --vtk2 f1,f2             Input VTK Polydata, multiple inputs. Use HueRange=(0,1). Specified for meshes from tensors colored by directions, generated by MeshFromTensors.
  --image <nifti_file>     Input 3D nifti image file.
  --sh sh_file             Input 4D nifti sphercial harmonic (SH) coefficient image file for ODF or EAP.
  --track f1,f2            Input track file (.trk, .tck, .fib, .vtk, .dpy). Multiple inputs.
  --tensor tensor_file     Input 4D tensor file with 6 dimension (lower triangle ).
  --axes x,y,z             Visualize image/tensor/sh along x,y,z axes. Default 1,1,1 to show 3 axes, -1,1,1 to show y z axes.  [Default: 1,1,1]
  --box x0,x1,y0,y1,z0,z1  Visualize tensor/sh glyphs inside the box. It is not for --image. Default -1,-1,-1,-1,-1,-1 shows no box. [Default: -1,-1,-1,-1,-1,-1]
  --scalar-range r1,r2     lowest and highest scalar values for the vtk coloring. It is used when scalar dimention is 1. If not set, use the range of the scalar values. [Default: -1,-1]
  --size s1,s2             Window size in pixels. [Default: 1200,900]
  --image-opacity opa      Slice opacity for --image. [Default: 0.8]
  --sh-opacity opacity     SH glyph opacity for --sh. [Default: 1.0]
  --sh-scale scale         SH radial scale for --sh. [Default: 1.0]
  --tensor-scale scale     Tensor scale for --tensor. [Default: 200]
  --tensor-opacity opa     Tensor glyph opacity for --tensor. [Default: 1.0]
  --wc                     Use world coordinates.
  --png png_file           Output png file.
  --zoom zoom              Camera zoom factor. [Default: 1.0]
  --bgcolor r,g,b          Back ground color. [Default: 0,0,0]
  --frame                  Wireframe visualization.
  --no-normal              Do not use vtkPolyDataNormals for polydata visualization.
  --ni                     No interpolation for image. Set InterpolateOff.

  -h --help                Show this screen.
  -v --verbose             Verbose.


Author(s): Jian Cheng (jian.cheng.1983@gmail.com)
"""

import os, re
import numpy as np
from docopt import docopt

import utlVTK
from utlVTK import vtk
import utlDMRITool as utl

import nibabel as nib
from fury import actor, window, ui
from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram
from dipy.io.vtk import load_vtk_streamlines
from dipy.io.dpy import Dpy
from fury.utils import fix_winding_order
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
from dipy.reconst.dti import from_lower_triangular, decompose_tensor
from dipy.data import get_sphere


def arg_values(value, typefunc, numberOfValues):
    '''set arguments based using comma. If numberOfValues<0, it supports arbitrary number of inputs.'''
    value = value.strip()
    if value[0]=='(' and value[-1]==')':
        value = value[1:-1]
    values = value.split(',')
    if numberOfValues > 0 and len(values) != numberOfValues:
        raise "aa"
    return list(map(typefunc, values))


def get_input_args(args):
    '''parse args'''
    _args = args
    _args['--vtk'] = args['--vtk'].split(',') if args['--vtk'] else args['--vtk']
    _args['--vtk2'] = args['--vtk2'].split(',') if args['--vtk2'] else args['--vtk2']
    _args['--track'] = args['--track'].split(',') if args['--track'] else args['--track']
    _args['--axes'] = arg_values(args['--axes'], float, 3)
    _args['--box'] = arg_values(args['--box'], int, 6)
    _args['--scalar-range'] = arg_values(args['--scalar-range'], float, 2)
    _args['--size'] = arg_values(args['--size'], int, 2)
    _args['--bgcolor'] = arg_values(args['--bgcolor'], float, 3)
    _args['--image-opacity'] = arg_values(args['--image-opacity'], float, 1)[0]
    _args['--tensor-opacity'] = arg_values(args['--tensor-opacity'], float, 1)[0]
    _args['--tensor-scale'] = arg_values(args['--tensor-scale'], float, 1)[0]
    _args['--sh-opacity'] = arg_values(args['--sh-opacity'], float, 1)[0]
    _args['--sh-scale'] = arg_values(args['--sh-scale'], float, 1)[0]
    _args['--zoom'] = arg_values(args['--zoom'], float, 1)[0]
    return _args


def set_box_on_shape(box, shape):
    '''correct box values based on shape'''

    for i in range(3):
        if box[2*i]>box[2*i+1]:
            raise("wrong box is given. box=", box)
        box[2*i] = max(box[2*i], 0)
        box[2*i+1] = shape[i]-1 if box[2*i+1]<0 else min(box[2*i+1], shape[i]-1)


def update_visualbox(box, vbox):
    '''update the visual vbox based on the given box'''

    # if box is default value, do not change vbox
    if box==[-1]*len(box):
        return

    #  if vbox is a x/y/z slice, and it is out of the box, make it invisible
    for i in range(3):
        if vbox[2*i]==vbox[2*i+1]:
            if box[2*i]>=0 and box[2*i+1]>=0 and not box[2*i]<=vbox[2*i]<=box[2*i+1]:
                vbox[2*i] = vbox[2*i+1] = -1
            elif box[2*i]<0 and box[2*i+1]>=0 and not vbox[2*i]<=box[2*i+1]:
                vbox[2*i] = vbox[2*i+1] = -1
            elif box[2*i]>=0 and box[2*i+1]<0 and not box[2*i]<=vbox[2*i]:
                vbox[2*i] = vbox[2*i+1] = -1

    # update vbox as the intersection between box and vbox
    for i in range(3):
        if vbox[2*i]!=-1 and box[2*i]!=-1:
            vbox[2*i] = max(box[2*i], vbox[2*i])
        if vbox[2*i+1]!=-1 and box[2*i+1]!=-1:
            vbox[2*i+1] = min(box[2*i+1], vbox[2*i+1])


def scene_add_tract(scene, track_file, affine, _args):
    '''add a track file'''

    if _args['--image']:
        tg = load_tractogram(track_file, _args['--image'], bbox_valid_check=False)
        streamlines = tg.streamlines
    else:
        _, extension = os.path.splitext(track_file)
        if extension == '.trk':
            tg = load_tractogram(track_file, 'same', bbox_valid_check=False)
            streamlines = tg.streamlines
        elif extension == '.tck':
            tractogram_obj = nib.streamlines.load(track_file).tractogram
            streamlines = tractogram_obj.streamlines
        elif extension in ['.vtk', '.fib']:
            streamlines = load_vtk_streamlines(track_file)
        elif extension in ['.dpy']:
            dpy_obj = Dpy(track_file, mode='r')
            streamlines = list(dpy_obj.read_tracks())
            dpy_obj.close()

    if not _args['--wc']:
        from dipy.tracking.streamline import transform_streamlines
        streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

    stream_actor = actor.line(streamlines)

    scene.add(stream_actor)


def scene_add_vtk(scene, vtk_file, _args, is_vtk2):
    '''add a vtk file'''

    polyData = utlVTK.readPolydata(vtk_file)

    if _args['--frame']:
        frame_mapper = vtk.vtkDataSetMapper()
        frame_mapper.SetInputData(polyData)
        frame_actor = vtk.vtkLODActor()
        frame_actor.SetMapper(frame_mapper)
        prop = frame_actor.GetProperty()
        prop.SetRepresentationToWireframe()
        prop.SetColor(0.0, 0.0, 1.0)
        scene.AddActor(frame_actor)

    surface_mapper = vtk.vtkDataSetMapper()

    if not _args['--no-normal'] and polyData.GetPointData().GetNormals() is None:
        polyDataNormals = vtk.vtkPolyDataNormals()
        try:
            polyDataNormals.SetInputData(polyData)
        except:
            polyDataNormals.SetInput(polyData)
        # polyDataNormals.SetFeatureAngle(90.0)
        surface_mapper.SetInputConnection(
            polyDataNormals.GetOutputPort())
    else:
        try:
            surface_mapper.SetInputData(polyData)
        except:
            surface_mapper.SetInput(polyData)

    if polyData.GetPointData().GetScalars() and polyData.GetPointData().GetScalars().GetNumberOfComponents()==1:
        lut = vtk.vtkLookupTable()
        if _args['--scalar-range'][0] == -1 or _args['--scalar-range'][1] == -1:
            valueRange = polyData.GetScalarRange()
        vr0 = _args['--scalar-range'][0] if _args['--scalar-range'][0] != -1 else valueRange[0]
        vr1 = _args['--scalar-range'][1] if _args['--scalar-range'][1] != -1 else valueRange[1]
        valueRange = (vr0, vr1)
        lut.SetTableRange(valueRange[0], valueRange[1])
        if is_vtk2:
            #  for tensors colored by directions
            lut.SetHueRange(0.0,1.0)
        else:
            lut.SetHueRange(0.6667, 0)
        #  lut.SetHueRange(args.hue_range[0], args.hue_range[1])
        lut.SetRampToLinear()
        lut.Build()

        surface_mapper.SetLookupTable(lut)
        surface_mapper.SetScalarRange(valueRange[0], valueRange[1])

    surface_actor = vtk.vtkLODActor()
    surface_actor.SetMapper(surface_mapper)
    prop = surface_actor.GetProperty()
    prop.SetRepresentationToSurface()

    scene.add(surface_actor)


def scene_add_image(scene, image_file, actor_dict, _args):
    '''add a 3D image'''

    data, affine = load_nifti(image_file)
    shape = data.shape
    if _args['--verbose']:
        print('image shape=', shape)
        print('image affine=', affine)

    if not _args['--wc']:
        actor_dict['image_actor_z'] = actor.slicer(data, affine=np.eye(4))
    else:
        actor_dict['image_actor_z'] = actor.slicer(data, affine)

    actor_dict['image_actor_z'].opacity(_args['--image-opacity'])

    actor_dict['image_actor_x'] = actor_dict['image_actor_z'].copy()
    x_midpoint = int(np.round(shape[0] / 2))
    actor_dict['image_actor_x'].display_extent(x_midpoint,
                                x_midpoint, 0,
                                shape[1] - 1,
                                0,
                                shape[2] - 1)

    actor_dict['image_actor_y'] = actor_dict['image_actor_z'].copy()
    y_midpoint = int(np.round(shape[1] / 2))
    actor_dict['image_actor_y'].display_extent(0,
                                shape[0] - 1,
                                y_midpoint,
                                y_midpoint,
                                0,
                                shape[2] - 1)

    actor_dict['image_actor_x'].InterpolateOff() if _args['--ni'] else actor_dict['image_actor_x'].InterpolateOn()
    actor_dict['image_actor_y'].InterpolateOff() if _args['--ni'] else actor_dict['image_actor_y'].InterpolateOn()
    actor_dict['image_actor_z'].InterpolateOff() if _args['--ni'] else actor_dict['image_actor_z'].InterpolateOn()

    if _args['--axes'][0]==1:
        scene.add(actor_dict['image_actor_x'])
    if _args['--axes'][1]==1:
        scene.add(actor_dict['image_actor_y'])
    if _args['--axes'][2]==1:
        scene.add(actor_dict['image_actor_z'])

    return affine, shape


def scene_add_sh(scene, sh_file, actor_dict, _args):
    '''add a 4D SH image file'''

    sh_img = nib.load(sh_file)
    sh = sh_img.get_fdata()
    sh_affine = sh_img.affine
    affine = sh_affine if _args['--wc'] else np.eye(4)
    grid_shape = sh.shape[:-1]
    sh_order = order_from_ncoef(sh.shape[-1])

    sphere_low = get_sphere('repulsion100')
    B_low = sh_to_sf_matrix(sphere_low, sh_order, return_inv=False)

    sphere_high = get_sphere('symmetric362')
    sphere_high.faces = fix_winding_order(sphere_high.vertices, sphere_high.faces, True)
    B_high = sh_to_sf_matrix(sphere_high, sh_order, return_inv=False)

    _args['sphere_dict'] = {'Low resolution': (sphere_low, B_low),
                'High resolution': (sphere_high, B_high)}

    scale = 0.5*_args['--sh-scale']
    norm = False
    colormap = None
    radial_scale = True
    opacity = _args['--sh-opacity']
    global_cm = False

    # SH (ODF/EAP) slicer for axial slice
    vbox = [0, grid_shape[0] - 1, 0, grid_shape[1] - 1, grid_shape[2]//2, grid_shape[2]//2]
    update_visualbox(_args['--box'], vbox)
    actor_dict['sh_actor_z'] = actor.odf_slicer(sh, affine=affine, sphere=sphere_low,
                                scale=scale, norm=norm,
                                radial_scale=radial_scale, opacity=opacity,
                                colormap=colormap, global_cm=global_cm,
                                B_matrix=B_low)
    actor_dict['sh_actor_z'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])

    # SH slicer for coronal slice
    vbox = [0, grid_shape[0] - 1, grid_shape[1]//2, grid_shape[1]//2, 0, grid_shape[2] - 1]
    update_visualbox(_args['--box'], vbox)
    actor_dict['sh_actor_y'] = actor.odf_slicer(sh, affine=affine, sphere=sphere_low,
                                scale=scale, norm=norm,
                                radial_scale=radial_scale, opacity=opacity,
                                colormap=colormap, global_cm=global_cm,
                                B_matrix=B_low)
    actor_dict['sh_actor_y'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])

    # SH slicer for sagittal slice
    vbox = [grid_shape[0]//2, grid_shape[0]//2, 0, grid_shape[1] - 1, 0, grid_shape[2] - 1]
    update_visualbox(_args['--box'], vbox)
    actor_dict['sh_actor_x'] = actor.odf_slicer(sh, affine=affine, sphere=sphere_low,
                                scale=scale, norm=norm,
                                radial_scale=radial_scale, opacity=opacity,
                                colormap=colormap, global_cm=global_cm,
                                B_matrix=B_low)
    actor_dict['sh_actor_x'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])

    if _args['--axes'][0]==1:
        scene.add(actor_dict['sh_actor_x'])
    if _args['--axes'][1]==1:
        scene.add(actor_dict['sh_actor_y'])
    if _args['--axes'][2]==1:
        scene.add(actor_dict['sh_actor_z'])

    return sh_affine, grid_shape


def scene_add_tensor(scene, tensor_file, actor_dict, _args):
    '''add a 4D tensor image file with 6 dimension (lower triangle format)'''

    tensor_img = nib.load(tensor_file)
    tensor = tensor_img.get_fdata()
    tensor_affine = tensor_img.affine

    affine = tensor_affine if _args['--wc'] else np.eye(4)
    grid_shape = tensor.shape[:-1]

    evals, evecs = decompose_tensor(from_lower_triangular(np.asarray(tensor)),
                                    min_diffusivity=0)


    # Do not normalize eigenvalues by default
    norm_evals = False

    #  sphere = get_sphere('symmetric362')
    sphere = get_sphere('repulsion100')
    scale = _args['--tensor-scale']
    opacity = _args['--tensor-opacity']

    vbox = [0, grid_shape[0] - 1, 0, grid_shape[1] - 1, grid_shape[2]//2, grid_shape[2]//2]
    update_visualbox(_args['--box'], vbox)
    actor_dict['tensor_actor_z'] = actor.tensor_slicer(evals, evecs, affine, norm=norm_evals, sphere=sphere, scale=scale, opacity=opacity)
    actor_dict['tensor_actor_z'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])

    vbox = [0, grid_shape[0] - 1, grid_shape[1]//2, grid_shape[1]//2, 0, grid_shape[2] - 1]
    update_visualbox(_args['--box'], vbox)
    actor_dict['tensor_actor_y'] = actor.tensor_slicer(evals, evecs, affine, norm=norm_evals, sphere=sphere, scale=scale, opacity=opacity)
    actor_dict['tensor_actor_y'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])

    vbox = [grid_shape[0]//2, grid_shape[0]//2, 0, grid_shape[1] - 1, 0, grid_shape[2] - 1]
    update_visualbox(_args['--box'], vbox)
    actor_dict['tensor_actor_x'] = actor.tensor_slicer(evals, evecs, affine, norm=norm_evals, sphere=sphere, scale=scale, opacity=opacity)
    actor_dict['tensor_actor_x'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])


    if _args['--axes'][0]==1:
        scene.add(actor_dict['tensor_actor_x'])
    if _args['--axes'][1]==1:
        scene.add(actor_dict['tensor_actor_y'])
    if _args['--axes'][2]==1:
        scene.add(actor_dict['tensor_actor_z'])

    return tensor_affine, grid_shape


def scene_add_ui(scene, _args, actor_dict, affine, shape):
    '''add ui for image slice'''

    line_slider_x = ui.LineSlider2D(min_value=0,
                                    max_value=shape[0] - 1 if shape[0]>1 else 1,
                                    initial_value=shape[0] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_y = ui.LineSlider2D(min_value=0,
                                    max_value=shape[1] - 1 if shape[1]>1 else 1,
                                    initial_value=shape[1] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1 if shape[2]>1 else 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    opacity_slider = ui.LineSlider2D(min_value=0.0,
                                    max_value=1.0,
                                    initial_value=_args['--image-opacity'],
                                    length=140)


    def change_slice_x(slider):
        x = int(np.round(slider.value))
        vbox = [x, x, 0, shape[1] - 1, 0, shape[2] - 1]
        update_visualbox(_args['--box'], vbox)
        if _args['--image']:
            actor_dict['image_actor_x'].display_extent(x, x, 0, shape[1] - 1, 0, shape[2] - 1)
        if _args['--tensor']:
            actor_dict['tensor_actor_x'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])
        if _args['--sh']:
            actor_dict['sh_actor_x'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])

    def change_slice_y(slider):
        y = int(np.round(slider.value))
        vbox = [0, shape[0] - 1, y, y, 0, shape[2] - 1]
        update_visualbox(_args['--box'], vbox)
        if _args['--image']:
            actor_dict['image_actor_y'].display_extent(0, shape[0] - 1, y, y, 0, shape[2] - 1)
        if _args['--tensor']:
            actor_dict['tensor_actor_y'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])
        if _args['--sh']:
            actor_dict['sh_actor_y'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])

    def change_slice_z(slider):
        z = int(np.round(slider.value))
        vbox = [0, shape[0] - 1, 0, shape[1] - 1, z, z]
        update_visualbox(_args['--box'], vbox)
        if _args['--image']:
            actor_dict['image_actor_z'].display_extent(0, shape[0] - 1, 0, shape[1] - 1, z, z)
        if _args['--tensor']:
            actor_dict['tensor_actor_z'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])
        if _args['--sh']:
            actor_dict['sh_actor_z'].display_extent(vbox[0],vbox[1],vbox[2],vbox[3],vbox[4],vbox[5])

    def change_opacity(slider):
        _args['--image-opacity'] = slider.value
        if _args['--image']:
            actor_dict['image_actor_x'].opacity(_args['--image-opacity'])
            actor_dict['image_actor_y'].opacity(_args['--image-opacity'])
            actor_dict['image_actor_z'].opacity(_args['--image-opacity'])

    def change_sphere(combobox):
        sphere, B = _args['sphere_dict'][combobox.selected_text]
        actor_dict['sh_actor_x'].update_sphere(sphere.vertices, sphere.faces, B)
        actor_dict['sh_actor_y'].update_sphere(sphere.vertices, sphere.faces, B)
        actor_dict['sh_actor_z'].update_sphere(sphere.vertices, sphere.faces, B)

    line_slider_x.on_change = change_slice_x
    line_slider_y.on_change = change_slice_y
    line_slider_z.on_change = change_slice_z
    opacity_slider.on_change = change_opacity

    if _args['--sh']:
        combobox = ui.ComboBox2D(items=list(_args['sphere_dict']))
        scene.add(combobox)
        combobox.on_change = change_sphere


    def build_label(text):
        label = ui.TextBlock2D()
        label.message = text
        label.font_size = 18
        label.font_family = 'Arial'
        label.justification = 'left'
        label.bold = False
        label.italic = False
        label.shadow = False
        label.background_color = (0, 0, 0)
        label.color = (1, 1, 1)

        return label


    num = int(_args['--axes'][0]==1)+ int(_args['--axes'][1]==1) + int(_args['--axes'][2]==1)
    bgc = _args['--bgcolor']
    panel = ui.Panel2D(size=(300, 50*(num+1)),
                    color=(1-bgc[0], 1-bgc[1], 1-bgc[2]),
                    opacity=0.1,
                    align="right")
    panel.center = (_args['--size'][0]-200, 120)

    high_1 = 0.6
    if _args['--axes'][0]==1:
        line_slider_label_x = build_label(text="X Slice")
        panel.add_element(line_slider_label_x, (0.1, high_1 if num==1 else 0.75))
        panel.add_element(line_slider_x, (0.38, high_1 if num==1 else 0.75))
    if _args['--axes'][1]==1:
        line_slider_label_y = build_label(text="Y Slice")
        panel.add_element(line_slider_label_y, (0.1, high_1 if num==1 else 0.55))
        panel.add_element(line_slider_y, (0.38, high_1 if num==1 else 0.55))
    if _args['--axes'][2]==1:
        line_slider_label_z = build_label(text="Z Slice")
        panel.add_element(line_slider_label_z, (0.1, high_1 if num==1 else 0.35))
        panel.add_element(line_slider_z, (0.38, high_1 if num==1 else 0.35))

    if _args['--axes'][0]==1 or _args['--axes'][1]==1 or _args['--axes'][2]==1:
        opacity_slider_label = build_label(text="Opacity")
        panel.add_element(opacity_slider_label, (0.1, 0.15))
        panel.add_element(opacity_slider, (0.38, 0.15))
        scene.add(panel)

    return panel



def main():

    args = docopt(utl.app_doc(__doc__), version='1.0')

    if (args['--verbose']):
        print(args)

    _args = get_input_args(args)

    if (args['--verbose']):
        print('_args=',_args)

    if not _args['--vtk'] and not _args['--vtk2'] and not _args['--image'] and not _args['--sh'] and not _args['--tensor'] and not _args['--track']:
        raise("need inputs for --vtk, --vtk2, --image, --sh, --tensor")

    affine=np.eye(4)
    shape=[]

    scene = window.Scene()
    actor_dict = {}


    #  add vtk files
    if _args['--vtk']:
        for tf in _args['--vtk']:
            scene_add_vtk(scene, os.path.expanduser(tf), _args, False)


    #  add vtk2 files for tensors
    if _args['--vtk2']:
        for tf in _args['--vtk2']:
            scene_add_vtk(scene, os.path.expanduser(tf), _args, True)

    #  add an image file
    if _args['--image']:
        affine, shape = scene_add_image(scene, _args['--image'], actor_dict, _args)

    if _args['--tensor']:
        tensor_affine, tensor_shape = scene_add_tensor(scene, _args['--tensor'], actor_dict, _args)

        if _args['--image'] and tensor_shape!=shape:
            print("Warning: tensor shape is different from image shape. tensor_shape=", tensor_shape, ", image shape=", shape)
            shape = min(shape, tensor_shape)
        if _args['--image'] and np.linalg.norm(tensor_affine-affine)>1e-5:
            print("Warning: tensor affine is different from image affine. tensor_affne=", tensor_affine, ", image affine=", affine)
        if not _args['--image']:
            affine, shape = tensor_affine, tensor_shape

    #  add a SH file
    if _args['--sh']:
        sh_affine, sh_shape = scene_add_sh(scene, _args['--sh'], actor_dict, _args)

        if _args['--image'] and sh_shape!=shape:
            print("Warning: sh shape is different from image shape. sh_shape=", sh_shape, ", image shape=", shape)
            shape = min(shape, sh_shape)
        if _args['--image'] and np.linalg.norm(sh_affine-affine)>1e-5:
            print("Warning: sh affine is different from image affine. sh_affne=", sh_affine, ", image affine=", affine)
        if not _args['--image']:
            affine, shape = sh_affine, sh_shape

    if _args['--verbose']:
        print('shape=', shape)
        print('affine=', affine)

    if _args['--image'] or _args['--tensor'] or _args['--sh']:
        set_box_on_shape(_args['--box'], shape)
        if _args['--verbose']:
            print('set box=', _args['--box'])

    #  add track files
    if _args['--track']:
        for tf in _args['--track']:
            scene_add_tract(scene, os.path.expanduser(tf), affine, _args)

    show_m = window.ShowManager(scene, size=(_args['--size']))
    show_m.initialize()

    # add ui for image slice
    if _args['--image'] or _args['--sh'] or _args['--tensor']:
        panel = scene_add_ui(scene, _args, actor_dict, affine, shape)



    global size
    size = scene.GetSize()

    def win_callback(obj, _event):
        global size
        if size != obj.GetSize():
            size_old = size
            size = obj.GetSize()
            size_change = [size[0] - size_old[0], 0]
            panel.re_align(size_change)

    scene.SetBackground(_args['--bgcolor'])

    show_m.initialize()

    scene.zoom(_args['--zoom'])
    scene.reset_clipping_range()

    if not _args['--png']:

        show_m.add_window_callback(win_callback)
        show_m.render()
        show_m.start()

    else:

        window.record(scene, out_path=_args['--png'], size=(_args['--size']),
                    reset_camera=False)





if __name__ == '__main__':
    main()

