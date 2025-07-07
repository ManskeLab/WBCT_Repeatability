# -----------------------------------------------------
# cbm.py
#
# Created by:   Tadiwa Waungana
# Created on:   1 March, 2023
# Modified on:  6 December, 2024
#
# Description: Estimate cortical bone thickness using grayscale image and bone meshes.
#              Firstly, surface normal rays are cast from a source bone mesh into the
#              grayscale image space. The sampled image data is used to estimate
#              cortical thickness using methods descrived by Treece et al., 2012.
#
# -----------------------------------------------------
# Usage: python cbm.py <input_image.nii> <bone_mesh.vtk> <trab_std> <method> <ray_length>
#
#
# Inputs:
# 	1. input image (.nii)
#   2. bone mesh (.vtk)
#   3. trab_std (standard deviation of tibial trabecular grayscale values in a 10x10x10 voi)
#   4. method (estimation method - hybrid or constant density)
#   5. ray_length (length of normal rays used to sample grayscale image)
#
# Outputs: (Written to the directory that holds the greyscale image files by default)
# 	1. Source bone surface model with cortical thickness distribution (.vtk)
#   2. Thickness estimation parameters and errors (.xsl)
#
# -----------------------------------------------------

import os
import vtk
import sys
import argparse
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv

#----------------------------Helper functions----------------------------------#

from cbm_util import compute_normals, cell_centers, find_thickness, mass_conservation, printProgressBar

#------------------------------------------------------------------------------#

if __name__ =="__main__":

    ''' Parameters for the peak detection algorithm in joint space mapping:
     Lag: The number of sample voxels values used to determine the moving average of signal intensity - default (5)
     Threshold: Scale factor to determine when a signal peak/cortical edge has been detected - default (2.0)
     Influence: How much each voxel value affects the moving average - default (0.5) '''

    LAG = 5
    THRESHOLD = 2.0             # default = 2.0
    INFLUENCE = 0.5             # default = 0.5

    colours = vtk.vtkNamedColors()

    # parse in necessary arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type = str, help = "The input image (path + filename)")
    parser.add_argument("input_bone_model", type = str, help = "The bone surface model (path + filename)")
    parser.add_argument("trab_std", type = float, help = "Estimate of HU standard deviation in trabecular bone - used in the peak detection algorithm")
    parser.add_argument("method", type = float, nargs='?', default = 1, help = "The selected cortical bone mapping approach: 1 = hybrid (Treece et al., 2012); 2 = constant density (Treece et al., 2012)" )
    parser.add_argument("ray_length",  type = float, nargs='?', default = 7, help = "The length of sampling ray used to sample image data")
    args = parser.parse_args()

    ''' Ray_length: length of the ray/search vector vector in coordinate space - default (7)'''
    ray_length = args.ray_length

    ''' Read grayscale image as nifti (.nii)'''
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(args.input_image)
    reader.Update()

    ''' Obtain image spacing and origin to correctly scale thickness measures'''
    spacing = reader.GetOutput().GetSpacing()[1]
    origin = reader.GetOutput().GetOrigin()

    ''' Read pre-processed bone meshes using PyVista '''
    surf1_mesh = pv.read(args.input_bone_model)             # map for thickness measures with original Ct. density estimate
    surf1_mesh_modified = pv.read(args.input_bone_model)    # map for thickness measures with refined/improved Ct. density estimate
    surf1_mesh_diff = pv.read(args.input_bone_model)        # mao for thickness difference between above two maps

    #   Incase scalar for thickness estimates and model parameters wasn't pre-set during mesh generation
    surf1_mesh['Thickness'] = np.empty(surf1_mesh.n_points)
    surf1_mesh['y1'] = np.empty(surf1_mesh.n_points)
    surf1_mesh['yb'] = np.empty(surf1_mesh.n_points)

    #   Incase scalar for thickness estimates wasn't pre-set during mesh generation
    surf1_mesh_modified['Thickness'] = np.empty(surf1_mesh_modified.n_points)

    #   Create scalar array to hold thickness difference values
    surf1_mesh_diff['Thickness difference'] = np.empty(surf1_mesh_modified.n_points)

    ''' Compute normal rays used to sample greyscale image data'''
    surface_normals = compute_normals(surf1_mesh)

    ''' Obtain the cells associated with each normal ray'''
    surf_mesh_cells = cell_centers(surface_normals)

    print('\n   Normal extraction starting...')

    surface_mesh_normals_data = surface_normals.GetOutput().GetPointData().GetArray("Normals")
    
    #   Create vtkData to hold information pertaining to the sample locations and sample directions
    valid_Normals = vtk.vtkCellArray()
    pts = vtk.vtkPoints()

    #   Convert original image to a pointset so we can probe the image data using the extracted surface normals
    image_points = vtk.vtkImageDataToPointSet()
    image_points.SetInputConnection(reader.GetOutputPort())
    image_points.Update()

    intersect_count = 0
    count = 0

    #   Create datafram to contain CBM measures from each sample location - this data is used to estimate the cortical density to be used for CBM
    thick_data =  pd.DataFrame({'apparent_peak_density':[],
                        'precision_y0':[],
                        'precision_x0':[],
                        'precision_sig':[],
                        'precision_y2':[],
                        'precision_x1':[],
                        'apparent_thickness':[],
                        'background_density':[],
                        'p(i)':[]})

    ''' CBM - Map cortial thickness using all available normals'''    
    #   Progress bar that is printed to terminal
    n = surf1_mesh.n_points
    printProgressBar(0, n, prefix = 'Estimating global cortical thickness:', suffix = 'Complete', length = 50)
    
    for idx in range(surf1_mesh.n_points):
        count +=1
        start = np.empty(3)
        surf1_mesh.GetPoint(idx, start)
        surf_fem1Normal = surface_mesh_normals_data.GetTuple(idx)

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 2*idx)
        line.GetPointIds().SetId(1, 2*idx + 1)

        valid_Normals.InsertNextCell(line)

        pSurf = start

        ''' Default parameters '''
        #   Position the sample ray so that it conpletely traverses the cortical bone
        p1 = list(np.array(list(start)) - 0.5*ray_length*np.array(list(surf_fem1Normal)))
        p2 = list(np.array(list(start)) + 1.5*ray_length*np.array(list(surf_fem1Normal)))

        ''' Ensure that the image data is sampled at a frequency that corresponds to the isotropic voxel spacing '''        
        math1 = vtk.vtkMath()
        distance = math.sqrt(math1.Distance2BetweenPoints(p1, p2))

        line = vtk.vtkLineSource()
        line.SetPoint1(p1)
        line.SetPoint2(p2)
        line.SetResolution( int(distance//spacing) )                              # Number of samples obtained along that probe line

        probe_spacing = (distance/line.GetResolution())

        #   Samples image data along the defined line
        probe_line = vtk.vtkProbeFilter()                                       
        probe_line.SetInputConnection(line.GetOutputPort())
        probe_line.SetSourceConnection(image_points.GetOutputPort())
        probe_line.Update()

        #   Default thickness placeholder
        thickness = -1

        #   Try map cortical thickness at each point on the surface
        #   The cortical density estimate is fixed to the maximum density in the sample profile:
        #   this is handled within the find_thickness function and controlled using the density_bool argument
        try:
            thickness, background, apparent_density, data, point_number_refined, intensity_vector_refined, model_fit, p_i, res2 =\
            find_thickness(probe_line, probe_spacing, trabecular_std = args.trab_std, \
                           LAG = LAG, THRESHOLD = THRESHOLD, INFLUENCE = INFLUENCE,\
                              density_bool = 0, apparent_density = 1000, idx = idx, p1 = 0, p2 = 0, pSurf = 0)
        except:
            thickness = -1

        if thickness > 0:
            #   add CBM data from each sample location to the dataframe that will be used to estimate the cortical density
            thick_data = pd.concat([thick_data, data], ignore_index=True)
            #   Assign CBM measures to each point on the surface model
            surf1_mesh['Thickness'][idx] = thickness
            surf1_mesh['y1'][idx] = apparent_density
            surf1_mesh['yb'][idx] = background
        
        #   Update the terminal progress bar after each measurement
        time.sleep(0.1)
        printProgressBar(idx + 1, n, prefix = 'Estimating global cortical thickness:', suffix = 'Complete', length = 50)

    #   Save the CBM data - for later verfication if necessary
    data_name = os.path.splitext(os.path.basename(args.input_bone_model))[0] + 'thickness_data.csv'
    thick_data.to_csv(os.path.join( os.path.dirname(args.input_image),data_name) ,index=False)
    
    #   Estimate the cortical density using the CBM data from all sample locations
    y1_estimate = mass_conservation(thick_data, args)

    #   Modify the cortical thickness estimates using the newly estimated cortical density
    if args.method == 1:
        #   Follows the hybrib method laid out in Treece et al.(2012)
        #   The thickness estimates from the first CBM iteration are quickly modified - fast
        for i in range(0,surf1_mesh.n_points):
            surf1_mesh_modified['Thickness'][i] = surf1_mesh['Thickness'][i] * ((surf1_mesh['y1'][i] - surf1_mesh['yb'][i])/(y1_estimate - surf1_mesh['yb'][i]))
            surf1_mesh_diff['Thickness difference'][i] = surf1_mesh_modified['Thickness'][i] - surf1_mesh['Thickness'][i]
    elif args.method == 2:
        #   Follows the hybrib method laid out in Treece et al.(2012)
        #   CBM is performed over the entire the surface again - this time using the cortical density estimate to constrain the estimation algorithm/model
        count = 0
        for idx in range(surf1_mesh.n_points): 
            count +=1
            start = np.empty(3)
            surf1_mesh.GetPoint(idx, start)
            surf_fem1Normal = surface_mesh_normals_data.GetTuple(idx)

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 2*idx)
            line.GetPointIds().SetId(1, 2*idx + 1)

            valid_Normals.InsertNextCell(line)

            pSurf = start
            ''' Default parameters '''
            #   Position the sample ray so that it conpletely traverses the cortical bone

            p1 = list(np.array(list(start)) - 0.5*ray_length*np.array(list(surf_fem1Normal)))
            p2 = list(np.array(list(start)) + 1.5*ray_length*np.array(list(surf_fem1Normal)))

            ''' Ensure that the image data is sampled at a frequency that corresponds to the isotropic voxel spacing ''' 
            math1 = vtk.vtkMath()
            distance = math.sqrt(math1.Distance2BetweenPoints(p1, p2))

            line = vtk.vtkLineSource()
            line.SetPoint1(p1)
            line.SetPoint2(p2)
            line.SetResolution( int(distance//spacing) )        # Number of samples obtained along that probe line

            probe_spacing = (distance/line.GetResolution())

            #   Samples image data along the defined line
            probe_line = vtk.vtkProbeFilter()
            probe_line.SetInputConnection(line.GetOutputPort())
            probe_line.SetSourceConnection(image_points.GetOutputPort())
            probe_line.Update()

             #   Default thickness placeholder
            thickness = -1

            #   Try map cortical thickness at each point on the surface
            #   The cortical density estimate is fixed to the newly estimated cortical density "y1_estimate"
            #   this is handled within the find_thickness function and controlled using the density_bool argument
            try:
                thickness, background, apparent_density, data, point_number_refined, intensity_vector_refined, model_fit, p_i, res2 =\
                find_thickness(probe_line, probe_spacing, trabecular_std = args.trab_std, \
                            LAG = LAG, THRESHOLD = THRESHOLD, INFLUENCE = INFLUENCE,\
                                density_bool = 1, apparent_density = y1_estimate, idx = idx, p1 = 0, p2 = 0, pSurf = 0)
            except:
                thickness = -1

            if thickness > 0:
                #   Assign CBM measures to each point on the surface model
                surf1_mesh_modified['Thickness'][idx] = thickness
                surf1_mesh_diff['Thickness difference'][idx] = surf1_mesh_modified['Thickness'][idx] - surf1_mesh['Thickness'][idx]

            #   Update the terminal progress bar after each measurement
            time.sleep(0.1)
            printProgressBar(idx + 1, n, prefix = 'Cortical Mapping:', suffix = 'Complete', length = 50)

    ''' Plot bone surface models with cortical thicknes estimates mapped across the surface'''
    p = pv.Plotter(shape=(1,3))
    sargs = dict(color = "black", bold = True)
    #   Visualizes  the cortical thickness measures obtained when the CBM model is constrained using the maximum density in the sample profile
    p.subplot(0,0)
    p.add_mesh(surf1_mesh, show_edges=True, scalars="Thickness", clim=[0,4], cmap="RdYlBl", smooth_shading=True, scalar_bar_args = sargs)
    light = pv.Light()
    light.intensity = 2.0
    p.background_color = "white"

    #   Visualizes  the cortical thickness measures obtained when the CBM model is constrained using the cortical density estimation
    #   this will be either the hybrid-derived thickness or constant density derived thickness - depending on original choice
    p.subplot(0,1)
    p.add_mesh(surf1_mesh_modified, show_edges=True, scalars="Thickness", clim=[0,4], cmap="RdYlBl", smooth_shading=True, scalar_bar_args = sargs)
    light = pv.Light()
    light.intensity = 2.0
    p.background_color = "white"

    #   Visualizes the difference between cortical thickness measures in the first two plots
    p.subplot(0,2)
    p.add_mesh(surf1_mesh_diff, show_edges=True, scalars="Thickness difference", clim=[-2,2], cmap="RdYlBl", smooth_shading=True, scalar_bar_args = sargs)
    light = pv.Light()
    light.intensity = 2.0
    p.background_color = "white"

    p.show()

    ''' Save the CBM results as surface meshes with thickness measures'''
    vtk_filename = os.path.splitext(os.path.basename(args.input_image))[0] + "_CBM.vtk"
    surf1_mesh.save(os.path.join( os.path.dirname(args.input_image),vtk_filename))

    if args.method == 1:
        vtk_filename = os.path.splitext(os.path.basename(args.input_image))[0] + "_CBM_Hybrid.vtk"
    else:
        vtk_filename = os.path.splitext(os.path.basename(args.input_image))[0] + "_CBM_Constant.vtk"
    surf1_mesh_modified.save(os.path.join( os.path.dirname(args.input_image),vtk_filename))

    if args.method == 1:
        vtk_filename = os.path.splitext(os.path.basename(args.input_image))[0] + "_CBM_Hydrib_Diff.vtk"
    else:
        vtk_filename = os.path.splitext(os.path.basename(args.input_image))[0] + "_CBM_Constant_Diff.vtk"
    surf1_mesh_diff.save(os.path.join( os.path.dirname(args.input_image),vtk_filename))
