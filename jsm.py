# -----------------------------------------------------
# jsm.py
#
# Created by:   Tadiwa Waungana
# Created on:   24 February, 2023
# Modified on:  9 October, 2024
#
# Description: Estimate joint space width using grayscale image and bone meshes.
#              Firstly, the greyscale image is smoothed to remove gaussian noise.
#              Surface normal rays are cast from a source bone mesh towards a
#              target bone mesh. The image grayscale data is sampled during the ray
#              casting process. Joint space width is estimated using methods
#              described by Turmezei et al., 2021.
#
# -----------------------------------------------------
# Usage: python jsm.py <input_image.nii> <source_bone_mesh.vtk> <target_bone_mesh.vtk> <joint> <trab_std> <ray_length> <save_patch>
#
#
# Inputs:
# 	1. input image (.nii)
#   2. source bone mesh (.vtk)
#   3. target bone mesh (.vtk)
#   4. joint (tibiofemoral --> tib or tfj, patellofemoral --> pat or pfj)
#   5. trab_std (standard deviation of tibial trabecular grayscale values in a 10x10x10 voi)
#   6. ray_length (length of normal rays used to sample grayscale image)
#   7. save_patch (save copy of the joint space patch)
#
# Outputs: (Written to the directory that holds the greyscale image files by default)
# 	1. Source bone surface model with joint space width distribution (.vtk)
#   2. Optional: joint space surface patch (cut-out from source bone mesh)(.vtk)
#
# -----------------------------------------------------

import os
import vtk
import argparse
import math
from pyacvd import Clustering
import pyvista as pv
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib.pyplot as plt

#----------------------------Helper functions----------------------------------#

from jsm_util import gaussian_filter,compute_normals, cell_centers, intersect, intersect_test, find_distance, check_normals

#------------------------------------------------------------------------------#
import time
start_time = time.time()

if __name__ =="__main__":

    ''' Parameters for the peak detection algorithm in joint space mapping:
     Lag: The number of sample voxels values used to determine the moving average of signal intensity - default (5)
     Threshold: Scale factor to determine when a signal peak/cortical edge has been detected - default (2.0)
     Influence: How much each voxel value affects the moving average - default (0.5) '''
    LAG = 5
    THRESHOLD = 2.0
    INFLUENCE = 0.5

    # parse in necessary arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type = str, help = "The grayscale input image (path + filename)")
    parser.add_argument("input_source_bone_surface", type = str, help = "The input source surface file (path + filename)")
    parser.add_argument("input_target_bone_surface", type = str, help = "The input target surface (path + filename)")
    parser.add_argument("joint", type = str, help = "The joint to be analyzed: tibiofemoral -> tib or tfj, patellofemoral -> pat or pfj, both -> all")
    parser.add_argument("trab_std", type = float, nargs='?', default = 50, help = "The standard deviation of the tibial trabecular bone")
    parser.add_argument("ray_length", type = float, nargs='?', default = 10, help = "The length of the initial search vector")
    parser.add_argument("save_patch",  type = str, nargs='?', default = "yes", help = "Whether or not to save the joint space width patch only")
    args = parser.parse_args()

    ''' Sampling length: length of the probe line in coordinate space - default (10)
        ray_length: length of the ray/search vector vector in coordinate space - default (10)'''
    sampling_length = 10
    ray_length = args.ray_length

    ''' Read grayscale image as nifti (.nii)'''
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(args.input_image)
    reader.Update()

    '''Obtain image voxel spacing and origin'''
    spacing = reader.GetOutput().GetSpacing()[1]
    origin = reader.GetOutput().GetOrigin()

    ''' Gaussian filter grayscale image'''
    gaussian = gaussian_filter(reader)

    ''' Read pre-processed bone meshes using PyVista '''

    if args.joint == 'tib' or args.joint == 'tfj' or args.joint == 'pat' or args.joint == 'pfj':

        source_mesh = pv.read(args.input_source_bone_surface)
        target_mesh = pv.read(args.input_target_bone_surface)

        #   Incase scalar for jsw estimates wasn't pre-set during mesh generation
        source_mesh['distance'] = np.empty(source_mesh.n_points)
        source_mesh['distance'][:] = np.nan

        # ------------------------------------------------------------------------------ #
        #   Uncomment below: visualize the source and target bone meshes in 
        #   their functional position
        # ------------------------------------------------------------------------------ #
        # p1 = pv.Plotter()
        # p1.add_mesh(surf1_mesh, color = 'seashell', show_edges = False)
        # p1.add_mesh(surf3_mesh, color = 'seashell', show_edges = False)
        # p1.set_background('white')
        # p1.show()

        # ------------------------------------------------------------------------------ #
        #   Create new object for target mesh - used when searching for intersecting rays 
        #   using to estimate JSW
        # ------------------------------------------------------------------------------ #
        ''' Sets the object to look for during mapping: Normal rays are cast from the source surface 
           (e.g., femur) towards the target surface (e.g., tibia) '''
        obbObject_target = vtk.vtkOBBTree()
        obbObject_target.SetDataSet(target_mesh)
        obbObject_target.BuildLocator()

        ''' Compute normal rays used to search for opposing bone surfaces:
        Default is to compute normals from the source surface model i.e., surf3_mesh '''
        print('\n   Extracting normals...')
        source_mesh_normals = compute_normals(source_mesh)

        ''' Obtain the cells associated with each normal ray'''
        source_mesh_cells = cell_centers(source_mesh_normals)

        source_mesh_normals_data = source_mesh_normals.GetOutput().GetPointData().GetArray("Normals")      # default is surf1_normals


    ''' Sets the object to look for during mapping:
        Normal rays are cast from the surface of surf3 (i.e., femur) towards either of the tibia
        or patella -> obbObject_tibia and obbObject_patella respectively '''

    valid_Normals = vtk.vtkCellArray()
    pts = vtk.vtkPoints()

    '''Convert original image to a pointset so we can probe the grayscale image data using the extracted surface normals'''
    image_grayscale_to_points = vtk.vtkImageDataToPointSet()
    image_grayscale_to_points.SetInputConnection(gaussian.GetOutputPort())
    image_grayscale_to_points.Update()

    ''' Create a new image which will be built using the measured joint space width scalar values'''
    # Can be used if generating a contour image of the joint space bone surfaces
    new_surface_grid = vtk.vtkStructuredGrid()
    new_surface_grid.CopyStructure(image_grayscale_to_points.GetOutput())

    new_point_values = vtk.vtkDoubleArray()
    new_point_values.SetNumberOfComponents(1)
    new_point_values.SetNumberOfValues(image_grayscale_to_points.GetOutput().GetPointData().GetNumberOfTuples())
    for arr_size in range (0,image_grayscale_to_points.GetOutput().GetPointData().GetNumberOfTuples()):
        new_point_values.SetValue(arr_size, -1)

    ''' Set the counter to track the number of sample locations on the femoral bone surface'''
    intersect_count = 0
    success_count = 0

    ''' Loop through all extracted normals to extract only those that
    intersect both the femoral and tibial surface for tibiofemoral jsm'''

    if args.joint == 'tib' or args.joint == 'tfj' or args.joint == 'pat' or args.joint == 'pfj':

        for idx in range(source_mesh.n_points):
            start = np.empty(3)
            source_mesh.GetPoint(idx, start)
            source_mesh_normal = source_mesh_normals_data.GetTuple(idx)                               

            end = list(np.array(list(start)) + ray_length*np.array(list(source_mesh_normal)))      

            pts.InsertNextPoint(start)
            pts.InsertNextPoint(end)

            if intersect_test(obbObject_target, start, end):
                intersect_count += 1
                pointsInter, cellidsInter = intersect(obbObject_target, start, end)

                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, 2*idx)
                line.GetPointIds().SetId(1, 2*idx + 1)

                valid_Normals.InsertNextCell(line)

                pSurf = start
                #   Position the sample ray so that it sufficiently spans the entire joint space
                p1 = list(np.array(list(start)) - 0.5*sampling_length*np.array(list(source_mesh_normal)))      # defaults is '-0.5' and surf1Normal
                p2 = list(np.array(list(start)) + 1.0*sampling_length*np.array(list(source_mesh_normal)))      # default is '+1.5'and surf1Normal

                math1 = vtk.vtkMath()
                distance = math.sqrt(math1.Distance2BetweenPoints(p1, p2))

                line = vtk.vtkLineSource()                  # (x,y,z) = (sagittal,coronal,axial)
                line.SetPoint1(p1)
                line.SetPoint2(p2)
                line.SetResolution( int(distance//spacing) )                              # Number of samples obtained along that probe line

                probe_spacing = (distance/line.GetResolution())

                #   Samples image data along the defined line
                probe_line = vtk.vtkProbeFilter()                                      
                probe_line.SetInputConnection(line.GetOutputPort())
                probe_line.SetSourceConnection(image_grayscale_to_points.GetOutputPort())
                probe_line.Update()

                pSurf_id = new_surface_grid.FindPoint(pSurf)

                jsw, vector_refined, binary_signal, average_filter, point_refined = find_distance(probe_line, probe_spacing, trabecular_std = args.trab_std, LAG = LAG, THRESHOLD = THRESHOLD, INFLUENCE = INFLUENCE, idx = idx, p1 = 0, p2 = 0, pSurf = 0)
                if jsw > 0:
                    success_count += 1
                    source_mesh['distance'][idx] = jsw
                
                '''   Uncomment below to view example profile sampled across joint space    '''
                # if jsw > 12:
                #     plt.plot(point_refined, vector_refined, 'r-', point_refined, binary_signal, 'b--', point_refined, average_filter, 'g--')
                #     plt.title(f'surface point {idx}')
                #     plt.show()

                new_point_values.SetValue(pSurf_id, jsw)

    if args.joint == 'tib' or args.joint == 'tfj' or args.joint == 'pat' or args.joint == 'pfj':
        print(f'\n   Joint: {args.joint}')
        print(f'\n   Number of normals probed: {intersect_count}')
        print(f'\n   Number of succesful mappings: {success_count}')
        print('   Normal extraction: Complete')

    #   Store rays/normals that intersect both the source and target meshes
    valid_normals_poly = vtk.vtkPolyData()
    valid_normals_poly.SetPoints(pts)
    valid_normals_poly.SetLines(valid_Normals)
    pv_normals = pv.PolyData(valid_normals_poly)

    '''   Uncomment below to visualize the above normals    '''
    # p1 = pv.Plotter()
    # p1.add_mesh(target_mesh, color = 'seashell', show_edges = True)
    # p1.add_mesh(source_mesh, color = 'seashell', show_edges = True)
    # p1.add_mesh(pv_normals)
    # p1.add_mesh(target_mesh, opacity = 0.5)
    # p1.show()

    intersect_count = 0
    success_count = 0

    '''------ Extract joint space patches and Calculate JSW parameters ------'''

    if args.joint == 'tib' or args.joint == 'tfj':

        jsm = source_mesh.threshold(value =(0.0,20.0),scalars = 'distance').extract_surface()

        if args.save_patch == "yes":
            vtk_filename = os.path.splitext(os.path.basename(args.input_image))[0] + "_TFJ_JSW_PATCH.vtk"
            jsm.save(os.path.join( os.path.dirname(args.input_image),vtk_filename))

        jsm_connect = vtk.vtkPolyDataConnectivityFilter()
        jsm_connect.SetInputData(jsm)
        jsm_connect.SetExtractionModeToAllRegions()
        jsm_connect.Update()

        numberofregions = jsm_connect.GetNumberOfExtractedRegions()
        regionsize = jsm_connect.GetRegionSizes()
        regionID = np.zeros(numberofregions)
        for i in range(numberofregions):
            regionID[i] = regionsize.GetTuple(i)[0]

        sortRegion = np.argsort(regionID)

        jsm_1 = vtk.vtkPolyDataConnectivityFilter()
        jsm_1.SetInputData(jsm)
        jsm_1.SetExtractionModeToSpecifiedRegions()
        jsm_1.AddSpecifiedRegion(sortRegion[-1])
        jsm_1.Update()

        ''' Medial TF JSW metrics'''
        pointIDs1 = vtk.vtkIdList()
        surf1_cells = jsm_1.GetOutput().GetNumberOfCells()
        for cellindex in range(surf1_cells):
            pointIDCheck = vtk.vtkIdList()                                          # used as temp ID list to ensure we don't get duplicates
            jsm_1.GetOutput().GetCellPoints(cellindex, pointIDCheck)
            for i in range (0,pointIDCheck.GetNumberOfIds()):
                pointIDs1.InsertUniqueId(pointIDCheck.GetId(i))

        surf1_SCALARS = jsm_1.GetOutput().GetPointData().GetScalars()
        #surf1_SCALARS = surface1_smoothed.GetPointData().GetScalars()
        s1_SCALAR_ARRAY = vtk.vtkFloatArray()
        s1_SCALAR_ARRAY.SetNumberOfComponents(1)
        s1_SCALAR_ARRAY.SetNumberOfTuples(pointIDs1.GetNumberOfIds())
        surf1_SCALARS.GetTuples(pointIDs1,s1_SCALAR_ARRAY)

        compartment1_JSW_data = vtk_to_numpy(s1_SCALAR_ARRAY)
        compartment1_JSW_data = compartment1_JSW_data[~np.isnan(compartment1_JSW_data)]
        print(f'\n  Compartment 1 mean JSW: {np.mean(compartment1_JSW_data)}')
        print(f'    Compartment 1 std JSW: {np.std(compartment1_JSW_data)}')
        print(f'    Compartment 1 min JSW: {np.min(compartment1_JSW_data)}')
        print(f'    Compartment 1 max JSW: {np.max(compartment1_JSW_data)}')

        jsm_2 = vtk.vtkPolyDataConnectivityFilter()
        jsm_2.SetInputData(jsm)
        jsm_2.SetExtractionModeToSpecifiedRegions()
        jsm_2.AddSpecifiedRegion(sortRegion[-2])
        jsm_2.Update()

        '''Lateral TF JSW metrics'''
        pointIDs2 = vtk.vtkIdList()
        surf2_cells = jsm_2.GetOutput().GetNumberOfCells()
        for cellindex in range(surf2_cells):
            pointIDCheck = vtk.vtkIdList()                                          # used as temp ID list to ensure we don't get duplicates
            jsm_2.GetOutput().GetCellPoints(cellindex, pointIDCheck)
            for i in range (0,pointIDCheck.GetNumberOfIds()):
                pointIDs2.InsertUniqueId(pointIDCheck.GetId(i))

        surf2_SCALARS = jsm_2.GetOutput().GetPointData().GetScalars()
        s2_SCALAR_ARRAY = vtk.vtkFloatArray()
        s2_SCALAR_ARRAY.SetNumberOfComponents(1)
        s2_SCALAR_ARRAY.SetNumberOfTuples(pointIDs2.GetNumberOfIds())
        surf2_SCALARS.GetTuples(pointIDs2,s2_SCALAR_ARRAY)

        compartment2_JSW_data = vtk_to_numpy(s2_SCALAR_ARRAY)
        compartment2_JSW_data = compartment2_JSW_data[~np.isnan(compartment2_JSW_data)]
        print(f'\n  Compartment 2 mean JSW: {np.mean(compartment2_JSW_data)}')
        print(f'    Compartment 2 std JSW: {np.std(compartment2_JSW_data)}')
        print(f'    Compartment 2 min JSW: {np.min(compartment2_JSW_data)}')
        print(f'    Compartment 2 max JSW: {np.max(compartment2_JSW_data)}')

        ''' Whole joint JSW metrics'''

        jsm_total_data = vtk_to_numpy(jsm_connect.GetOutput().GetPointData().GetScalars())
        jsm_total_data_filtered = jsm_total_data[~np.isnan(jsm_total_data)]     # filter data to remove any NaN values

        print(f'\n  Total mean JSW: {np.mean(jsm_total_data_filtered)}')
        print(f'    Total std JSW: {np.std(jsm_total_data_filtered)}')
        print(f'    Total min JSW: {np.min(jsm_total_data_filtered)}')
        print(f'    Total max JSW: {np.max(jsm_total_data_filtered)}')
        print(f'\n  Medial/Lateral Ratio: {(np.mean(compartment1_JSW_data)/np.mean(compartment2_JSW_data))}')

        '''-------- Final visualization ---------------'''
        string_base1 = "Compartment 1 JSW_Mean: "
        string1 = string_base1 + str(np.mean(compartment1_JSW_data))

        string_base2 = "Compartment 2 JSW_Mean: "
        string2 = string_base2 + str(np.mean(compartment2_JSW_data))

        string = string1 + "\n" + string2

        p = pv.Plotter(shape=(2,2))
        p.subplot(0,0)
        p.add_mesh(source_mesh, smooth_shading=True,scalars = 'distance',clim=[3,12], cmap='gist_rainbow', show_edges=True)

        p.subplot(0,1)
        p.add_mesh(jsm, show_edges=False, scalars ='distance', clim=[3, 12], cmap='gist_rainbow', smooth_shading=True)
        p.add_text(string, position = "upper_left", viewport = True, font_size = 12)

        p.subplot(1,0)
        mapped_surface1 = pv.PolyData(jsm_1.GetOutput())
        mapped_surface1["distance"] = jsm_1.GetOutput().GetPointData().GetScalars()
        p.add_mesh(jsm_1.GetOutput(), scalars = "distance", clim=[3, 12], cmap='gist_rainbow', smooth_shading=True)
        p.add_text(string1, position = "upper_left", viewport = True, font_size = 12)

        p.subplot(1,1)
        mapped_surface2 = pv.PolyData(jsm_2.GetOutput())
        mapped_surface2["distance"] = jsm_2.GetOutput().GetPointData().GetScalars()
        p.add_mesh(mapped_surface2, scalars = "distance", clim=[3, 12], cmap='gist_rainbow', smooth_shading=True)
        p.add_text(string2, position = "upper_left", viewport = True, font_size = 12)

        print("---------------- Runtime: %s --------------------- "%(time.time()-start_time))
        p.show()

        ''' Save obj of JSM'''

        vtk_filename = os.path.splitext(os.path.basename(args.input_image))[0] + "_TFJ_JSW.vtk"
        source_mesh.save(os.path.join( os.path.dirname(args.input_image),vtk_filename))

    if args.joint =='pat' or args.joint == 'pfj':

        jsm = source_mesh.threshold(value =(0.0,20.0),scalars = 'distance').extract_surface()

        jsm_connect = vtk.vtkPolyDataConnectivityFilter()
        jsm_connect.SetInputData(jsm)
        jsm_connect.SetExtractionModeToAllRegions()
        jsm_connect.Update()

        numberofregions = jsm_connect.GetNumberOfExtractedRegions()
        regionsize = jsm_connect.GetRegionSizes()
        regionID = np.zeros(numberofregions)
        for i in range(numberofregions):
            regionID[i] = regionsize.GetTuple(i)[0]

        sortRegion = np.argsort(regionID)

        jsm_1 = vtk.vtkPolyDataConnectivityFilter()
        jsm_1.SetInputData(jsm)
        jsm_1.SetExtractionModeToSpecifiedRegions()
        jsm_1.AddSpecifiedRegion(sortRegion[-1])
        jsm_1.Update()

        ''' PF JSW metrics'''
        pointIDs1 = vtk.vtkIdList()
        surf1_cells = jsm_1.GetOutput().GetNumberOfCells()
        for cellindex in range(surf1_cells):
            pointIDCheck = vtk.vtkIdList()                                          # used as temp ID list to ensure we don't get duplicates
            jsm_1.GetOutput().GetCellPoints(cellindex, pointIDCheck)
            for i in range (0,pointIDCheck.GetNumberOfIds()):
                pointIDs1.InsertUniqueId(pointIDCheck.GetId(i))

        surf1_SCALARS = jsm_1.GetOutput().GetPointData().GetScalars()
        #surf1_SCALARS = surface1_smoothed.GetPointData().GetScalars()
        s1_SCALAR_ARRAY = vtk.vtkFloatArray()
        s1_SCALAR_ARRAY.SetNumberOfComponents(1)
        s1_SCALAR_ARRAY.SetNumberOfTuples(pointIDs1.GetNumberOfIds())
        surf1_SCALARS.GetTuples(pointIDs1,s1_SCALAR_ARRAY)

        compartment1_JSW_data = vtk_to_numpy(s1_SCALAR_ARRAY)
        compartment1_JSW_data = compartment1_JSW_data[~np.isnan(compartment1_JSW_data)]
        print(f'\n  Compartment 1 mean JSW: {np.mean(compartment1_JSW_data)}')
        print(f'    Compartment 1 std JSW: {np.std(compartment1_JSW_data)}')
        print(f'    Compartment 1 min JSW: {np.min(compartment1_JSW_data)}')
        print(f'    Compartment 1 max JSW: {np.max(compartment1_JSW_data)}')

        '''-------- Final visualization ---------------'''
        string_base1 = "Compartment 1 JSW_Mean: "
        string1 = string_base1 + str(np.mean(compartment1_JSW_data))

        string = string1

        p = pv.Plotter(shape=(1,3))
        p.subplot(0,0)
        p.add_mesh(source_mesh, smooth_shading=True,scalars = 'distance',clim=[3,12], cmap='gist_rainbow', show_edges=True)

        p.subplot(0,1)
        p.add_mesh(jsm, show_edges=False, scalars ='distance', clim=[3, 12], cmap='gist_rainbow', smooth_shading=True)
        p.add_text(string, position = "upper_left", viewport = True, font_size = 12)

        p.subplot(0,2)
        p.add_mesh(jsm_1.GetOutput(), show_edges=False, scalars ='distance', clim=[3, 12], cmap='gist_rainbow', smooth_shading=True)
        p.add_text(string, position = "upper_left", viewport = True, font_size = 12)
        p.show()

        ''' Save obj of JSM'''

        vtk_filename = os.path.splitext(os.path.basename(args.input_image))[0] + "_PFJ_JSW.vtk"
        source_mesh.save(os.path.join( os.path.dirname(args.input_image),vtk_filename))
