''' Open joint space width .vtk file in pyVista an visualize '''

import os
import sys
import pyvista as pv
import argparse
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import pandas as pd
# from jsm_util import smooth_surface_scalars

def smooth_surface_scalars(surface = 0, itr = 0):
    ''' Convert pyvista polydata to a vtk polydata'''
    # surf = pv.PolyData(surface)
    # surf.plot()
    surface_points = surface.GetPoints()
    num_surface_points = surface_points.GetNumberOfPoints()

    ''' set number of smoothing iterations '''
    max_iter = itr

    point_cells = vtk.vtkIdList()   # stores the ids of the cells that a query point is connected to
    cell_points = vtk.vtkIdList()   #stores the ids of the points associated with the above points


    ''' create a copy of the surface that will carry the smoothed scalar values '''
    surface_smoothed = vtk.vtkPolyData()
    surface_smoothed.DeepCopy(surface)

    ''' create array to store the smoothed point scalar values '''
    smooth_scalars = vtk.vtkFloatArray()
    smooth_scalars.SetNumberOfTuples(num_surface_points)
    smooth_scalars.SetNumberOfComponents(1)
    smooth_scalars.SetName("Thickness")

    ''' create upward links from points to the cells that use each point'''
    surface.BuildLinks()

    for i in range(0, max_iter, 1):         # dictates how many times the entire surface is smoothed
        for pid in range(0, num_surface_points, 1):
            point_scalar = surface_smoothed.GetPointData().GetArray("Thickness").GetTuple1(pid)
            # if 800 < pid < 803:
            #     print(f'\nPoint {pid} has scalar value {point_scalar}')
            point_coords = [0] * 3
            surface.GetPoint(pid, point_coords)
            # if 800 < pid < 803:
            #     print(point_coords)
            ''' Get the cells connected to the current point '''
            surface.GetPointCells(pid, point_cells)
            num_cells = point_cells.GetNumberOfIds()

            # if 800 < pid < 803:
            #     print(f'\nPid = {pid}')
            #     print(f'numcells = {num_cells}')
            #     print(point_cells)

            ''' scalar vector to carry all the scalar values associated with smoothing the current point '''
            scalar_vector = np.zeros(1)
            scalar_vector[0] = point_scalar     # the scalar value of the current point

            for cell in range (0, num_cells, 1):
                neighbour_cell_id = point_cells.GetId(cell)
                ''' get the points associated with the current cell '''
                surface.GetCellPoints(neighbour_cell_id, cell_points)
                num_cell_points = cell_points.GetNumberOfIds()
                # if 800 < pid < 803:
                #     print(f'\nCell Id {neighbour_cell_id} has {num_cell_points} points')

                ''' loop through the neighbour points (nip) associated with each neighbouring cell '''
                for nip in range (0, num_cell_points, 1):
                    np_id = cell_points.GetId(nip)
                    np_coords = [0] * 3
                    surface.GetPoint(np_id, np_coords)
                    # if 800 < pid < 803:
                    #     print(np_coords)

                    ''' Get scalar value of the current neighbouring point '''
                    nip_scalar = surface_smoothed.GetPointData().GetScalars().GetTuple1(np_id)
                    # if 800 < pid < 803:
                    #     print("nip = ",nip_scalar)

                    ''' add the scalar value to the smoothing vector/equation '''
                    scalar_vector = np.append(scalar_vector, nip_scalar)
                    # if 800 < pid < 803:
                    #     print(scalar_vector)
                    # end the neighbouring points loop
                # end the neighbouring cells loop

            # if 800 < pid < 803:
            #     print(scalar_vector)
            original_point_scalar_index = np.where(scalar_vector == point_scalar)
            scalar_vector = np.delete(scalar_vector,original_point_scalar_index)
            scalar_vector = np.append(scalar_vector, point_scalar)
            scalar_vector = scalar_vector[~np.isnan(scalar_vector)]
            # if 800 < pid < 803:
            #     print(scalar_vector)
            smoothed_scalar = np.median(scalar_vector)

            ''' add the smoothed scalar value to the smoothed scalars array '''
            smooth_scalars.SetTuple1(pid, smoothed_scalar)
        surface_smoothed.GetPointData().SetScalars(smooth_scalars)

    ''' return the surface with smoothed scalar values '''
    print(f'Scalar smoothing complete with {itr} iterations')

    # surf_smooth = pv.PolyData(surface_smoothed)
    # surf_smooth.plot(scalars = "distance")

    return surface_smoothed

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_whole_bone_JSM", type = str, help = "The JSW.ply file (path + filename)")
    parser.add_argument("input_whole_bone_CBM", type = str, help = "The JSW.ply file (path + filename)")
    parser.add_argument("smooth_option", type = str, help = "Whether or not smoothing should be performed")
    parser.add_argument("save_patch",  type = str, nargs='?', default = "yes", help = "Whether or not to save the joint space width patch only (default is yes)")
    parser.add_argument("save_Excel",  type = str, nargs='?', default = "data", help = "Whether or not to save the SBP.Th metrics (default is yes)")
    args = parser.parse_args()

    bone_jsm = pv.read(args.input_whole_bone_JSM)
    bone_cbm_unfiltered = pv.read(args.input_whole_bone_CBM)
    bone_cbm_original = pv.read(args.input_whole_bone_CBM)
    bone_cbm = pv.read(args.input_whole_bone_CBM)

    mask1 = np.isnan(bone_jsm["distance"])
    bone_cbm["Thickness"][mask1] = np.nan
    bone_cbm_joint_surface = bone_cbm.threshold(value = 0.0, scalars = 'Thickness').extract_surface()
    cbm_shell = bone_cbm.threshold(value =(0.0,20.0),invert = True, scalars = 'Thickness').extract_surface()

    if args.smooth_option == "s":
        smooth = True
    else:
        smooth = False

    if smooth:
        bone_cbm_joint_surface = smooth_surface_scalars(bone_cbm_joint_surface, 4)
        bone_cbm_original = smooth_surface_scalars(bone_cbm_original, 4)

    cbm_connect = vtk.vtkPolyDataConnectivityFilter()
    cbm_connect.SetInputData(bone_cbm_joint_surface)
    cbm_connect.SetExtractionModeToAllRegions()
    cbm_connect.Update()

    numberofregions = cbm_connect.GetNumberOfExtractedRegions()
    regionsize = cbm_connect.GetRegionSizes()
    regionID = np.zeros(numberofregions)
    for i in range(numberofregions):
        regionID[i] = regionsize.GetTuple(i)[0]

    sortRegion = np.argsort(regionID)

    cbm_1 = vtk.vtkPolyDataConnectivityFilter()
    cbm_1.SetInputData(bone_cbm_joint_surface)
    cbm_1.SetExtractionModeToSpecifiedRegions()
    cbm_1.AddSpecifiedRegion(sortRegion[-1])
    cbm_1.Update()

    cbm_2 = vtk.vtkPolyDataConnectivityFilter()
    cbm_2.SetInputData(bone_cbm_joint_surface)
    cbm_2.SetExtractionModeToSpecifiedRegions()
    cbm_2.AddSpecifiedRegion(sortRegion[-2])
    cbm_2.Update()

    ''' Medial TF JSW metrics'''
    pointIDs1 = vtk.vtkIdList()
    surf1_cells = cbm_1.GetOutput().GetNumberOfCells()
    for cellindex in range(surf1_cells):
        pointIDCheck = vtk.vtkIdList()                                          # used as temp ID list to ensure we don't get duplicates
        cbm_1.GetOutput().GetCellPoints(cellindex, pointIDCheck)
        for i in range (0,pointIDCheck.GetNumberOfIds()):
            pointIDs1.InsertUniqueId(pointIDCheck.GetId(i))

    surf1_SCALARS = cbm_1.GetOutput().GetPointData().GetScalars()
    #surf1_SCALARS = surface1_smoothed.GetPointData().GetScalars()
    s1_SCALAR_ARRAY = vtk.vtkFloatArray()
    s1_SCALAR_ARRAY.SetNumberOfComponents(1)
    s1_SCALAR_ARRAY.SetNumberOfTuples(pointIDs1.GetNumberOfIds())
    surf1_SCALARS.GetTuples(pointIDs1,s1_SCALAR_ARRAY)

    compartment1_CBM_data = vtk_to_numpy(s1_SCALAR_ARRAY)
    compartment1_CBM_data = compartment1_CBM_data[~np.isnan(compartment1_CBM_data)]
    print(f'\n  Compartment 1 mean SCB.Th: {np.mean(compartment1_CBM_data)}')
    print(f'    Compartment 1 std SCB.Th: {np.std(compartment1_CBM_data)}')
    print(f'    Compartment 1 min SCB.Th: {np.min(compartment1_CBM_data)}')
    print(f'    Compartment 1 max SCB.Th: {np.max(compartment1_CBM_data)}')


    '''Lateral TF JSW metrics'''
    pointIDs2 = vtk.vtkIdList()
    surf2_cells = cbm_2.GetOutput().GetNumberOfCells()
    for cellindex in range(surf2_cells):
        pointIDCheck = vtk.vtkIdList()                                          # used as temp ID list to ensure we don't get duplicates
        cbm_2.GetOutput().GetCellPoints(cellindex, pointIDCheck)
        for i in range (0,pointIDCheck.GetNumberOfIds()):
            pointIDs2.InsertUniqueId(pointIDCheck.GetId(i))

    surf2_SCALARS = cbm_2.GetOutput().GetPointData().GetScalars()
    s2_SCALAR_ARRAY = vtk.vtkFloatArray()
    s2_SCALAR_ARRAY.SetNumberOfComponents(1)
    s2_SCALAR_ARRAY.SetNumberOfTuples(pointIDs2.GetNumberOfIds())
    surf2_SCALARS.GetTuples(pointIDs2,s2_SCALAR_ARRAY)

    compartment2_CBM_data = vtk_to_numpy(s2_SCALAR_ARRAY)
    compartment2_CBM_data = compartment2_CBM_data[~np.isnan(compartment2_CBM_data)]
    print(f'\n  Compartment 2 mean SCB.Th: {np.mean(compartment2_CBM_data)}')
    print(f'    Compartment 2 std SCB.Th: {np.std(compartment2_CBM_data)}')
    print(f'    Compartment 2 min SCB.Th: {np.min(compartment2_CBM_data)}')
    print(f'    Compartment 2 max SCB.Th: {np.max(compartment2_CBM_data)}')

    ''' Whole joint JSW metrics'''

    cbm_total_data = vtk_to_numpy(cbm_connect.GetOutput().GetPointData().GetScalars())
    cbm_total_data_filtered = cbm_total_data[~np.isnan(cbm_total_data)]     # filter data to remove any NaN values

    print(f'\n  Total mean SCB.Th: {np.mean(cbm_total_data_filtered)}')
    print(f'    Total std SCB.Th: {np.std(cbm_total_data_filtered)}')
    print(f'    Total min SCB.Th: {np.min(cbm_total_data_filtered)}')
    print(f'    Total max SCB.Th: {np.max(cbm_total_data_filtered)}')
    print(f'\n  Medial/Lateral Ratio: {(np.mean(compartment1_CBM_data)/np.mean(compartment2_CBM_data))}')

    data = pd.DataFrame({'Compartment 1 mean SBP.Th':[np.mean(compartment1_CBM_data)],
                            'Compartment 1 median SBP.Th':[np.median(compartment1_CBM_data)],
                            'Compartment 1 95-ile':[np.percentile(compartment1_CBM_data, 95)],
                            'Compartment 1 85-ile':[np.percentile(compartment1_CBM_data, 85)],
                            'Compartment 1 75-ile':[np.percentile(compartment1_CBM_data, 75)],
                            'Compartment 1 std SBP.Th':[np.std(compartment1_CBM_data)],
                            'Compartment 1 min SBP.Th':[np.min(compartment1_CBM_data)],
                            'Compartment 1 max SBP.Th':[np.max(compartment1_CBM_data)],
                            'Compartment 2 mean SBP.Th':[np.mean(compartment2_CBM_data)],
                            'Compartment 2 median SBP.Th':[np.median(compartment2_CBM_data)],
                            'Compartment 2 95-ile':[np.percentile(compartment2_CBM_data, 95)],
                            'Compartment 2 85-ile':[np.percentile(compartment2_CBM_data, 85)],
                            'Compartment 2 75-ile':[np.percentile(compartment2_CBM_data, 75)],
                            'Compartment 2 std SBP.Th':[np.std(compartment2_CBM_data)],
                            'Compartment 2 min SBP.Th':[np.min(compartment2_CBM_data)],
                            'Compartment 2 max SBP.Th':[np.max(compartment2_CBM_data)],
                            'Joint mean SBP.Th':[np.mean(cbm_total_data_filtered)],
                            'Joint median SBP.Th':[np.median(cbm_total_data_filtered)],
                            'Joint 95-ile':[np.percentile(cbm_total_data_filtered, 95)],
                            'Joint 85-ile':[np.percentile(cbm_total_data_filtered, 85)],
                            'Joint 75-ile':[np.percentile(cbm_total_data_filtered, 75)],
                            'Joint std SBP.Th':[np.std(cbm_total_data_filtered)],
                            'Joint min SBP.Th':[np.min(cbm_total_data_filtered)],
                            'Joint max SBP.Th':[np.max(cbm_total_data_filtered)]})

    if args.save_Excel == "data":
        data_name = 'CBM_DATA_' + os.path.splitext(os.path.basename(args.input_whole_bone_CBM))[0] + '.csv'
        data.to_csv(os.path.join( os.path.dirname(args.input_whole_bone_CBM),data_name) ,index=False)

    string_base1 = "Compartment 1 CBM_Mean: "
    string1 = string_base1 + str(np.mean(compartment1_CBM_data))

    string_base2 = "Compartment 2 CBM_Mean: "
    string2 = string_base2 + str(np.mean(compartment2_CBM_data))

    string = string1 + "\n" + string2

    sargs = dict(title = ' SBP.Th (mm)',italic = False, title_font_size = 20, label_font_size = 15, vertical=True, position_y = 0.25, color = 'black', bold = True)
    sargs_img = dict(title = ' SBP.Th (mm)',italic = False, title_font_size = 25, label_font_size = 20, vertical=True, position_y = 0.25, color = 'black', bold = True)

    fig = pv.Plotter()
    mapped_surface = pv.PolyData(bone_cbm_original)
    mapped_surface["Thickness"] = bone_cbm_original.GetPointData().GetScalars()
    fig.add_mesh(mapped_surface, show_edges=True,edge_opacity = 0.25, scalars ='Thickness', clim=[0, 2.0], cmap='Spectral', smooth_shading=True, scalar_bar_args = sargs_img)
    fig.add_mesh(pv.PolyData(bone_cbm_joint_surface).extract_feature_edges(), color = "black", line_width=4)
    # fig.add_mesh(cbm_shell, show_edges=True, edge_opacity = 0.25, scalars = None, nan_color="seashell", smooth_shading=True, show_scalar_bar=False)
    fig.set_background('white')
    fig.show()

    # p = pv.Plotter(shape=(2,2))

    # mapped_surface_joint_surface = pv.PolyData(bone_cbm_joint_surface)
    # mapped_surface_joint_surface["Thickness"] = bone_cbm_joint_surface.GetPointData().GetScalars()
    # p.add_mesh(mapped_surface_joint_surface, scalar_bar_args = sargs, show_edges = False, scalars = "Thickness", clim=[0,0.5], cmap='RdYlBu', smooth_shading=True)
    # # p.add_mesh(surf3_mesh, opacity = 0.5, smooth_shading=True, color='#F9F6EE', show_edges=True)
    # bone_jsm['empty'] = np.empty(bone_cbm.n_points)
    # bone_jsm['empty'][:] = np.nan
    # bone_jsm['empty'][mask1] = 1
    # bone_jsm = bone_jsm.threshold(value = 0.0, scalars = 'empty').extract_surface()
    # bone_jsm['empty'][:] = np.nan
    # # bone_cbm['Empty'] = np.empty(bone_cbm.n_points)
    # p.add_mesh(bone_jsm ,scalars = None, color = None, smooth_shading=True)
    # p.remove_scalar_bar(title = 'distance')
    # p.view_yx()
    # p.set_viewup([0,-0.6,0])
    # p.set_background('white')
    # p.show()

        # Generate rendering for figures
    fig = pv.Plotter()
    mapped_surface = pv.PolyData(bone_cbm_joint_surface)
    mapped_surface["Thickness"] = bone_cbm_joint_surface.GetPointData().GetScalars()
    fig.add_mesh(mapped_surface, show_edges=True,edge_opacity = 0.25, scalars ='Thickness', clim=[0, 0.5], cmap='Spectral', smooth_shading=True, scalar_bar_args = sargs_img)
    fig.add_mesh(cbm_shell, show_edges=True, edge_opacity = 0.25, scalars = None, nan_color="seashell", smooth_shading=True, show_scalar_bar=False)
    fig.set_background('black')
    # fig.set_background('dimgray')
    fig.show()

    p = pv.Plotter(shape=(2,2))

    p.subplot(0,0)
    mapped_surface = pv.PolyData(bone_cbm_original)
    mapped_surface["Thickness"] = bone_cbm_original.GetPointData().GetScalars()
    # sargs = dict(title_font_size = 12, label_font_size = 10, fmt = "%.3f", color = "white")
    # p.add_mesh(mapped_surface, show_edges = True,scalars ='Thickness', scalar_bar_args = sargs, clim=[0,0.8], cmap='coolwarm_r', smooth_shading=True)
    p.add_mesh(mapped_surface, show_edges = False,scalars ='Thickness', scalar_bar_args = sargs, clim=[0, 0.5], cmap='RdYlBu', smooth_shading=True)
    # p.add_mesh(surf3_mesh, opacity = 0.5, smooth_shading=True, color='#F9F6EE', show_edges=True)
    p.view_yx()
    p.set_viewup([0,-0.6,0])
    p.set_background('white')

    p.subplot(0,1)
    # p.add_mesh(mapped_surface1, scalars = "distance", clim=[3, 12], cmap='gist_rainbow', smooth_shading=True)
    mapped_surface_joint_surface = pv.PolyData(bone_cbm_joint_surface)
    mapped_surface_joint_surface["Thickness"] = bone_cbm_joint_surface.GetPointData().GetScalars()
    p.add_mesh(mapped_surface_joint_surface, show_edges = False,scalars = "Thickness", clim=[0,0.5], cmap='RdYlBu', smooth_shading=True)
    p.add_text(string + f'\nMapped area: {mapped_surface_joint_surface.area}', position = "upper_left", viewport = True, font_size = 12)

    p.subplot(1,0)
    mapped_surface1 = pv.PolyData(cbm_1.GetOutput())
    mapped_surface1["distance"] = cbm_1.GetOutput().GetPointData().GetScalars()
    # p.add_mesh(mapped_surface1, scalars = "distance", clim=[3, 12], cmap='gist_rainbow', smooth_shading=True)
    p.add_mesh(mapped_surface1, scalars = "Thickness", clim=[0.0, 0.75], cmap='coolwarm_r', smooth_shading=True)
    p.add_text(string1 + f'\nMapped area: {mapped_surface1.area}', position = "upper_left", viewport = True, font_size = 12)

    p.subplot(1,1)
    mapped_surface2 = pv.PolyData(cbm_2.GetOutput())
    mapped_surface2["distance"] = cbm_2.GetOutput().GetPointData().GetScalars()
    p.add_mesh(mapped_surface2, scalars = "Thickness", clim=[0.0, 0.75], cmap='coolwarm_r', smooth_shading=True)
    p.add_text(string2 + f'\nMapped area: {mapped_surface2.area}', position = "upper_left", viewport = True, font_size = 12)

    if args.save_patch != "no":
        vtk_filename = os.path.splitext(os.path.basename(args.input_whole_bone_CBM))[0] + "_PATCH.vtk"
        mapped_surface_joint_surface.save(os.path.join( os.path.dirname(args.input_whole_bone_CBM),vtk_filename))

    # p.background_color = "white"
    p.show()
