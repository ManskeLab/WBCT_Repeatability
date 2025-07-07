''' Open joint space width .vtk file in pyVista an visualize '''

import os
import sys
import pyvista as pv
import scipy
import argparse
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from jsm_util import smooth_surface_scalars
import matplotlib.pyplot as plt

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_obj", type = str, help = "The JSW.ply file (path + filename)")
    parser.add_argument("smooth_option", type = str, help = "Whether or not smoothing should be performed")
    parser.add_argument("save_patch",  type = str, nargs='?', default = "yes", help = "Whether or not to save the joint space width patch only")
    parser.add_argument("plot",  type = str, nargs='?', default = "yes", help = "Whether or not to plot the jsw distribution")
    args = parser.parse_args()

    obj = pv.read(args.input_obj)

    jsm_original = obj.threshold(value =(0.0,20.0),scalars = 'distance').extract_surface()
    jsm_shell = obj.threshold(value =(0.0,20.0),invert = True, scalars = 'distance').extract_surface()

    if args.smooth_option == "s":
        smooth = True
    else:
        smooth = False

    if smooth:
        jsm = smooth_surface_scalars(jsm_original, 2)
    else:
        jsm = jsm_original

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

    if "TFJ" in args.input_obj:

        if args.save_patch != "no":
            vtk_filename = os.path.splitext(os.path.basename(args.input_obj))[0] + "_PATCH.vtk"
            mapped_surface = pv.PolyData(jsm)
            mapped_surface["distance"] = jsm.GetPointData().GetScalars()
            mapped_surface.save(os.path.join( os.path.dirname(args.input_obj),vtk_filename))

        jsm_1 = vtk.vtkPolyDataConnectivityFilter()
        jsm_1.SetInputData(jsm)
        jsm_1.SetExtractionModeToSpecifiedRegions()
        jsm_1.AddSpecifiedRegion(sortRegion[-1])
        jsm_1.Update()

        jsm_2 = vtk.vtkPolyDataConnectivityFilter()
        jsm_2.SetInputData(jsm)
        jsm_2.SetExtractionModeToSpecifiedRegions()
        jsm_2.AddSpecifiedRegion(sortRegion[-2])
        jsm_2.Update()

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

        print(f'    Compartment 1 5%-ile JSW: {np.percentile(compartment1_JSW_data, 5)}')
        print(f'    Compartment 1 15%-ile JSW: {np.percentile(compartment1_JSW_data, 15)}')
        print(f'    Compartment 1 25%-ile JSW: {np.percentile(compartment1_JSW_data, 25)}')

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

        print(f'    Compartment 2 5%-ile JSW: {np.percentile(compartment2_JSW_data, 5)}')
        print(f'    Compartment 2 15%-ile JSW: {np.percentile(compartment2_JSW_data, 15)}')
        print(f'    Compartment 2 25%-ile JSW: {np.percentile(compartment2_JSW_data, 25)}')

        ''' Whole joint JSW metrics'''

        jsm_total_data = vtk_to_numpy(jsm_connect.GetOutput().GetPointData().GetScalars())
        jsm_total_data_filtered = jsm_total_data[~np.isnan(jsm_total_data)]     # filter data to remove any NaN values

        print(f'\n  Total mean JSW: {np.mean(jsm_total_data_filtered)}')
        print(f'    Total std JSW: {np.std(jsm_total_data_filtered)}')
        print(f'    Total min JSW: {np.min(jsm_total_data_filtered)}')
        print(f'    Total max JSW: {np.max(jsm_total_data_filtered)}')
        print(f'\n  Medial/Lateral Ratio: {(np.mean(compartment1_JSW_data)/np.mean(compartment2_JSW_data))}')

        print(f'    Total 5%-ile JSW: {np.percentile(jsm_total_data_filtered, 5)}')
        print(f'    Total 15%-ile JSW: {np.percentile(jsm_total_data_filtered, 15)}')
        print(f'    Total 25%-ile JSW: {np.percentile(jsm_total_data_filtered, 25)}')

        if args.plot == "yes":
            bin_num_total = int((np.max(jsm_total_data_filtered)-np.min(jsm_total_data_filtered))//0.2)
            bin_num_C1 = int((np.max(compartment1_JSW_data)-np.min(compartment1_JSW_data))//0.2)
            bin_num_C2 = int((np.max(compartment2_JSW_data)-np.min(compartment2_JSW_data))//0.2)
            ind = np.percentile(jsm_total_data_filtered, 25)

            x_t, y_t, _ = plt.hist(jsm_total_data_filtered,histtype='step', bins = bin_num_total, range = [jsm_total_data_filtered.min(),jsm_total_data_filtered.max()],
                              density = True, color = 'k', alpha = 1.0, label = "Whole Joint")
            x_a, y_a,_ = plt.hist(compartment1_JSW_data,histtype='stepfilled', bins = bin_num_C1, range = [compartment1_JSW_data.min(),compartment1_JSW_data.max()],
                      density = True, facecolor = 'r', alpha = 0.3,label = "Compartment A")
            x_b, y_b, _ = plt.hist(compartment2_JSW_data,histtype='stepfilled', bins = bin_num_C2, range = [compartment2_JSW_data.min(),compartment2_JSW_data.max()],
                     density = True, facecolor = 'b', alpha = 0.3, label = "Compartment B")
            plt.vlines(ind,0,np.max([x_t.max(), x_a.max(),x_b.max()]), linestyles='dotted',colors="k", label = "Whole Joint 25th percentile JSW")
            plt.title(os.path.splitext(os.path.basename(args.input_obj))[0])
            plt.legend()
            font = {'family': 'sans',
            'color':  'black',
            'weight': 'normal',
            'size': 10,
            }
            plt.xlabel("Joint space width (mm)",fontdict=font )
            plt.ylabel("Frequency (p.d.f.)",fontdict=font )
            plt.text(ind-1,np.max([x_t.max(), x_a.max(),x_b.max()]),f'{str(round(ind,2))} mm')
            plot_name = os.path.splitext(os.path.basename(args.input_obj))[0] + "_JSW_Distribution.png"
            plt.savefig((os.path.join( os.path.dirname(args.input_obj),plot_name)))
            plt.show(block = True)

        string_base1 = "Compartment 1 JSW_Mean: "
        string1 = string_base1 + str(np.mean(compartment1_JSW_data))

        string_base2 = "Compartment 2 JSW_Mean: "
        string2 = string_base2 + str(np.mean(compartment2_JSW_data))

        string = string1 + "\n" + string2

        sargs = dict(title = ' JSW (mm)',italic = False, title_font_size = 25, label_font_size = 20, vertical=True, position_y = 0.25, color = 'black', bold = True)
        sargs_img = dict(title = ' JSW (mm)',italic = False, title_font_size = 25, label_font_size = 20, vertical=True, position_y = 0.25, color = 'black', bold = True)

        # Generate rendering for figures
        fig = pv.Plotter()
        mapped_surface = pv.PolyData(jsm)
        mapped_surface["distance"] = jsm.GetPointData().GetScalars()
        fig.add_mesh(mapped_surface, show_edges=True,edge_opacity = 0.25, scalars ='distance', clim=[1, 10], cmap='turbo_r', smooth_shading=True, scalar_bar_args = sargs_img)
        # fig.add_mesh(jsm_shell, show_edges=True, edge_opacity = 0.25, scalars = None, nan_color="seashell", smooth_shading=True, show_scalar_bar=False)
        fig.add_mesh(jsm_shell, show_edges=True, edge_opacity = 0.25, scalars = None, nan_color="seashell", smooth_shading=True, show_scalar_bar=False)
        # fig.set_background('black')
        fig.set_background('white')
        fig.view_yx()
        fig.set_viewup([0,-0.6,0])
        # fig.set_background('dimgrey')
        fig.show()

        p = pv.Plotter(shape=(2,2))
        p.subplot(0,0)
        p.add_mesh(obj, smooth_shading=True,nan_color="seashell", scalars = 'distance',clim=[1,10], cmap='turbo_r', show_edges=False, scalar_bar_args = sargs)
        # p.add_mesh(obj, smooth_shading=True,scalars = 'distance',clim=[2,12], cmap='RdYlBu', show_edges=True)
        p.add_text(f'Mapped area: {obj.area}',color = 'black', position = "upper_left", viewport = True, font_size = 12)
        p.view_yx()
        p.set_viewup([0,-0.6,0])
        p.set_background('white')

        p.subplot(0,1)
        mapped_surface = pv.PolyData(jsm)
        mapped_surface["distance"] = jsm.GetPointData().GetScalars()
        p.add_mesh(mapped_surface, show_edges=True,edge_opacity = 0.25, scalars ='distance', clim=[1, 10], cmap='turbo_r', smooth_shading=True)
        p.add_mesh(jsm_shell, show_edges=True, edge_opacity = 0.25, scalars ='distance', nan_color="seashell", smooth_shading=True)
        p.add_text(string + f'\nMapped area: {mapped_surface.area}',color = 'black', position = "upper_left", viewport = True, font_size = 12)
        # p.add_mesh(surf3_mesh, opacity = 0.5, smooth_shading=True, color='#F9F6EE', show_edges=True)
        p.subplot(1,0)
        mapped_surface1 = pv.PolyData(jsm_1.GetOutput())
        mapped_surface1["distance"] = jsm_1.GetOutput().GetPointData().GetScalars()
        # p.add_mesh(mapped_surface1, scalars = "distance", clim=[3, 12], cmap='gist_rainbow', smooth_shading=True)
        p.add_mesh(mapped_surface1, scalars = "distance", clim=[1, 10], cmap='turbo_r', smooth_shading=True)
        # p.add_mesh(mapped_surface1, scalars = "distance", clim=[2, 8], cmap='RdYlBu', smooth_shading=True)
        p.add_text(string1 + f'\nMapped area: {mapped_surface1.area}',color = 'black', position = "upper_left", viewport = True, font_size = 12)

        p.subplot(1,1)
        mapped_surface2 = pv.PolyData(jsm_2.GetOutput())
        mapped_surface2["distance"] = jsm_2.GetOutput().GetPointData().GetScalars()
        p.add_mesh(mapped_surface2, scalars = "distance", clim=[1, 10], cmap='turbo_r', smooth_shading=True)
        # p.add_mesh(mapped_surface2, scalars = "distance", clim=[2, 8], cmap='RdYlBu', smooth_shading=True)
        p.add_text(string2 + f'\nMapped area: {mapped_surface2.area}',color = 'black', position = "upper_left", viewport = True, font_size = 12)
        p.show()

        # print(jsm.n_points)
        print(compartment1_JSW_data.size)
        print(compartment2_JSW_data.size)

    elif "PFJ" in args.input_obj:

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

        string_base1 = "Compartment 1 JSW_Mean: "
        string1 = string_base1 + str(np.mean(compartment1_JSW_data))

        string = string1

        p = pv.Plotter(shape=(1,3))
        # p.background_color = "white"
        p.subplot(0,0)
        p.add_mesh(obj, smooth_shading=True,scalars = 'distance',clim=[1,10], cmap='gist_rainbow', show_edges=True)

        p.subplot(0,1)
        p.add_mesh(jsm, show_edges=True, scalars ='distance', clim=[1, 10], cmap='gist_rainbow', smooth_shading=True)
        p.add_text(string, position = "upper_left", viewport = True, font_size = 12)

        p.subplot(0,2)
        p.add_mesh(jsm_1.GetOutput(), show_edges=False, scalars ='distance', clim=[1, 10], cmap='gist_rainbow', smooth_shading=True)
        p.add_text(string, position = "upper_left", viewport = True, font_size = 12)

        p.show()
