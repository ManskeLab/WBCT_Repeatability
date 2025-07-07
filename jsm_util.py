# -----------------------------------------------------
# jsm_util.py
#
# Created by:   Tadiwa Waungana
# Created on:   6 July, 2023
#
# Description: Supporting/helper functions for joint space mapping.
# -----------------------------------------------------
# Usage: Helper functions are called directly from jsm.py
#
#
# -----------------------------------------------------
''' Supporting functions for joint space mapping'''
import vtk
from scipy.interpolate import CubicSpline
import numpy as np

#   Median image filter - salt/pepper noise 
def median_filter(image):
    #   ----------------------  #
    #  image is greyscale image
    #   ----------------------  #
    median = vtk.vtkImageMedian3D()
    median.SetInputConnection(image.GetOutputPort())
    median.SetKernelSize(9,9,9)
    median.Update()

    return median

#   Gaussian image filter - gaussian noise 
def gaussian_filter(image):
    #   ----------------------  #
    #  image is greyscale image
    #   ----------------------  #
    gaussian = vtk.vtkImageGaussianSmooth()
    gaussian.SetInputConnection(image.GetOutputPort())
    gaussian.SetStandardDeviations(1,1,1)
    gaussian.SetRadiusFactors(1,1,1)
    gaussian.Update()

    return gaussian

#   Compute surface normals of the bone surface models
def compute_normals(surface_polydata):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface_polydata)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.Update()

    return normals

#   Compute centers of the triangulated mesh elements 
def cell_centers(surface_polydata):
    surface_cell_centers = vtk.vtkCellCenters()
    surface_cell_centers.VertexCellsOn()
    surface_cell_centers.SetInputConnection(surface_polydata.GetOutputPort())
    surface_cell_centers.Update()

    return surface_cell_centers

#   Test is rays extending from one source model intersect the target mesh
def intersect_test(obbObject, start, end):
    test = obbObject.IntersectWithLine(start, end, None, None)
    if test == 0:
        return False
    return True

#   Store rays extending from one source model that intersect the target mesh
def intersect(obbObject, start, end):
    points = vtk.vtkPoints()
    cellids = vtk.vtkIdList()

    intersect = obbObject.IntersectWithLine(start, end, points, cellids)

    pointData = points.GetData()
    numberofpoints = pointData.GetNumberOfTuples()
    numberofcellids = cellids.GetNumberOfIds()

    pointsInter = []
    cellidsInter = []

    for idx in range(numberofpoints):
        pointsInter.append(pointData.GetTuple3(idx))
        cellidsInter.append(cellids.GetId(idx))

    return pointsInter, cellidsInter

#   Estimate the joint space width between surface meshes using variation/constrast in image greyscale values
def find_distance(probe_line, probe_spacing, trabecular_std = 0, LAG = 0, THRESHOLD = 0, INFLUENCE = 0, idx = 0, p1 = 0, p2 = 0, pSurf = 0):

    NumberOfSamples = probe_line.GetOutput().GetPointData().GetScalars().GetNumberOfTuples()
    point_number = np.zeros(NumberOfSamples)
    intensity_vector = np.zeros_like(point_number)

    for tuple_id in range( 0 , NumberOfSamples):
        intensity_vector[tuple_id] = np.array(list(probe_line.GetOutput().GetPointData().GetScalars().GetTuple(tuple_id)))
        point_number[tuple_id] = tuple_id * probe_spacing

    # cubic spline interpolation to enable sub-resolution measurements
    point_number_refined = np.linspace(point_number[0], point_number.max(), 100)
    cs = CubicSpline(point_number, intensity_vector)
    intensity_vector_refined = cs(point_number_refined)

    cortical_attenuation = intensity_vector_refined.max()
    trabecular_attenuation = np.percentile(intensity_vector_refined, 25)

    minimum_peak_requirement_JS_min = False
    minimum_peak_requirement_JS_diff = False
    peak_test = False
    jsw = -1

    joint_space_candidate = intensity_vector_refined.argmin()

    intensity_BinSignal = np.zeros_like(intensity_vector_refined , dtype = type(trabecular_attenuation))
    intensity_BinSignal[0:LAG+1] = trabecular_attenuation
    peak_count = np.zeros_like(intensity_vector_refined)                # peak start = 1, within peak = 2
    intensity_vector_filtered = np.copy(intensity_vector_refined)
    intensity_vector_filtered[:] = trabecular_attenuation        # this is set based off the upper limit intensity of trabecular bone/dense tissue
    avg_filter = np.zeros_like(intensity_vector_refined)
    std_filter = np.zeros_like(intensity_vector_refined)
    avg_filter[0 : LAG + 1] = trabecular_attenuation
    std_filter[0 : LAG + 1] = trabecular_std
    std_vis_u = avg_filter + std_filter
    std_vis_d = avg_filter - std_filter

    for p in range(LAG+1,len(intensity_vector_refined)):
        if np.abs(intensity_vector_refined[p] - avg_filter[p-1]) > (THRESHOLD * std_filter[p-1]): # this tests whether the new sample point is more than 'threshold' standard deviations away from the moving mean value
            if intensity_vector_refined[p] > avg_filter[p-1] and intensity_vector_refined[p] >= (0.6*((cortical_attenuation + trabecular_attenuation)/2)): #0.40* cortical_attenuation:     # this checks if the new sample point qualifies as a peak based off the AU intensity in the image
                intensity_BinSignal[p] = cortical_attenuation                   # this sets the binary value indicating that we are 'in' a peak zone
                intensity_vector_filtered[p] =  intensity_vector_filtered[p-1]
                if intensity_BinSignal[p-1] == trabecular_attenuation:
                    peak_count[p] = 1                                   # this will indicate the point at which a peak begins

            elif intensity_vector_refined[p] > avg_filter[p-1] and intensity_vector_refined[p] >= ((cortical_attenuation + trabecular_attenuation)/2):      #this checks if the new sample point qualifies as a peak based off the AU intensity in the image
                intensity_BinSignal[p] = cortical_attenuation                   # this sets the binary value indicating that we are 'in' a peak zone
                intensity_vector_filtered[p] =  intensity_vector_filtered[p-1]
                if intensity_BinSignal[p-1] == trabecular_attenuation:
                    peak_count[p] = 1

            else:
                intensity_BinSignal[p] = trabecular_attenuation                   # this sets the binary value indicating that we are 'out' of a peak zone
                if intensity_BinSignal[ p - 1] == cortical_attenuation:
                    peak_count[p] = 1                                   # This will indicate the point at which a peak ends
                intensity_vector_filtered[p] = (INFLUENCE * intensity_vector_refined[p]) + ((1-INFLUENCE) * intensity_vector_filtered[p-1]) # if the new point is more than threshold std deviations from the moving avg, influence determines how much it affects the filtered data
        else:
            if intensity_BinSignal[ p - 1] == cortical_attenuation:
                intensity_BinSignal[p] = trabecular_attenuation
                peak_count[p] = 1
            else:
                intensity_BinSignal[p] = trabecular_attenuation                      # if new point does not surpass threshold --> do nothing i.e., we are 'out' of a peak zone and filtered data point is equal to original data point
                intensity_vector_filtered[p] =  intensity_vector_refined[p]
            intensity_vector_filtered[p] = (INFLUENCE * intensity_vector_refined[p]) + ((1-INFLUENCE) * intensity_vector_filtered[p-1])

        avg_filter[p] = np.mean(intensity_vector_filtered[(p-LAG+1):p + 1])   # new value of filters are updated based on updated filtered data
        std_filter[p] = np.std(intensity_vector_filtered[(p-LAG+1):p + 1])
        std_vis_u[p] = avg_filter[p] + std_filter[p]
        std_vis_d[p] = avg_filter[p] - std_filter[p]

    number_of_peak_points = np.sum(peak_count)
    sample_limits = False

    try:
        sample_start_limit = peak_count.nonzero()[0].min()
        sample_end_limit = peak_count.nonzero()[0].max()
        sample_limits = True
    except:
        sample_limits = False

    if sample_limits and sample_start_limit != sample_end_limit:
        try:
            num_regions = int(number_of_peak_points)
            region_max = np.zeros(num_regions)
            peak_place = peak_count.nonzero()[0]
            for i in range(0,num_regions):
                if i < (num_regions-1):
                    region_max[i] = np.max(intensity_vector_refined[peak_place[i]:peak_place[i+1]])
                else:
                    region_max[i] = np.max(intensity_vector_refined[peak_place[i]:len(intensity_vector_refined)])
            region_sort = np.argsort(region_max)

            peak_1_interval = peak_place[region_sort[-1]:region_sort[-1]+2]
            peak_2_interval = peak_place[region_sort[-2]:region_sort[-2]+2]

            if len(peak_1_interval) == 1:
                peak_1_interval = np.concatenate((peak_1_interval, np.array([len(intensity_vector_refined) - 1])))

            if len(peak_2_interval) == 1:
                peak_2_interval = np.concatenate((peak_2_interval, np.array([len(intensity_vector_refined) - 1])))


            sort_peak_interval = np.sort(np.concatenate((peak_1_interval, peak_2_interval)))
            joint_space_interval = sort_peak_interval[1:3]
            joint_space_candidate = intensity_vector_refined[joint_space_interval[0]:(joint_space_interval[1] + 1)].argmin() + joint_space_interval[0]
        except:
            '''No joint space detected'''


    if number_of_peak_points >= 2 and np.sum(peak_count[0:joint_space_candidate]) >= 1 and np.sum(peak_count[joint_space_candidate:(len(peak_count) + 1)]) >=1:
        joint_space = joint_space_candidate
        if joint_space != 0 and joint_space != len(intensity_vector_refined):
            minimum_peak_requirement_JS_min = True                                 # if number of peak points is greater than three --> check that atleast one peak is  before and that the other is after the halfway point of the probe line

    elif number_of_peak_points >= 2:
        bin_signal_diff = np.ediff1d(intensity_BinSignal, to_begin = 0)                           # to find the actual joint space location
        joint_space = np.where(bin_signal_diff < 0)[0][-1] + 2                                                 # this is where the joint space will start
        minimum_peak_requirement_JS_diff = True

    if minimum_peak_requirement_JS_min or minimum_peak_requirement_JS_diff:

        if len(sort_peak_interval) == 4:
            peak_1_ID = intensity_vector_refined[peak_1_interval[0]: (peak_1_interval[1] + 1)].argmax() + peak_1_interval[0]
            peak_2_ID = intensity_vector_refined[peak_2_interval[0]: (peak_2_interval[1] + 1)].argmax() + peak_2_interval[0]

        else:
            peak_1_ID = -1
            peak_2_ID = -1


        if peak_1_ID > 0 and peak_2_ID > 0:
            if intensity_vector_refined[peak_1_ID] > 250 and intensity_vector_refined[peak_2_ID] > 250 and peak_1_ID != peak_2_ID:
                peak_test = True


    if peak_test:
        if minimum_peak_requirement_JS_diff:
            joint_space = intensity_vector_refined[peak_1_ID : peak_2_ID].argmin() + peak_1_ID                  # update joint space if the JS_diff method was used
        try:
            if peak_1_ID < peak_2_ID:
                tibia_difference_vector = np.abs(intensity_vector_refined[peak_1_ID : (joint_space + 1)] - ((intensity_vector_refined[peak_1_ID] + intensity_vector_refined[joint_space])/2))      # find difference between probe data entry and the half-maximum intensity value (do this for all in the defined region of the probe data)
                tibia_smallest_difference_index = tibia_difference_vector.argmin() + peak_1_ID                                                                             # get index of the point where this difference is the smallest

                femur_difference_vector = np.abs(intensity_vector_refined[joint_space : (peak_2_ID + 1)] - ((intensity_vector_refined[peak_2_ID] + intensity_vector_refined[joint_space])/2))      # repeat same as above
                femur_smallest_difference_index = femur_difference_vector.argmin() + joint_space

                tib_surface = point_number_refined[tibia_smallest_difference_index]
                fem_surface = point_number_refined[femur_smallest_difference_index]

                jsw = fem_surface - tib_surface

                peak_1_ID = []
                peak_2_ID = []

            elif peak_2_ID < peak_1_ID:
                tibia_difference_vector = np.abs(intensity_vector_refined[peak_2_ID : (joint_space + 1)] - ((intensity_vector_refined[peak_2_ID] + intensity_vector_refined[joint_space])/2))
                tibia_smallest_difference_index = tibia_difference_vector.argmin() + peak_2_ID

                femur_difference_vector = np.abs(intensity_vector_refined[joint_space : (peak_1_ID + 1)] - ((intensity_vector_refined[peak_1_ID] + intensity_vector_refined[joint_space])/2))
                femur_smallest_difference_index = femur_difference_vector.argmin() + joint_space

                tib_surface = point_number_refined[tibia_smallest_difference_index]
                fem_surface = point_number_refined[femur_smallest_difference_index]

                jsw = fem_surface - tib_surface

                peak_1_ID = []
                peak_2_ID = []

        except:
            jsw = -1

    return jsw , intensity_vector_refined, intensity_BinSignal, avg_filter, point_number_refined

#   Visualize the normals used to estimate the distribution of joint space width across the joint
def check_normals(pts, valid_Normals, surf_fem_1_cells, colours, new_surface_grid, surf1_actor, surf2_actor, surf3_actor, check = 0):

    '''check what the normals look like'''
    valid_normals_poly = vtk.vtkPolyData()
    valid_normals_poly.SetPoints(pts)
    valid_normals_poly.SetLines(valid_Normals)

    validNormalMap = vtk.vtkPolyDataMapper()
    validNormalMap.SetInputData(valid_normals_poly)

    validNormalActor = vtk.vtkActor()
    validNormalActor.SetMapper(validNormalMap)
    validNormalActor.GetProperty().SetOpacity(0.1)

    arrow = vtk.vtkArrowSource()
    arrow.SetShaftRadius(0.01)
    arrow.SetTipRadius(0.02)

    normal_glyph = vtk.vtkGlyph3D()
    normal_glyph.SetInputConnection(surf_fem_1_cells.GetOutputPort())
    normal_glyph.SetSourceConnection(arrow.GetOutputPort())
    normal_glyph.SetVectorModeToUseNormal()        #SetVectorModeToUseVector() can be used but the normals are what we want anyway
    normal_glyph.SetScaleFactor(3)

    normal_map = vtk.vtkPolyDataMapper()
    normal_map.SetInputConnection(normal_glyph.GetOutputPort())

    normal_actor = vtk.vtkActor()
    normal_actor.SetMapper(normal_map)
    normal_actor.GetProperty().SetColor(colours.GetColor3d("blue"))
    normal_actor.GetProperty().SetOpacity(0.05)
    ''' to here'''

    lut = vtk.vtkLookupTable()
    lut.SetTableRange(3, 12)
    lut.SetHueRange(0.0, 0.667)
    lut.SetValueRange(0.8, 1.0)
    lut.SetSaturationRange(0.8, 1.0)
    lut.SetRampToLinear()
    lut.SetNanColor(0.0, 0.0, 0.0, 0.9)
    lut.Build()

    point_surface_threshold = vtk.vtkThresholdPoints()
    point_surface_threshold.SetInputData(new_surface_grid)
    point_surface_threshold.ThresholdByUpper(0.0)
    point_surface_threshold.Update()

    if check:
        surface_mapper = vtk.vtkDataSetMapper()
        surface_mapper.SetInputData(point_surface_threshold.GetOutput())
        surface_mapper.ScalarVisibilityOn()
        surface_mapper.SetLookupTable(lut)
        surface_mapper.UseLookupTableScalarRangeOn()

        surface_actor = vtk.vtkActor()
        surface_actor.SetMapper(surface_mapper)
        surface_actor.GetProperty().SetPointSize(2)
        surface_actor.GetProperty().SetOpacity(1.0)

        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(lut);
        scalarBar.SetTitle("Distance");
        scalarBar.SetNumberOfLabels(7);
        scalarBar.UnconstrainedFontSizeOn();

        #Assign Actor to a renderer
        window = vtk.vtkRenderWindow()
        window.SetSize(600,550)

        renderer1 = vtk.vtkRenderer()
        renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
        renderer1.AddActor(surf1_actor)
        renderer1.AddActor(surf2_actor)
        renderer1.AddActor(surf3_actor)
        renderer1.AddActor(validNormalActor)
        window.AddRenderer(renderer1)

        renderer2 = vtk.vtkRenderer()
        renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)
        renderer2.AddActor(surface_actor)
        renderer2.AddActor(scalarBar)
        window.AddRenderer(renderer2)

        interact = vtk.vtkRenderWindowInteractor()
        interact.SetRenderWindow(window)

        interact.Initialize()
        window.Render()
        interact.Start()

    return point_surface_threshold , lut

#   Smooth the joint space width values across the surface using median filtering
def smooth_surface_scalars(surface = 0, itr = 0):

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
    smooth_scalars.SetName("distance")

    ''' create upward links from points to the cells that use each point'''
    surface.BuildLinks()

    for i in range(0, max_iter, 1):         # dictates how many times the entire surface is smoothed
        for pid in range(0, num_surface_points, 1):
            point_scalar = surface_smoothed.GetPointData().GetArray("distance").GetTuple1(pid)
            point_coords = [0] * 3
            surface.GetPoint(pid, point_coords)

            ''' Get the cells connected to the current point '''
            surface.GetPointCells(pid, point_cells)
            num_cells = point_cells.GetNumberOfIds()

            ''' scalar vector to carry all the scalar values associated with smoothing the current point '''
            scalar_vector = np.zeros(1)
            scalar_vector[0] = point_scalar     # the scalar value of the current point

            for cell in range (0, num_cells, 1):
                neighbour_cell_id = point_cells.GetId(cell)
                ''' get the points associated with the current cell '''
                surface.GetCellPoints(neighbour_cell_id, cell_points)
                num_cell_points = cell_points.GetNumberOfIds()

                ''' loop through the neighbour points (nip) associated with each neighbouring cell '''
                for nip in range (0, num_cell_points, 1):
                    np_id = cell_points.GetId(nip)
                    np_coords = [0] * 3
                    surface.GetPoint(np_id, np_coords)

                    ''' Get scalar value of the current neighbouring point '''
                    nip_scalar = surface_smoothed.GetPointData().GetScalars().GetTuple1(np_id)

                    ''' add the scalar value to the smoothing vector/equation '''
                    scalar_vector = np.append(scalar_vector, nip_scalar)
                    # end the neighbouring points loop
                # end the neighbouring cells loop

            original_point_scalar_index = np.where(scalar_vector == point_scalar)
            scalar_vector = np.delete(scalar_vector,original_point_scalar_index)
            scalar_vector = np.append(scalar_vector, point_scalar)
            scalar_vector = scalar_vector[~np.isnan(scalar_vector)]

            smoothed_scalar = np.median(scalar_vector)

            ''' add the smoothed scalar value to the smoothed scalars array '''
            smooth_scalars.SetTuple1(pid, smoothed_scalar)
        surface_smoothed.GetPointData().SetScalars(smooth_scalars)

    ''' return the surface with smoothed scalar values '''
    print(f'Scalar smoothing complete with {itr} iterations')

    return surface_smoothed
