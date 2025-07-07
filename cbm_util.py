# -----------------------------------------------------
# cbm_util.py
#
# Created by:   Tadiwa Waungana
# Created on:   4 October, 2023
#
# Description: Supporting/helper functions for cortical bone mapping.
# -----------------------------------------------------
# Usage: Helper functions are called directly from cbm.py
#
#
# -----------------------------------------------------
''' Supporting functions for cortical bone mapping'''
import os
import vtk
import sys
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf
from scipy.optimize import least_squares
from vtk.util.misc import vtkGetTempDir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Median image filter - salt/pepper noise 
def median_fn(image):
    #   ----------------------  #
    #  image is greyscale image
    #   ----------------------  #
    median = vtk.vtkImageMedian3D()
    median.SetInputConnection(image.GetOutputPort())
    median.SetKernelSize(5,5,5)
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

#   Estimate the cortical thickness using variation/constrast in image greyscale values across cortical bone
def find_thickness(probe_line, probe_spacing, trabecular_std = 0, LAG = 0, THRESHOLD = 0, INFLUENCE = 0, density_bool = 0, apparent_density = 0, idx = 0, p1 = 0, p2 = 0, pSurf = 0):
    #   ----------------------  #
    #  Uses joint space mapping methodology to center/focus the grayscale
    #  sample data on the cortical peak closest to the bone surface
    #  model.
    #   ----------------------  #
    NumberOfSamples = probe_line.GetOutput().GetPointData().GetScalars().GetNumberOfTuples()
    point_number = np.zeros(NumberOfSamples)
    intensity_vector = np.zeros_like(point_number)

    for tuple_id in range( 0 , NumberOfSamples):
        intensity_vector[tuple_id] = np.array(list(probe_line.GetOutput().GetPointData().GetScalars().GetTuple(tuple_id)))
        point_number[tuple_id] = tuple_id * probe_spacing

    ''' if the image data is particularly noisy, you may try a gaussian filter '''
    # intensity_vector_blurred = gaussian_filter1d(intensity_vector, 5)

    point_number_refined = np.linspace(point_number[0], point_number.max(), 100)
    ''' interpolation to enable sub-resolution measurements '''
    cs = interp1d(point_number, intensity_vector)
    ''' cubic spline interpolation to enable sub-resolution measurements can be used instead of interp1d '''
    # cs = CubicSpline(point_number, intensity_vector)
    
    #   Resampled grayscale image data
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

        ''' Uncomment below to visualize an example profile across the cortical bone '''
        #     print(region_max)
        #     print(region_sort)
        #     print(peak_1_interval)
        #     print(peak_2_interval)
        #     plt.plot(point_number_refined, intensity_vector_refined, 'b-', point_number_refined, 1000*peak_count, "r.-")
        #     plt.show()

        intensity_vector_refined = intensity_vector_refined[0:joint_space+5]
        point_number_refined = point_number_refined[0:joint_space+5]

    center = point_number_refined[intensity_vector_refined.argmax()]

    if intensity_vector.max() > 0:
        plt.clf()
        try:
            x0 = center - 2.0
            x1 = center + 2.0
            y0 = 0
            if density_bool:
                # Uses predefined Ct. density for initial estimates
                y1 = apparent_density
            else:
                # Uses max CT number in sample profile as Ct. density for initial estimates
                y1 = intensity_vector_refined.max()
            y2 = 0
            sig = 1.0
            # initialize model parameter estimates to be solved
            params2 = [y0, x0, sig, y2, x1]

            def model2(params2, y1, point_number_refined):
                # Fits sample profile data to equation described in Treece et al, 2012.
                ''' Model equation '''
                ''' y0 + ((y1 - y0)/2)*(1 + math.erf((x - x0)/(sig * math.sqrt(2)))) + ((y2 - y1)/2)*(1 + math.erf((x - x1)/(sig * math.sqrt(2)))) '''
                return params2[0] + ((y1 - params2[0])/2)*(1 + erf((point_number_refined - params2[1])/(params2[2] * np.sqrt(2)))) +\
                    ((params2[3] - y1)/2)*(1 + erf((point_number_refined - params2[4])/(params2[2] * np.sqrt(2))))

            def minimize2(params2):
                ''' Function that minimizes the residuals'''
                return model2(params2, y1, point_number_refined) - intensity_vector_refined

            # Minimize initial parameter estimates using Levenberg-Marquardt method
            res2 = least_squares(minimize2, params2, method='lm')

            params2 = res2.x
            # Uncomment below to visualize sample profile data together with model fit
            # plt.plot(point_number_refined, intensity_vector_refined, 'b-', point_number_refined, model2(params2,y1,point_number_refined),'r--')
            # plt.show()

            # Extract covariance matrix to obtain uncertainty errors associated with each model parameter
            covariance_matrix = np.linalg.inv(res2.jac.T @ res2.jac)
            # print(f'Covariance Matrix: size ({covariance_matrix.shape}) \n {covariance_matrix}')
            precision_y0 = 1/covariance_matrix[0,0]
            precision_x0 = 1/covariance_matrix[1,1]
            precision_sig = 1/covariance_matrix[2,2]
            precision_y2 = 1/covariance_matrix[3,3]
            precision_x1 = 1/covariance_matrix[4,4]

            apparent_density = y1
            apparent_thick = np.abs(params2[4] - params2[1])

            ''' check thickness'''
            background = ((params2[0] + params2[3]) / 2)

            p_i = round((apparent_thick * (apparent_density - background )))

            # Create dataframe to store model parameters and parameter errors from all sample locations
            data = pd.DataFrame({'apparent_peak_density':[apparent_density],
                            'precision_y0':[precision_y0],
                            'precision_x0':[precision_x0],
                            'precision_sig':[precision_sig],
                            'precision_y2':[precision_y2],
                            'precision_x1':[precision_x1],
                            'apparent_thickness':[apparent_thick],
                            'background_density':[background],
                            'p(i)':[p_i],
                            'y_times_precision':[apparent_density*precision_sig]})

            thickness = apparent_thick

            if apparent_density < background or background < 0:
                data['apparent_peak_density'] = [0]
                data['precision_y0'] = [0]
                data['precision_x0'] = [0]
                data['precision_sig'] = [0]
                data['precision_y2'] = [0]
                data['precision_x1'] = [0]
                data['apparent_thickness'] = [0]
                data['background_density'] = [0]
                data['p(i)'] = [0]
                data['y_times_precision'] = 0

                thickness = -1

        except:
            thickness = -1
            data = pd.DataFrame({'apparent_peak_density':[0],
                            'precision_y0':[precision_y0],
                            'precision_x0':[precision_x0],
                            'precision_sig':[precision_sig],
                            'precision_y2':[precision_y2],
                            'precision_x1':[precision_x1],
                            'apparent_thickness':[0],
                            'background_density':[0],
                            'p(i)':[0],
                            'y_times_precision':[0]})

    return thickness, background, apparent_density, data, point_number_refined, intensity_vector_refined, model2(params2, y1,point_number_refined), p_i, res2

#  Estimate new Ct. density that accounts for depressed CT numbers
# (partial volume effects) at thinner bone regions - Treece et al, 2012
def mass_conservation(thick_data, args):
    #   ----------------------  #
    #  Estimate new Ct. density that accounts for depressed CT numbers
    # (partial volume effects) at thinner bone regions - Treece et al, 2012
    #   ----------------------  #
    bin_length = 150 # +/- cortical mass of 75 --> t(y1-yb)
    num_bins = int((np.asarray(thick_data['p(i)']).max() - np.asarray(thick_data['p(i)']).min())/bin_length)
    thick_data['p(i)_bin'] = pd.cut(thick_data['p(i)'],bins = num_bins, duplicates='drop')

    '''
    Weight flag - this accounts for knees which ACLR or other situations where very dense structures such as metal
        may adversely affect the estimation of the global cortical density estimation. If the maximum apparent density exceeds cutoff
        the apparent densities are weighted based on the sigma precision in each bin. If the maximum apparent density does not exceed the
        cutoff then the un-weighted densities are used for estimation

        -- Default cut-off: 4000

    '''
    weight_flag = False
    if len(np.where(thick_data['apparent_peak_density'] > 4000)[0]) > 10:
          weight_flag = True

    average_peak = np.zeros(num_bins)
    average_peak_observed = np.zeros(num_bins)
    bin_precision = np.zeros(num_bins)
    bin_index = np.zeros(num_bins)
    t_i = np.zeros(num_bins)

    y1_estimate = 3000      #default = 3000 AU
    yBackground_estimate = 420  # default = 420 AU
    sigma_estimate = 1.0        # default = 1.0

    num_observations = np.asarray(thick_data['p(i)_bin']).size

    for i in range(0,num_bins):
        i_bin = thick_data[thick_data['p(i)_bin'] == pd.Interval(thick_data['p(i)_bin'].cat.categories[i].left,thick_data['p(i)_bin'].cat.categories[i].right)]
        if not i_bin.empty:
            ''' New addition'''
            total_bin_precision = np.sum(i_bin['precision_sig'])
            weighted_apparent_density = np.asarray(i_bin['apparent_peak_density']) * (np.asarray(i_bin['precision_sig'])/total_bin_precision)
            if np.all(weighted_apparent_density):
                if weight_flag:
                    average_peak[i] = np.mean(weighted_apparent_density)
                else:
                    average_peak[i] = np.mean(i_bin['apparent_peak_density'])

                average_peak_observed[i] = np.mean(i_bin['apparent_peak_density'])

                ''' --------------------------------------- '''
                bin_precision[i] = np.mean(i_bin['precision_sig'])
                bin_index[i] = np.mean(i_bin['p(i)'])
                t_i[i] = bin_index[i] / (y1_estimate - yBackground_estimate)

    # Initialize new parameters y1_estimate, yBackground_estimate and sigma_estiamte to
    # to estimate an improve global Ct. density for thickness estimation
    conservation_params = np.array([y1_estimate, yBackground_estimate, sigma_estimate])

    def model_conservation(conservation_params, bin_index):
            ''' Model equation '''
            ''' yBackground_estimate + (y1_estimate - yBackground_estimate)*erf[(p(i)/((y1-yB)*2*np.sqrt(2)*sigma_estimate))] '''
            return conservation_params[1] + (conservation_params[0] - conservation_params[1])*\
                (erf(bin_index/((conservation_params[0]-conservation_params[1])*2*np.sqrt(2)*conservation_params[2])))

    def minimize_conservation(conservation_params):
            ''' Function that minimizes the residuals'''
            return bin_precision * (average_peak - model_conservation(conservation_params, bin_index))

    res = least_squares(minimize_conservation, conservation_params, method='lm') #, args = (tuple(bin_index), tuple(average_peak), tuple(bin_precision)))
    # conservation_params_out, pcov = curve_fit(model_conservation, bin_index, average_peak, p0=[y1,yb,sig], method='lm')
    conservation_params_out = res.x
    # covariance_matrix = np.linalg.inv(res.jac.T @ res.jac)
    print(f'\nconservation parameters = {conservation_params_out}')

    t_i_sort = t_i.argsort()
    sorted_ti = t_i[t_i_sort]
    sorted_bin = bin_index[t_i_sort]

    def model_conservation_fit(conservation_params_out, bin_index):
            ''' yBackground_estimate + (y1_estimate - yBackground_estimate)*erf[(t(i)/((y1-yB)*2*np.sqrt(2)*sigma_estimate))] '''
            return conservation_params_out[1] + (conservation_params_out[0] - conservation_params_out[1])*\
                (erf(bin_index/((conservation_params_out[0] - conservation_params_out[1]) * 2*np.sqrt(2)*conservation_params_out[2])))

    y1_estimate = np.absolute(conservation_params_out[0])

    font = {'family': 'sans',
        'color':  'black',
        'weight': 'normal',
        'size': 10,
        }

    # Visualize and assess if model converged appropriately to a value
    # representative of the Ct. density unaffected by image blurr (partial volume)
    plt.plot(t_i, average_peak_observed,'b.', label = "Observed distribution")
    plt.plot(sorted_ti, model_conservation_fit(conservation_params_out, sorted_bin), 'r-', label = "Best fit model")
    plt.xlabel("Thickness (mm)",fontdict=font )
    plt.ylabel("Apparent peak density (AU)",fontdict=font )
    plt.text(np.percentile(t_i, 95),y1_estimate-100,r'$\hat{%s}\ = {%s}$'%("y",str(round(y1_estimate,2))))
    plt.legend()
    # Uncomment below to save above plot for later
    # plot_name = os.path.splitext(os.path.basename(args.input_bone_model))[0] + "_Global_Density_Estimation.png"
    # plt.savefig((os.path.join( os.path.dirname(args.input_image),plot_name)))
    plt.show()

    print(f'    Global Cortical Density (AU): {y1_estimate}')

    return y1_estimate

# Print progressBar to terminal to track mapping progress -
# duration will depend on density of points on the mesh
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r      {prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()