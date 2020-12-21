#!/usr/bin/python3

import sys
import re
import math
import numpy as np
import pandas as pd
import random
from datetime import datetime

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy import interpolate

from scipy import stats

def read_file(fname):
   x_label = []
   y = []
   with open(fname, 'r') as reader:
       count = 1
       for line in reader:
           x_label.append(count)
           y.append(float(line.rstrip()))
           count += 1
   return (x_label, y)

def first_order_derivative(y_arr, x_arr):
    y = np.diff(y_arr) / np.diff(x_arr)
    y = np.append(0, y)
    return (y)

def second_order_derivative(y_arr, x_arr):
    y_arr = first_order_derivative(y_arr, x_arr)
    #print("First Order: ", y_arr)

    y_arr = y_arr[1:]
    x_temp = x_arr[1:]

    y = np.diff(y_arr) / np.diff(x_temp)
    y = np.append(0, y)
    y = np.append(0, y)
 
    return (y)

def initial_differnece(line):
    differential_array=[]
    for point in range(0,len(line)-1):
        difference=line[point+1]-line[point]
        differential_array.append(abs(difference))
    return differential_array    

def local_minima_of_curve(line):
    local_minima=line[0]
    for ct in range(0,20):
        #print(ct)
        if line[ct] <= local_minima:
            local_minima=line[ct]
            #print(ct,line[ct],local_minima)
        
        if not (line[ct]<=local_minima):
            if line[ct+1]<=local_minima:
                local_minima=line[ct+1]
                #print(ct,line[ct],local_minima,"at +1")
                #ct=ct+1
            elif not (line[ct+1]<=local_minima):
                if line[ct+2]<=local_minima:
                    local_minima=line[ct+1]
                    #print(ct,line[ct],local_minima,"at +2")
                    #ct=ct+2
            else:
                exit
        else:
            exit
    return local_minima

def base_line_start_point(local_minima, line):
    start_point = 0
    for index in range(0,len(line)):
        if line[index] == local_minima:
            position = index

    for ct in range(position,0,-1):
        if (local_minima * 1.5) < line[ct]:
            if ct >= 3 and (local_minima * 1.5) < line[ct - 1]:
                if(local_minima * 1.5) > line[ct - 2]:
                    start_point = ct - 2
                else:
                    start_point = ct - 1   

        if ct >= 3 and (local_minima * 1.5) > line[ct - 1]:
            if(local_minima * 1.5) > line[ct - 2]:
                start_point = ct - 2
            else:
                start_point = ct - 1

    if start_point == None or start_point == 0:
        start_point = 3
    return (start_point)
 
def baseline_slope_check(local_minima, line):
    # Index of local_minima
    position = line.index(local_minima)
    end_point = 0

    # end_point is the index of the line when y
    for ct in range(position, len(line)):
        if (local_minima * 1.5) < line[ct]:
            #print(">>",ct)
            end_point = ct
            break

    #if end_point == None or end_point == 0:
    #    end_point = 3
    return end_point

def regression_function(start_point, end_point, line):
    if start_point == None or start_point == 0:
        start_point = 3

    if end_point == None or end_point <= start_point:
        end_point = start_point + 2

    #print("End: ", end_point)
    regression_line_x = list(range(start_point, end_point + 1))
    regression_line_y = line[start_point: end_point + 1]

    x = np.array(regression_line_x)
    y = np.array(regression_line_y)

    #print("X in regress: ", x)
    #print("Y in regress: ", y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return (slope, intercept, r_value, p_value, std_err)

def base_line_y_axis(slope, intercept, line):
    # Returning interpolated line based on start and end point
    y_axis = [(slope * x) + intercept for x in range(0, len(line))]
    return y_axis

def base_line_subtraction(line, y_axis):
    for index in range (0, len(line)):
        line[index] = line[index] - y_axis[index]
    return line    


def sigmoid(x, L, x0, k, b):
    return (L/(1 + np.exp(-k * (x-x0))) + b)

def lamp_sigmoid(x, Fmax, Fb, x_half, slope):
    return (Fb + (Fmax/(1 + np.exp(-(x-x_half)/slope))))

def lamp_log(x, Fmax, Fb, r, s, q):
    y = Fb + ((Fmax + Fb) / pow((1 + (np.exp((q * np.log(x)) - np.log(r)))), s))
    return y

def cycle_threshold_value(y_arr, x_arr):
    """
    Input: Raw data from LAMP in numpy array
    Returns: Cycle of the first found maxima in second derivative or amplification point
    Standard Deviation Check
    """
    line = list(y_arr)

    first_order_differential = initial_differnece(line)
    local_minima = local_minima_of_curve(first_order_differential)
    end_point = baseline_slope_check(local_minima,first_order_differential)
    start_point = base_line_start_point(local_minima, first_order_differential)

    if start_point == end_point:
        end_point = start_point + 2
    #if start_point == None or start_point == 0:
    #    start_point = 3
    #if end_point == None or end_point == 0 or start_point == end_point:
    #    end_point = start_point + 2

    slope, intercept, r_value, p_value, std_err = regression_function(start_point, end_point, line)

    y_init = line[start_point]
    y_last = line[len(line) - 1]

    err_range = 50

    if y_last < (y_init + (3 * std_err)) or (y_last - y_init) < err_range:
        return (-1)

    y_arr = np.array(y_arr)
    y2_derivative = second_order_derivative(y_arr, x_arr)

    # Returns the index of the maxima of second derivative
    #print(np.where(y_2derivative == max(y_2derivative)) + np.array(1))

    # Calculates the indices of the sorted array and find the indices for the 3 maxima elements
    # Returns the min of the indices
    #print(min(y_arr.argsort()[-3:] + np.array(1)))
    return (min(y2_derivative.argsort()[-3:] + np.array(1)))

def calculate_std_err_regress(pcov):
    # Return sqrt of variance from pcov
    #print(np.sqrt(np.diagonal(pcov)))
    return np.sqrt(np.diagonal(pcov))

def calculate_chi_square(y_obs, y_exp):
    y_obs = np.array(y_obs)
    y_exp = np.array(y_exp)
    # Return p-value
    #print(chisquare(f_obs=y_obs, f_exp=y_exp, ddof=(len(y_obs) - 3))[1])
    return (chisquare(f_obs=y_obs, f_exp=y_exp, ddof=(len(y_obs) - 3))[1])

def calculate_r_square(y_obs, y_exp, popt):
    residuals = y_obs - y_exp
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r_square = 1 - (ss_res / ss_tot)

    return r_square

def smoothen(x_arr, y_arr):
    x_smooth = np.linspace(x_arr.min(), x_arr.max(), len(x_arr))
    a_BSpline = interpolate.make_interp_spline(x_arr, y_arr)
    y_smooth = a_BSpline(x_smooth)
    return (x_smooth, y_smooth)


def method_function_pair(y_arr, x_arr, *args, **kwargs):
    """
    Input: y_arr, x_arr
    Output: y_fit based on the best chi_square
    """
    switcher = {
            0: 'lm_sig',
            1: 'trf_sig',
            2: 'lm_log',
            3: 'trf_log',
            }

    y_arr = np.array(y_arr)
    x_arr = np.array(x_arr)

    #std_err_all = [100] * 4 
    r_square = [-100] * 4
    chi_squares = [100] * 4
    popt_all = [None] * 4
    pcov_all = [None] * 4
    y_fit_all = [None] * 4

    Fmax = max(y_arr)
    Fb = min(y_arr)
    r = int(cycle_threshold_value(y_arr, x_arr))
    s = 1
    q = 1

    if r == -1:
        # Return y_fit as all zeroes
        y_arr[:] = 0
        return y_arr

    p_sigmoid = [Fmax, Fb, np.median(x_arr), 1] # this is an mandatory initial guess
    p_lamp_log = [Fmax, Fb, r, s, q]

    # Try all combination and calculate chi_squares and append to the list
    # Try lm_sigmoid
    try:
        #print("Trying LM Sigmoid\n")
        popt_all[0], pcov_all[0] = curve_fit(lamp_sigmoid, x_arr, y_arr, p_sigmoid, method='lm', maxfev=1200)
        y_fit_all[0] = lamp_sigmoid(x_arr, *popt_all[0])
        r_square[0] = calculate_r_square(y_arr, y_fit_all[0], popt_all[0])
        chi_squares[0] = calculate_chi_square(y_arr, y_fit_all[0])
        #std_err_all[0] = calculate_std_err_regress(pcov_all[0])
    except Exception as e:
        #print(e)
        pass

    # Try trf_sigmoid
    try:
        #print("Trying TRF Sigmoid\n")
        popt_all[1], pcov_all[1] = curve_fit(lamp_sigmoid, x_arr, y_arr, p_sigmoid, method='trf', maxfev=1200)
        y_fit_all[1] = lamp_sigmoid(x_arr, *popt_all[1])
        r_square[1] = calculate_r_square(y_arr, y_fit_all[1], popt_all[1])
        chi_squares[1] = calculate_chi_square(y_arr, y_fit_all[1])
        #std_err_all[1] = calculate_std_err_regress(pcov_all[1])
    except Exception as e:
        #print(e)
        pass

    # Try lm_log
    try:
        #print("Trying LM Log\n")
        popt_all[2], pcov_all[2] = curve_fit(lamp_log, x_arr, y_arr, p_lamp_log, method='lm', maxfev=2000)
        y_fit_all[2] = lamp_log(x_arr, *popt_all[2])
        r_square[2] = calculate_r_square(y_arr, y_fit_all[2], popt_all[2])
        chi_squares[2] = calculate_chi_square(y_arr, y_fit_all[2])
        #std_err_all[2] = calculate_std_err_regress(pcov_all[2])
    except Exception as e:
        #print(e)
        pass
    
    # Try trf_log
    try:
        #print("Trying TRF Log\n")
        popt_all[3], pcov_all[3] = curve_fit(lamp_log, x_arr, y_arr, p_lamp_log, method='trf', maxfev=1200)
        chi_squares[3] = calculate_chi_square(y_arr, y_fit_all[3])
        r_square[3] = calculate_r_square(y_arr, y_fit_all[3], popt_all[3])
        y_fit_all[3] = lamp_log(x_arr, *popt_all[3])
        #std_err_all[3] = calculate_std_err_regress(pcov_all[3])
    except Exception as e:
        #print(e)
        pass

    # Find the index of the max value in chi_squares
    #if(max(chi_squares) > 0.8):
    #    method_flag = chi_squares.index(max(chi_squares))
    #else:
    #    method_flag = -1

    #print("pcov: ", pcov_all)
    #print("TEST std: ", std_err_all)
    #print("TEST r2: ", r_square)
    #print("TEST chi2: ", chi_squares)
    #print()

    method_flag = chi_squares.index(min(chi_squares))
    r_square_flag = r_square.index(max(r_square))
    #std_err_flag = std_err_all.index(max(std_err_all))

    if args:
        method_flag = args[0][0]


    #print("Method selected using chi_square: ", switcher[method_flag])
    #print("Popt for optimized: ", popt_all[method_flag])
    #print("Method selected using r_square: ", switcher[r_square_flag])
    #print("Method selected using std_err: ", switcher[std_err_flag])

    #print("Y fit for all options: ", y_fit_all)
    #print("y_fit_all: ", y_fit_all[method_flag])
    #print("y_fit_all len: ", len(y_fit_all[method_flag]))

    #print(type(list(popt_all[method_flag])))

    # return the index of the max value in chi_squares
    #try:
    #    # Check to smooth in case the optimized value is not found
    #    if type(list(popt_all[method_flag])) != type([]):
    #        # If popt == 0 and not a list of parameters, return smooth data instead of fitted one
    #        return (y_fit_all[method_flag])
    #        #return (smoothen(x_arr, y_arr)[1])
    #except:
    #    return (smoothen(x_arr, y_arr)[1])

    return (y_fit_all[method_flag])

def analyze_data():
    # Main function to call everything
    pass

def plot_lamp_curve(y_fit, y_arr, x_arr):
    plt.plot(x_arr, y_arr, 'o', label='Raw Data')
    plt.plot(x_arr, y_fit, label="Curve Fitted")

    plt.title("Real Time LAMP")
    plt.ylabel('Fluorescence')
    plt.xlabel('Cycles')

    plt.legend()
    plt.xticks(x_arr)
    plot.show()


if __name__ == "__main__":
    """
    0: lm_sig
    1: trf_sig
    2: lm_log
    3: trf_log
    """

    f1 = sys.argv[1]

    x1, y1 = read_file(f1)

    #y_fit = analyze_data(x1, y1)

    methods = [2]

    line = y1.copy()
    line_raw = line.copy()

    # Curve Fitting
    x1 = np.array(x1)
    y1 = np.array(y1)

    first_order_differential = initial_differnece(line)
    local_minima = local_minima_of_curve(first_order_differential)
    end_point = baseline_slope_check(local_minima, first_order_differential)
    start_point = base_line_start_point(local_minima, first_order_differential)

    slope, intercept, r_value, p_value, std_err = regression_function(start_point, end_point, line)

    print("local minima = ", local_minima)
    print("end point index = ", end_point)
    print("starting point = ", start_point)

    # Regression line for the curve
    y_axis = base_line_y_axis(slope, intercept, line)

    # Subtracting regression line from raw line
    y_sub_raw = base_line_subtraction(line, y_axis)
    y_sub_raw = np.array(y_sub_raw)
    y_sub_raw[:start_point] = 0

    #y_fit = method_function_pair(y_sub_raw, x1, methods)
    y_fit = method_function_pair(y_sub_raw, x1) 

    # Zeroing all negative values
    #y_fit[y_fit < 0] = 0

    #if y_fit == None:
    #    y_fit = smoothen(x1, y1)

    #print("Y fit: ", y_fit)
    #print("x: ", x1)

    # Keeping raw data for plotting with data
    line_raw = np.array(line_raw)

    ct_raw = cycle_threshold_value(y1, x1)
    ct_raw_sub = cycle_threshold_value(y_sub_raw, x1)
    ct_fitted = cycle_threshold_value(y_fit, x1)

    print("Ct from raw derivative: ", ct_raw)
    print("Ct from raw and subtracted: ", ct_raw_sub)
    print("Ct from fitted: ", ct_fitted)

    #plt.plot(x1, line_raw, 'o', label='Raw Data')
    plt.plot(x1, y_sub_raw, 'o', label='Raw Data and Subtracted')
    plt.plot(x1, y_fit, label="Curve Fitted on Baseline Subtracted")
    plt.plot(x1, second_order_derivative(y_sub_raw, x1), label="Second Order of Raw and subtracted")
    #plt.plot(x1, second_order_derivative(y_fit, x1), label="Second Order Derivative of fitted graph")

    plt.title("Real Time LAMP")
    plt.ylabel('Relative Fluorescence Unit')
    plt.xlabel('Cycles')

    plt.legend()
    plt.xticks(x1)
    plt.show()
    plt.savefig(f'{sys.argv[1]}.png')
