#!/bin/python3

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


def baseline_slope_check(local_minima,line):
    for index in range(0,len(line)):
        if line[index]==local_minima:
            position=index
            

    base_line_end_point=0

    for ct in range(position,len(line)):
        #print(local_minima*1.5)
        if (local_minima*1.5)<line[ct]:
            #print(">>",ct)
            return ct

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
    return (start_point)
        
def base_line_y_axis(slope,intercept,line):
    y_axis=[]
    for x in range(0,len(line)):
        #y=mx+c
        y=slope*x+intercept
        y_axis.append(y)
    return y_axis

def base_line_subtraction(line,y_axis):
    for index in range (0,len(line)):
        line[index]=line[index]-y_axis[index]
    return line    


def regression_function(start_point,end_point,line):
    regression_line_x=[]
    regression_line_y=[]
    for index in range(start_point,end_point+1):
        regression_line_x.append(index)
        regression_line_y.append(line[index])
    x = np.array(regression_line_x)
    y = np.array(regression_line_y)

    slope,intercept,r_value,p_value,std_err=stats.linregress(x,y)
    return slope,intercept,r_value,p_value,std_err


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
    """
    y_arr = np.array(y_arr)

    y2_derivative = second_order_derivative(y_arr, x_arr)
    
    # Returns the index of the maxima of second derivative
    #print(np.where(y_2derivative == max(y_2derivative)) + np.array(1))

    # Calculates the indices of the sorted array and find the indices for the 3 maxima elements
    # Returns the min of the indices
    #print(min(y_arr.argsort()[-3:] + np.array(1)))
    return (min(y2_derivative.argsort()[-3:] + np.array(1)))


def calculate_chi_square(y_obs, y_exp):
    y_obs = np.array(y_obs)
    y_exp = np.array(y_exp)
    # Return p-value
    return (chisquare(f_obs=y_obs, f_exp=y_exp, ddof=(len(y_obs) - 3))[1])

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
    y_arr = np.array(y_arr)
    x_arr = np.array(x_arr)

    chi_squares = [-1] * 4
    popt_all = [0] * 4
    y_fit_all = [None] * 4

    Fmax = max(y_arr)
    Fb = min(y_arr)
    r = int(cycle_threshold_value(y_arr, x_arr))
    s = 1
    q = 1

    p_sigmoid = [Fmax, Fb, np.median(x_arr), 1] # this is an mandatory initial guess
    p_lamp_log = [Fmax, Fb, r, s, q]

    # Try all combination and calculate chi_squares and append to the list

    # Try lm_sigmoid
    try:
        popt_all[0], pcov = curve_fit(lamp_sigmoid, x_arr, y_arr, p_sigmoid, method='lm', maxfev=1200)
        y_fit_all[0] = lamp_sigmoid(x_arr, *popt_all[0])
        chi_squares[0] = calculate_chi_square(y_arr, y_fit_all[0])
    except:
        pass

    # Try trf_sigmoid
    try:
        popt_all[1], pcov = curve_fit(lamp_sigmoid, x_arr, y_arr, p_sigmoid, method='trf', maxfev=1200)
        y_fit_all[1] = lamp_sigmoid(x_arr, *popt_all[1])
        chi_squares[1] = calculate_chi_square(y_arr, y_fit_all[1])
    except:
        pass

    # Try lm_log
    try:
        popt_all[2], pcov = curve_fit(lamp_log, x_arr, y_arr, p_lamp_log, method='lm', maxfev=1200)
        y_fit_all[2] = lamp_log(x_arr, *popt_all[2])
        chi_squares[2] = calculate_chi_square(y_arr, y_fit_all[2])
    except:
        pass
    
    # Try trf_log
    try:
        popt_all[3], pcov = curve_fit(lamp_log, x_arr, y_arr, p_lamp_log, method='trf', maxfev=1200)
        chi_squares[3] = calculate_chi_square(y_arr, y_fit_all[3])
        y_fit_all[3] = lamp_log(x_arr, *popt_all[3])
    except:
        pass

    # Find the index of the max value in chi_squares
    #if(max(chi_squares) > 0.8):
    #    method_flag = chi_squares.index(max(chi_squares))
    #else:
    #    method_flag = -1

    method_flag = chi_squares.index(max(chi_squares))

    if args:
        method_flag = args[0][0]

    print("Method selected: ", method_flag)
    print("Popt for optimized: ", popt_all[method_flag])
    #print("y_fit_all: ", y_fit_all[method_flag])
    #print("y_fit_all len: ", len(y_fit_all[method_flag]))

    #print(type(list(popt_all[method_flag])))

    # return the index of the max value in chi_squares
    try:
        if type(list(popt_all[method_flag])) != type([]):
            return (y_fit_all[method_flag])
    except:
        return (smoothen(x_arr, y_arr)[1])

    return (y_fit_all[method_flag])

def analyze_data():
    # Main function to call everything
    pass

#def curve_fit_method_selector(i)
#    switcher = {
#            0: 'lm_sig',
#            1: 'trf_sig',
#            2: 'lm_log',
#            3: 'trf_log',
#            }
#    return switcher.get(i, 'Unable to find optimized pair for best fitted curve!')

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

    methods = [1]

    line = y1.copy()
    line_raw = line.copy()

    # Curve Fitting
    x1 = np.array(x1)
    y1 = np.array(y1)

    first_order_differential = initial_differnece(line)
    local_minima = local_minima_of_curve(first_order_differential)
    end_point = baseline_slope_check(local_minima,first_order_differential)

    print("local minima=", local_minima)
    print("end point index=", end_point)

    start_point = base_line_start_point(local_minima, first_order_differential)
    print("starting point=", start_point)

    slope, intercept, r_value, p_value, std_err = regression_function(start_point, end_point, line)

    # Regression line for the curve
    y_axis = base_line_y_axis(slope, intercept, line)

    # Subtracting regression line from raw line
    y_sub_raw = base_line_subtraction(line, y_axis)

    y_sub_raw = np.array(y_sub_raw)

    y_sub_raw[:start_point] = 0

    #y_fit = method_function_pair(y_sub_raw, x1, methods)
    y_fit = method_function_pair(y_sub_raw, x1) 
    # Keeping raw data for plotting with data
    line_raw = np.array(line_raw)

    print("Y fit: ", y_fit)
    print("x: ", x1)

    plt.plot(x1, y_sub_raw, 'o', label='Raw Data and Subtracted')
    plt.plot(x1, y_fit, label="Curve Fitted on Baseline Subtracted")
    plt.plot(x1, second_order_derivative(y_sub_raw, x1), label="Second Order of Raw and subtracted")

    plt.title("Real Time LAMP")
    plt.ylabel('Relative Fluorescence Unit')
    plt.xlabel('Cycles')

    plt.legend()
    plt.xticks(x1)

    print("Ct from raw derivative: ", cycle_threshold_value(y1, x1))
    print("Ct from raw and subtracted: ", cycle_threshold_value(y_sub_raw, x1))
    print("Ct from fitted: ", cycle_threshold_value(y_fit, x1))

    #plt.show()
    plt.savefig(f'{sys.argv[1]}.png')

    #plt.plot(x1, y_2derivative, label="Second Order Derivative")
    #plt.plot(x1, y_2der_fitted, label="Second Order Derivative of fitted graph")
    #plt.show()

