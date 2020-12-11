import numpy as np
from scipy import stats 
import matplotlib.pyplot as pyplot


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

def base_line_start_point(local_minima,line):
    for index in range(0,len(line)):
        if line[index]==local_minima:
            position=index

    for ct in range(position,0,-1):
        if (local_minima*1.5)<line[ct]:
            if ct>=3 and (local_minima*1.5)<line[ct-1]:
                if(local_minima*1.5)>line[ct-2]:
                    return ct-2
                else:
                    return ct-1   

        if ct>=3 and (local_minima*1.5)>line[ct-1]:
            if(local_minima*1.5)>line[ct-2]:
                return ct-2
                  

            else:
                return ct-1
        

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




    
        

if __name__=='__main__':
    line=[118.818748
 
,118.106246
 
,117.243751

,117.012496
 
,117.862503

,118.056251

,118.9375

,119.256248

,120.125

,119.418746

,117.3125

,117.59375

,118.237503

,152.193756
 
,157.100006
 
,160.131256
 
,164.306243
 
,169.03125
 
,173.256256
 
,177.431243
 
,181.425003
 
,185.162506
 
,190.762496
 
,197.731246
 
,207.831253
 
,222.212493
 
,233.143753
 
,235.018753
 
,238.956253
 
,263.118743
 
,274.287506

,278.100006
 
,280.024993

,281.818756
 
,282.543762

,282.600006
 
,283.456237
 
,283.831237
 
,283.731262
 
,284.456237
 
,285.09375
 
,285.131256

,285.787506
 
,285.606262
 
,285.162506

,284.662506
 
,285.71875

,285.331237

,285.137512

,285.693756

,285.643737
 
,285.943756
 
,286.112487
 
,286.331237
 
,286.8125
 
,286.075012
 
,286.112487
 
,285.818756

,285.668762
]
    '''line=[
    2525.84,
    2531.91,
    2538.84,
    2545.43,
    2554.30,
    2554.84,
    2558.88,
    2570.34,
    2575.52,
    2589.24,
    2612.22,
    2633.68,
    2678.99,
    2750.15,
    2866.48,
    2998.61,
    3174.53,
    3354.29,
    3555.73,
    3720.90,
    3887.88,
    3998.88,
    4120.55,
    4237.41,
    4296.00,
    4409.78,
    4468.09,
    4527.91,
    4597.67,
    4650.05,
    4688.96,
    4749.85,
    4775.06,
    4836.63,
    4862.57,
    4906.62,
    4928.09,
    4959.39,
    4991.94,
    5028.05,
    5061.02,
    5109.81,
    5114.73,
    5177.51,
    5176.47,
    5214.39,
    5246.77,
    5263.09,
    5300.24,
    5315.44,
    5340.77,
    5345.73,
    5376.28,
    5363.46,
    5430.18
        ]'''



    #print(initial_differnece(line))
    first_order_differential=initial_differnece(line)
    local_minima=local_minima_of_curve(first_order_differential)
    end_point=baseline_slope_check(local_minima,first_order_differential)
    print("local minima=",local_minima)
    print("end point index=",end_point)
    start_point=base_line_start_point(local_minima,first_order_differential)
    print("starting point=",start_point)




    slope,intercept,r_value,p_value,std_err=regression_function(start_point,end_point,line)

    y_axis=base_line_y_axis(slope,intercept,line)


    subtracted_y_line=base_line_subtraction(line,y_axis)
    line_x=[]
    for index in range(0,len(line)):
        line_x.append(index)
    

    
 
    pyplot.plot(line_x,subtracted_y_line)
    pyplot.title('subtracted baseline')
    pyplot.show() 



    



