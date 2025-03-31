import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from lmfit import Minimizer, Parameters, report_fit
from tkinter import filedialog
from pathlib import Path
from lmfit.models import GaussianModel, LorentzianModel
from scipy import signal, optimize
import datetime
import matplotlib

matplotlib.use('TkAgg')
path = filedialog.askopenfilenames()
print(path)
Deg = np.zeros(len(path))
Coercivity = np.zeros(len(path))
Remanence = np.zeros(len(path))
Sigma = np.zeros(len(path))

if len(path[0].split('/')[-1].split('_')) >= 1:
    fnamebase = path[0].split('/')[-1].split('_')[0:-1]
    base = str()
    for i, item in enumerate(fnamebase):
        base = base + item + '_'
    fnamebase = base
else:
    fnamebase = path[0].split('/')[-1].split('_')[0]

#print(fnamebase)
data_folder = path[0].split(path[0].split('/')[-1])[0]
#print(data_folder)

# Read the first line of the file and extract the units of moment
with open(path[0], 'r') as file:
    first_line = file.readline()
    field_units = first_line.split('[')[1].split(']')[0]
    moment_units = first_line.rsplit('(m)')[1].split('[')[1].split(']')[0]
    print(f"Units of moment: {moment_units}")
    print(f"Units of Field: {field_units}")

# Plot the hysteresis loop before asking for the saturation field
def plot_hysteresis(file):
    data = np.loadtxt(file, delimiter='\t', skiprows=1)
    X = data[:, 0]
    Y = data[:, 1]
    plt.figure()
    plt.plot(X, Y, '.-')
    plt.xlabel(f'B [{field_units}]')
    plt.ylabel(f'Moment [{moment_units}]')
    plt.title('Hysteresis Loop')
    plt.show()
    
print('Plotting the hysteresis loop, evaluate the saturation field.')
plot_hysteresis(path[0])

default = input('Default run (y/n) ')
if default == 'y':
    thickness = '20'
    deltad = '0.1'
    deltawidth = '0.025'
    width = '4'
    plotting = 'y'
    offset = 'y'
    savedat = 'n'
    savepdf = 'n'
    filename = ''
    circ='y'
    if circ == 'n':
        square = input('Square (y/n) ')
        if square == 'y':
            area = float(input('Area in mm^2 ')) / 1000**2  # area in m^2
    else:
        width = input('Diameter in mm ')
        area = (float(width) / (2 * 1000))**2 * np.pi  # area in m^2

    if field_units == 'T':
        fieldsat = 0.006
    elif field_units == 'Oe':
        fieldsat = 10
else:
    circ = input('Circular (y/n) ')
    if circ == 'n':
        square = input('Square (y/n) ')
        if square == 'y':
            area = float(input('Area in mm^2 ')) / 1000**2  # area in m^2
    else:
        width = input('Diameter in mm ')
        area = (float(width) / (2 * 1000))**2 * np.pi  # area in m^2

    thickness = input('Thickness in nm ')
    plotting = input('Plot each loop (y/n) ')
    offset = input('Correct for moment offset (y/n) ')
    deltad = input('Uncertainty of d [nm] ')
    deltawidth = input('Uncertainty of diameter/width [mm] ')
    savedat = input('Save dat file (y/n) ')
    savepdf = input('Save data pdf (y/n) ')
    if field_units == 'T':
        fieldsat = float(input('Saturation field [T] '))
    elif field_units == 'Oe':
        fieldsat = float(input('Saturation field [Oe] '))

    if savepdf == 'y':
        filename = input('Filename: ')
    else:
        filename = ''

if moment_units == 'emu':
    volume = area * float(thickness) * 10**-9 * (100**3)
    magnetization_units = 'emu/cm^3'
    print('Magnetization units: emu/cm^3')
    print('Volume: ', volume)
elif moment_units == 'A·m²':
    volume = area * float(thickness) * 10**-9
    magnetization_units = 'A/m'
    print('Magnetization units: A/m')

def Magnetic_eval(file):
    data=np.loadtxt(file,delimiter='\t',skiprows=1)
    X = data[:,0]
    Y = data[:,1]

    stepsize=np.abs(X[np.argmin(np.abs(X))])*2

    #remanence
    indexzero = np.where((-stepsize < X) & (X < stepsize))

    while np.size(indexzero) < 4:
        stepsize = stepsize * 2
        indexzero = np.where((-stepsize < X) & (X < stepsize))

    #Coercivity
    index=np.where(X>fieldsat)
    polyInitial=np.polyfit(X[index],Y[index],1)
#----------------------------------------------------------------------------------------------------------
    Ycorr=Y-X*polyInitial[0]
    polyBsat=np.polyfit(X[np.where(-fieldsat>X)],Ycorr[np.where(-fieldsat>X)],1)
    polyAsat=np.polyfit(X[index], Ycorr[index],1)
    plt.xlabel(f'B [{field_units}]')
    plt.ylabel(f'Moment [{moment_units}]')
    if offset=='y':
        shift=(polyAsat[1]+polyBsat[1])/2
        Y=Y-shift
        string='Moment ' +str((polyAsat[1]-polyBsat[1])/2)+f' {moment_units}'
        moment=(polyAsat[1]-polyBsat[1])/2
        string1='delta_moment '+str((polyAsat[1]+polyBsat[1])/2)+f' {moment_units}'
        dmoment=(polyAsat[1]+polyBsat[1])/2
        print('Volume: ',volume)
        Ms=(polyAsat[1]-polyBsat[1])/(2*volume)
        if circ=='y':
            dMs=Ms*(float(deltad)/float(thickness)+2*float(deltawidth)/float(width))
            strengur1='delta_Ms ' +str(dMs)+f' {magnetization_units}'
            print(strengur1)


        strengur='Ms ' +str((polyAsat[1]-polyBsat[1])/(2*volume)) + f' {magnetization_units}'
        print(string)
        print(string1)
        print(strengur)
    elif offset=='n':
        string='Moment ' +str((polyAsat[1]-polyBsat[1])/2)+f' {moment_units}'
        string1='delta_moment '+str((polyAsat[1]+polyBsat[1])/2)+f' {moment_units}'
        print('Volume: ',volume)
        print()
        Ms=(polyAsat[1]-polyBsat[1])/(2*volume)
        strengur='Ms ' +str((polyAsat[1]-polyBsat[1])/(2*volume)) + f' {magnetization_units}'
        print(strengur)


    try:
        remanence=abs(Y[indexzero[0][0]])
        print(remanence)
        rangeY=np.where((Y>(-remanence*1.4)) & (Y<remanence*1.4))
        Xgradient=X[rangeY]
        gradY=np.gradient(Y[rangeY])
        minIndex=np.where(Xgradient<0)
        maxIndex=np.where(Xgradient>0)
        minIndex=np.where(gradY==min(gradY[minIndex]))
        maxIndex=np.where(gradY==max(gradY[maxIndex]))
    
        #maxIndexPeakfinder=signal.argrelextrema(gradY,np.greater,order=20)
        maxindex=int(maxIndex[0][0])
        minindex=minIndex[0][0]
        indexspanMax=np.arange(maxindex-2,maxindex+2)
        indexspanMin=np.arange(minindex-2,minindex+2)
    except:
        print('Skipped zero')
    
    model=LorentzianModel()
    try:
        paramsMax=model.guess(gradY[indexspanMax],x=Xgradient[indexspanMax])
        paramsMin=model.guess(abs(gradY[indexspanMin]),x=Xgradient[indexspanMin])
        resultMax=model.fit(gradY[indexspanMax],paramsMax,x=Xgradient[indexspanMax])
        resultMin=model.fit(abs(gradY[indexspanMin]),paramsMin,x=Xgradient[indexspanMin])
        plt.plot(Xgradient[indexspanMax],resultMax.best_fit)
        plt.plot(Xgradient[indexspanMin],resultMin.best_fit)
        plt.plot(resultMax.params['center'].value,0,'o')
        plt.plot(resultMin.params['center'].value,0,'o')
        #plt.xlim(-fieldsat,fieldsat)
        #print(resultMax.fit_report())
        Maxsigma=resultMax.params['sigma'].value
        Minsigma=resultMin.params['sigma'].value
        sigma=(Maxsigma+Minsigma)/2
        #Minsigma=resultMin.params['sigma'].value
        print(sigma)
    except:
        sigma=1
        print('failed')

    rangeY=np.where((Y>(-remanence*1)) & (Y<remanence*1))
    Yreduced=Y[rangeY]
    Xgradient=X[rangeY]
    #plt.plot(Xgradient,Yreduced,'+')
    gradsgn=np.gradient(Yreduced)
    commonLowerValues=np.where(gradsgn<0)
    polyLower=np.polyfit(Xgradient[commonLowerValues],Yreduced[commonLowerValues],2)
    plt.plot(Xgradient[commonLowerValues],Yreduced[commonLowerValues],'+')
    commonUpperValues=np.where(gradsgn>0)
    polyUpper=np.polyfit(Xgradient[commonUpperValues],Yreduced[commonUpperValues],2)
    plt.plot(Xgradient[commonUpperValues],Yreduced[commonUpperValues],'+')
    rightSwitch=np.roots(polyUpper)
    leftSwitch=np.roots(polyLower)
    #print(rightSwitch[np.where(rightSwitch>0)])
    #print(leftSwitch[np.where(
    # leftSwitch<0)])
    if field_units=='T':
        sigmaref=0.2/1000
    elif field_units=='Oe':
        sigmaref=0.2

    if sigma>sigmaref:
        if len(rightSwitch)>1:
            rightSwitch=rightSwitch[np.where(rightSwitch>0)]
            if len(rightSwitch)>1:
                rightSwitch=rightSwitch[np.where(rightSwitch<X[index[0][0]])]
        else:
            rightSwitch=rightSwitch[0]
        if len(leftSwitch)>1:
            leftSwitch=leftSwitch[np.where(leftSwitch<0)]
        else:
            leftSwitch=leftSwitch[0]
            
        coercivity=(rightSwitch[0]-leftSwitch[0])/2
        print('Linear fit')
        print('Coercivity ',coercivity,f'{field_units}')
        sigma=abs((rightSwitch[0]+leftSwitch[0])/2)
        print(sigma)
    else:
        coercivity=(resultMax.params['center'].value-resultMin.params['center'].value)/2
        bias=resultMax.params['center'].value+resultMin.params['center'].value
        print('Coercivity ',coercivity,f'{field_units} Bias ',bias,f'{field_units}')
        #print((resultMax.params['center'].value+resultMin.params['center'].value)/2)

    x=np.array([-coercivity,coercivity])
    y=np.array([0,0])
    plt.plot(x,y,'r*')

#----------------------------------------------------------------------------------------------------------
    plt.scatter(X, Y, marker = '^', linewidth = 0.5, s = 10, label = 'data')
    plt.plot(X,Y,linewidth=0.5)
    plt.plot(X[indexzero],Y[indexzero],'+r')
    plt.xlabel(f'B [{field_units}]')
    plt.ylabel(f'Moment [{moment_units}]')
    plt.legend()
    upperRemanence=(Y[indexzero[0][0]]+Y[indexzero[0][1]])/2
    lowerRemanence=(Y[indexzero[0][2]]+Y[indexzero[0][3]])/2
    return upperRemanence, lowerRemanence, coercivity, sigma, X, Y


if len(path)>1:
    for n, item in enumerate(path):
        #print(item)
        degree=int(eval(item.split('_')[-1].split('.txt')[0]))
        print(degree)
        Deg[n]=degree
        upper, lower, coercivity, sigma, X, Y= Magnetic_eval(item)
        Remanence[n]=(upper-lower)/2
        Coercivity[n] = coercivity
        if sigma<float(X[np.argmin(np.abs(X))])/2:
            sigma=float(X[np.argmin(np.abs(X))])/2
        Sigma[n]=sigma

    fig1=plt.figure()
    ax=plt.subplot(111,projection='polar')
    ax.errorbar(np.deg2rad(Deg), Coercivity,yerr=Sigma)
    ax.set_title("Coercivity")
    ax.set_xlabel("Degrees")
    plt.savefig(data_folder+'Coercivity.pdf')
    fig2=plt.figure()
    ax2=plt.subplot(111,projection='polar')
    ax2.plot(np.deg2rad(Deg),Remanence,'.--')
    ax2.set_title("Remanence")
    ax2.set_xlabel("Degrees")

    if filename!='':
        head=f'''#VSM data plotted {datetime.datetime.now()}\n#Deg [rad]\tCoercivity [Oe]\tDelta_Coercivity [Oe]\tRemanence [emu]'''
        dataout=np.column_stack((np.deg2rad(Deg),Coercivity,Sigma,Remanence))
        np.savetxt(data_folder+filename+'.txt',dataout, delimiter='\t',header=head,fmt='%.5e')
    plt.savefig(data_folder+'Remanence.pdf')
    plt.show()
else:
    try:
        degree=float(path[0].split('/')[-1].split('_')[-1].split('.txt')[0])
        #print(path[0].split(path[0].split('/')[-1])[0])
    except:
        degree=0

    upper, lower, coercivity, sigma, X, Y = Magnetic_eval(path[0])
    remanence=(upper-lower)/2
    if savepdf=='y':
        fig, ax=plt.subplots(figsize=[13,8])
        ax.plot(X,Y,'.-')
        ax.set_xlabel(f'B [{field_units}]')
        ax.set_ylabel(f'Moment [{moment_units}]')
        plt.savefig(data_folder+fnamebase+'.png')
        plt.show()
    
    if filename!='':
        head=f'''#VSM data plotted {datetime.datetime.now()}\n#Deg [rad]\tCoercivity [Oe]\tDelta_Coercivity [Oe]\tRemanence [emu]'''
        dataout=np.column_stack((np.deg2rad(degree),coercivity,sigma,remanence))
        filename=data_folder+filename+'.txt'
        np.savetxt(filename,dataout, delimiter='\t',header=head,fmt='%.5e')

    if plotting=='y':
        fig, ax=plt.subplots(figsize=[13,8])
        ax.plot(X,Y,'.-')
        ax.set_xlabel(f'B [{field_units}]')
        ax.set_ylabel(f'Moment [{moment_units}]')
        plt.pause(10)

    plt.close()