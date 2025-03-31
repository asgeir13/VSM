import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tkinter import filedialog
import pdb  #debugging package
t=filedialog.askopenfilenames(initialdir='B:/phd/IPC027-B')

tree=ET.parse(t[0])
root=tree.getroot()
c=root.find('MeasurementSettings').find('ProfileSequence').find('Nodes').findall('.//Angle') #In the header of sequence measurements are all the steps written out in ProfileSequence in Nodes, we parse all the angles that we measure with the following For loop.
#temperature=root.find('MeasurementSettings').find('ProfileSequence').find('Nodes').find('Node').findall('.//Name')
#print(temperature.find('Moment vs Temperature'))
print(root.find('ResultCharts').find('.//MeasurementChart').find('.//ChartXaxis').find('.//AxisType').text)

angle=np.arange(len(c))
for i, ele in enumerate(c): #We found all the angles with findall, but they are not extracted we only know the "tag" so we extract with a for loop
    angle[i]=ele.text

#pdb.set_trace()
d=root.find('ResultCharts').findall('.//MeasurementChart')[0].findall('.//ChartDataPoint') #after the header with all the postprocessing features and the sequence information comes all the data without detailed separation. All the data is in ResultCharts
fig, ax = plt.subplots()
folder=filedialog.askdirectory(initialdir='/home/fmrdata/Documents/VSMmaelingar')#pdb.set_trace()
if angle.size==0:
    Field=np.empty(len(root.find('ResultCharts').findall('.//MeasurementChart')[1].findall('.//ChartDataPoint')))
    moment=np.empty(len(Field))
    for i, ele in enumerate(root.find('ResultCharts').findall('.//MeasurementChart')[1].findall('.//ChartDataPoint')):
        Field[i]=float(ele.find('.//X').text)
        moment[i]=float(ele.find('.//Y').text)
    ax.clear()
    ax.plot(Field,moment)
    strng=folder+'/'+folder.split('/')[-1]+'_'+t[0].split('/')[-1].split('.')[0]
    plt.savefig(strng, dpi=400)
    strng=strng+'.txt'
    dataout=np.column_stack((Field,moment))
    unitx=root.find('ResultCharts').find('.//MeasurementChart').find('.//ChartXAxis').find('.//Title').text
    unity=root.find('ResultCharts').find('.//MeasurementChart').find('.//ChartYAxis').find('.//Title').text
    gap=root.find('FieldConfigurations').find('GapSetting').text
    SlopeCorrection=root.find('MeasurementSettings').find('ProfileSequence').find('Nodes').find('Node').find('DataAdjustmentSetupModel').find('UseSlopeCorrection').text
    intro=f'#{unitx}\t{unity}\t Gap setting: {gap}\tSlope correction: {SlopeCorrection}'
    np.savetxt(strng, dataout, delimiter='\t', header=intro, fmt='%.5e',comments='')

else:
    for i, element in enumerate(angle):
        print(i)
        try:
            Field=np.empty(len(root.find('ResultCharts').findall('.//MeasurementChart')[i*2+1].findall('.//ChartDataPoint')))
            moment=np.empty(len(Field))
        except:
            print('Batch measurement stopped before completion')
            break
        for n, ele in enumerate(root.find('ResultCharts').findall('.//MeasurementChart')[i*2+1].findall('.//ChartDataPoint')):
            Field[n]=float(ele.find('.//X').text)
            moment[n]=float(ele.find('.//Y').text)
        
        step=root.find('ResultCharts').findall('.//MeasurementChart')[2*i+1].find('.//Step').text
        ax.clear()
        ax.plot(Field,moment,label=step)
        ax.legend()
       
        strng=folder+'/'+folder.split('/')[-1]+'_'+str(element)
        plt.savefig(strng, dpi=400)
        strng=strng+'.txt'
        dataout=np.column_stack((Field,moment))
        unitx=root.find('ResultCharts').find('.//MeasurementChart').find('.//ChartXAxis').find('.//Title').text
        unity=root.find('ResultCharts').find('.//MeasurementChart').find('.//ChartYAxis').find('.//Title').text
        gap=root.find('FieldConfigurations').find('GapSetting').text
        for i, ele in enumerate(root.find('MeasurementSettings').find('ProfileSequence').find('Nodes').findall('Node')):
            try:
                SlopeCorrection=root.find('MeasurementSettings').find('ProfileSequence').find('Nodes').findall('Node')[i].find('DataAdjustmentSetupModel').find('UseSlopeCorrection').text
            except:
                None
        intro=f'#{unitx}\t{unity}\tStep {step}\tGap: {gap}\tSlope correction: {SlopeCorrection}'
        np.savetxt(strng, dataout, delimiter='\t', header=intro, fmt='%.5e',comments='')

