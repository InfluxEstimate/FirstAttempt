
# coding: utf-8

# In[1]:


import pandas
###Read the csv file containing the data. As of now, if there is a column without header 
###(usually the last one after memo), this will raise an exception.

def GetDataFromCSVFile(FileName, RigStateCol = 'Rig Super State', DateCol = 'YYYY/MM/DD',TimeCol = 'HH:MM:SS'):

    df = pandas.read_csv(FileName, parse_dates = [[DateCol, TimeCol]])
    df.index = df["{}_{}".format(DateCol, TimeCol)]

    ###Process the data and determine time chunks based on "Rig Super State".
    ###Three lists will be outputted: OpStartT (denoting the start time stamp) OpEndT (denoting the end time stamp)
    ###and OpCode (denoting the super stateb associed with each time interval)

    First = df[RigStateCol][0]
    OpCode = [First]
    OpStartT = [df.index[0]]
    LastIndex = df.index[0]
    OpEndT = []
    LastOpCode = First
    NASup = False
    SuperStateVals = {0, 1, 4, 5, 6, 7}
    for index, entry in df.iterrows():
        if NASup == True:
            OpCode.append(entry[RigStateCol])
        if entry[RigStateCol] not in SuperStateVals:
            OpEndT.append(index)
            OpStartT.append(index)
            NASup = True
        elif entry[RigStateCol] != LastOpCode and not NASup:
    ###     MeanIndex = pandas.Timestamp((index.value + LastIndex.value)/2.0)  ###Not using this line
            OpEndT.append(LastIndex)
            OpStartT.append(index)
            OpCode.append(entry[RigStateCol])
            NASup = False
        else:
            NASup = False

        LastIndex = index
        LastOpCode = entry[RigStateCol]
    OpEndT.append(LastIndex)

    ### Create a list of dataframes (dfOps) with elements comprised of a chunk of data with same operation code
    dfOps = []
    for i in range(len(OpCode)):
        dfOps.append(df.loc[OpStartT[i]:OpEndT[i]])
    return (OpCode , dfOps)


# In[2]:


class Well:
    def __init__(self, HDia, CasingID, CasingSet, TubularTable, Survey = [[],[]]):
        for entry in TubularTable:
            if entry[1] <= entry[2]:
                raise Exception("Tubular ID cannot be larger or equal to its OD")
        self.dia = HDia
        self.csgID = CasingID
        self.csgSet = CasingSet
        self.ttable = TubularTable
        ###TubularTable to be inputted as [[x1, y1, z1],[x2, y2, z2],...,[xn, yn, zn]], where x1 is the 
        ###length of first tubular in hole, y1 is the OD and z1 is the ID of that tubular. They must be enetered in the order they go in hole.
        ###Last element would be the drillpipe for which the length should be enteretd as None.

    def modifyTTable(self, BDepth):
        ###Modify a tubular table simply by adding the casing point as one of the intervals end/start points
        TubularTable = self.ttable
        if TubularTable[-1][0] is not None:
            raise Exception("last member of tubular table should not have length specificed")
        CurrentD = BDepth
        for i in range(len(TubularTable)):
            if TubularTable[i][0] is not None:
                CurrentD -= TubularTable[i][0]
                if CurrentD == self.csgSet:
                    break
                elif CurrentD < self.csgSet:
                    CutLength = self.csgSet - CurrentD
                    TotalLength = TubularTable[i][0]
                    TubularTable[i][0] -= CutLength
                    TubularTable.insert(i+1, [CutLength, TubularTable[i][1], TubularTable[i][2]])
                    break
            elif TubularTable[i][0] is None:
                CutLength = CurrentD - self.csgSet
                TubularTable[i][0] = CutLength
                TubularTable.insert(i+1, [None, TubularTable[i][1], TubularTable[i][2]])
                break
        return TubularTable
    
    def capacity(self, BDepth):
        ###Return list of the form: 
        ###depth start, depth end, inner capacity, steel capacity, outer capacity
        Caps = []
        TubularTable = self.modifyTTable(BDepth)  ###Using this function to create a point at casing point, if doesn't exist already
        CurrentBotD = BDepth
        CurrentTopD = BDepth - TubularTable[0][0]
        i = 0
        while TubularTable[i][0] is not None:
            inner = TubularTable[i][2]**2 / 1029.4
            steel = (TubularTable[i][1]**2 - TubularTable[i][2]**2) / 1029.4
            if (CurrentTopD - self.csgSet) > -0.1:
                outer = (self.dia**2 - TubularTable[i][1]**2) / 1029.4    
            else:
                outer = (self.csgID**2 - TubularTable[i][1]**2) / 1029.4
            Caps.append([CurrentBotD, CurrentTopD, inner, steel, outer])
            CurrentBotD = CurrentTopD
            if TubularTable[i+1][0] is not None:
                CurrentTopD -= TubularTable[i+1][0]
            else:
                CurrentTopD = 0
            i += 1
            
        inner = TubularTable[i][2]**2 / 1029.4
        steel = (TubularTable[i][1]**2 - TubularTable[i][2]**2) / 1029.4
        outer = (self.csgID**2 - TubularTable[i][1]**2) / 1029.4
        Caps.append([CurrentBotD, CurrentTopD, inner, steel, outer])
        return Caps               


# In[3]:


##This function is defined outside the class, and globally, to be able to detect the transfer intervals better
##Usually having the larger picture data set makes the algorithm work better
def actvTrnsAutoDetect(Data, Interval, Time = 'YYYY/MM/DD_HH:MM:SS', Model = "mahalanobis", Penalty = 50): 
        #Interval is in Minutes
        ###Not recommended to use. Current decetion algorithm is not very robust.
        import ruptures
        algo = ruptures.Pelt(model=Model).fit(Data)
        result = algo.predict(pen=Penalty)
        times = []
        Ttimes = []
        #plt.plot(Data)
        #plt.xticks(rotation=45)
        #plt.show()
        ruptures.display(Data, result)
        plt.show()
        for entry in result:
            time = Data.index[entry - 1]
            print("Time is: {}".format(time))
            #if entry is not 1 and entry is not len(self.data):  
            if True:
                ###Excluding results that are first and last. It's usually meaningless.
                times.append(time)
                Ttimes.append([time - pandas.Timedelta(minutes = Interval / 2) , time + pandas.Timedelta(minutes = Interval / 2)])
        return Ttimes

###Interpolate function is designed to take a dataframe and take a list of times in the "Time" input.
###The function interpolates the value of "Label" column of the data frame for the requested time and returns
###in a list format. If the time is an exact hit, it will return the exact match. If time is out of bounds of dataframe
###an exception will be raised. The Time list can only contain datetime objects.
def InterpolateDF(df, Time, Label, TimeStampLabel = "YYYY/MM/DD_HH:MM:SS"):
    Outputs = []
    for entry in Time:
        try:
            Outputs.append(df.iloc[df.index.get_loc(entry)][Label])
        except KeyError:
            try:
                TimeBefore = df.iloc[df.index.get_loc(entry, 'ffill')][TimeStampLabel]
                ValueBefore = df.iloc[df.index.get_loc(entry, 'ffill')][Label]
                TimeAfter = df.iloc[df.index.get_loc(entry, 'bfill')][TimeStampLabel]
                ValueAfter = df.iloc[df.index.get_loc(entry, 'bfill')][Label]
                Coef = (entry - TimeBefore) / (TimeAfter - TimeBefore)
                Interpolation = ValueBefore + Coef * (ValueAfter - ValueBefore)
                Outputs.append(Interpolation)
            except KeyError:
                raise Exception('Invalid interpolation for time {0} in a sub-database containing data for times                                 between {1} and {2}'.format(entry, df.iloc[0][TimeStampLabel],df.iloc[-1][TimeStampLabel]))
    return Outputs
            


# In[4]:


###Class defined to take a chunk of timeseries data plus additional optional information and estimate influx rate
import numpy
import scipy
from datetime import timedelta
import matplotlib.pyplot as plt   ###NOT NEEDED

class InfluxEst:
    def __init__(self, Data, WellData, Op = 0, DilutionRate = 0):
        self.data = Data
        self.sortedData = self.data.sort_index()
        self.op = Op
        self.well = WellData   ###Well object created for the well to which this data belongs
        
    def length(self):
        return self.sortedData.iloc[-1].loc['YYYY/MM/DD_HH:MM:SS'] - self.sortedData.iloc[0].loc['YYYY/MM/DD_HH:MM:SS']
    
    def startT(self):
        return self.sortedData.iloc[0].loc['YYYY/MM/DD_HH:MM:SS']

    def endT(self):
        return self.sortedData.iloc[-1].loc['YYYY/MM/DD_HH:MM:SS']
    
    def pipeInVol(self):
        EndBitDepth = self.sortedData.iloc[-1].loc['Bit Depth']
        StartBitDepth = self.sortedData.iloc[0].loc['Bit Depth']     
        EndCapacity = self.well.capacity(EndBitDepth)
        StartCapacity = self.well.capacity(StartBitDepth)
        EndVol = 0
        StartVol = 0
        for entry in EndCapacity:
            EndVol += (entry[0] - entry[1]) * entry[3]
        for entry in StartCapacity:
            StartVol += (entry[0] - entry[1]) * entry[3]
        return EndVol - StartVol
    
    def holeMadeVol(self):
        MaxHoleDepth = self.data['Hole Depth'].max()
        MinHoleDepth = self.data['Hole Depth'].min()
        return (MaxHoleDepth - MinHoleDepth) * self.well.dia ** 2 / 1029.4
    
    def volPumpIn(self):
        return (numpy.trapz(self.data['Flow'],self.data.index)/numpy.timedelta64(1, 'm'))/42
    
    def activeVolChange(self, TransferInt = [], Method = "Average", TransferAutoDetect = False, InterpolateTransferInt = True):
        VolData = self.sortedData.loc[:, 'Total Mud Volume'].copy()
        if TransferInt == [] and not TransferAutoDetect:
            return self.data['Total Mud Volume'][-1] - self.data['Total Mud Volume'][0]
        
        elif TransferInt != []:
            CurrentTime = self.startT()
            if TransferInt[0][0] < CurrentTime or TransferInt[-1][1] > self.endT():
                raise Exception("Transfer time interval out of bounds for the dataset")
            NoTransInt = []
            for entry in TransferInt:
                VolData[entry[0]:entry[1]] = numpy.nan
                if entry[1]  < entry[0]:
                    raise Exception("a transfer interval should be a pair of increasing order")
                if entry[0] < CurrentTime:
                    raise Exception("transfer intervals should be of increasing order")
                NoTransInt.append([CurrentTime, entry[0]])
                CurrentTime = entry[1]
            NoTransInt.append([CurrentTime, self.endT()])
        
        elif TransferAutoDetect and TransferInt == []:
            TransferInt = self.actvTrnsAutoDetect(10)
            NoTransInt = []
            CurrentTime = self.startT()
            for entry in TransferInt:
                if entry[0] > CurrentTime:
                    NoTransInt.append([CurrentTime, entry[0]])
                    CurrentTime = entry[1]
            NoTransInt.append([TransferInt[-1][1], self.endT()])
            
        if Method is "Interpolation":
            
            try:
                SmWindow = int(round(timedelta(minutes = 20)/(VolData.index[1]-VolData.index[0])))
            except:
                SmWindow = 20
            VolDiff = VolData.rolling(SmWindow).median().diff()
            IntpVolDiff = VolDiff.interpolate(method = 'linear', limit_direction = 'both')
            VolChange = scipy.trapz(IntpVolDiff)
        if Method is "Average":
            TotalNoTransTime = timedelta(minutes = 0)
            TotalNoTransVolDel = 0
            for entry in NoTransInt:
                TotalNoTransTime += entry[1] - entry[0]
                Vols = InterpolateDF(self.data, entry, "Total Mud Volume")
                TotalNoTransVolDel += Vols[1] - Vols[0]
            VolChange = TotalNoTransVolDel * (self.endT() - self.startT()) / TotalNoTransTime
            ###print("TotalNoTransTime is {} and TotalNoTransVolDel is {}".format(TotalNoTransTime,TotalNoTransVolDel))
        return VolChange
    
    def actvTrnsAutoDetect(self, Interval, Model = "mahalanobis", Penalty = 50): #Interval is in Minutes
        ###Not recommended to use. Current decetion algorithm is not very robust.
        import ruptures
        algo = ruptures.Pelt(model=Model).fit(self.data['Total Mud Volume'])
        result = algo.predict(pen=Penalty)
        times = []
        Ttimes = []
        for entry in result:
            time = self.data.iloc[entry - 1].loc['YYYY/MM/DD_HH:MM:SS']
            #if entry is not 1 and entry is not len(self.data):  
            if True:
                ###Excluding results that are first and last. It's usually meaningless.
                times.append(time)
                Ttimes.append([time - pandas.Timedelta(minutes = Interval / 2) , time + pandas.Timedelta(minutes = Interval / 2)])
                return Ttimes


# In[7]:


(OpCode, dfOps) = GetDataFromCSVFile('Data2.csv')


# In[8]:


MyWell = Well(8.75, 8.835, 2414, [[120, 6.75, 2.7],[1477, 5, 3], [None, 4, 3.344]])


# In[9]:


MyInflux = InfluxEst(dfOps[0], MyWell)


# In[10]:


MyInflux.pipeInVol()

