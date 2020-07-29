#
#
# Data is processed in this file
#
#
import json
import numpy as np
import matplotlib.pyplot as plt
import urllib.request as ur
from datetime import date
from matplotlib.backends.backend_pdf import PdfPages

####################################################
# Constants values 
#
####################################################

c19key='cases_time_series'
c19TotalKey='statewise'

c19RawDataUrl='https://api.covid19india.org/raw_data.json'
c19DataUrl='https://api.covid19india.org/data.json'

####################################################
#  Data Structure 
#
####################################################

class c19:
    
    __instance = None 
    
    ################################################
    # Get instance
    #
    ################################################
    def getInst():
        if c19.__instance == None:
            c19()

        return c19.__instance()

    ################################################
    # Constructor 
    #
    ################################################
    def __init__(self):
    
        if c19.__instance == None:
           
            c19.__instance = self

            #Load data from Server
            c19.UpdateDB(c19DataUrl, 'C19_DB.json')
            c19.UpdateDB(c19RawDataUrl, 'C19_RawDB.json')
        else:
           raise Exception("use c19.getInst method!")
    ################################################
    #load Data from Server
    #
    ################################################
    def UpdateDB(URL, filename):

        try:
            C19_Res = ur.urlopen(URL)
            
            # Data received Properly. Store the Data
            C19_Data = json.loads(C19_Res.read())
            
            #store data 
            with open(filename, 'w') as fp:
                json.dump(C19_Data, fp)

        except urllib.error.URLError as e:
            print('use previous data')
    


    ####################################################
    # Load Data from file  
    #
    ####################################################
    def LoadC19CasesListData(self):

        with open('C19_DB.json') as f:
              c19data = json.load(f)

        self.days=0

        #get state wise list as well  
        self.StateWiseList = c19data[c19TotalKey]
        k = self.StateWiseList[0]
        TotalCount = np.array([k['confirmed']]).astype(np.int)
        
        # [days, TotalConfirmed, TotalRecovered, TotalDeceased]
        aC19_l = np.array([0,0,0,0])
        for p in c19data[c19key]:
                self.days = self.days + 1
                temp = np.array([self.days, p['totalconfirmed'], p['totalrecovered'], p['totaldeceased']])
                aC19_l = np.vstack((aC19_l, temp))

        return aC19_l

    ####################################################
    # Calculate Slope  
    #
    ####################################################
    def CalculateSlope(self, Xi, Yi, arr, index):

        #initalize float value 
        S_Num=0.00
        S_Den=0.00
        i = 0
        weight = Yi + (Yi * 2)
        #weight = Yi
        #step1 Caculate slope 
        # Slope =  E (x-X') (y-Y')/(x-x')pow 2
        #
        for p in arr:
            x = p[0]
            y = p[index]
            
            if y > weight:
                S_Num = S_Num + (Xi-x) *(Yi-y)
                S_Den = S_Den + (Xi-x)*(Xi-x)


        slope = S_Num/S_Den
        return slope

    ####################################################
    # Linear Regression for Prediction  
    #
    ####################################################
    def PlotLinearRegression(self, C19List,pdf):

        #set the type to numeric
        arr = np.array(C19List).astype(np.int)

        #print(C19List)
        #find centroid
        self.centroid = np.mean(arr, axis=0)
        print(self.centroid) 

        #Centroid cordinates
        Xi = self.centroid[0]  ## Days
        Yi = self.centroid[1]  ## Total Cases
        Zi = self.centroid[2]  ## Total Recovered
        Wi = self.centroid[3]  ## Total Deaths

        #Calculate Slope 
        m = self.CalculateSlope(Xi, Zi, arr, 2)


        #Calculate Slope for C19 cases 
        mC19Cases = self.CalculateSlope(Xi, Yi, arr, 1)

        temp = m*Xi    
        
        #X- intercept 
        c = Zi - temp

        #X-intercept for c19 cases
        c_c19cases = Yi - (m*Xi)

        print(m)
        print(c)
        
        w_plt = arr[:,0]  ## Days
        x_plt = arr[:,1]  ## Total Cases
        y_plt = arr[:,2]  ## Total Recovered
        z_plt = arr[:,3]  ## Total Deaths

        Y_recov_plt =  m * w_plt + c

        plt.figure(figsize=(7,7))
        plt.ylabel('No of people')
        plt.xlabel('No of days')
        plt.plot(w_plt, x_plt, 'b', label='Active',linewidth=3)
        plt.plot(w_plt, y_plt, 'g', label='Recovered',linewidth=3)
        plt.plot(w_plt, z_plt, 'r', label='Dead',linewidth=3)
        plt.grid()
        plt.legend(loc='best')
        title1='c19 Cases-RawData     Date:' + date.today().strftime("%d/%m/%Y")
        plt.title(title1)
        pdf.savefig()
        plt.close()
        

        # second figure Recovery rate Prediction
        plt.figure(figsize=(7,7))
        plt.ylabel('No of people')
        plt.xlabel('No of days')
        plt.plot(w_plt, Y_recov_plt, 'b--', label='Predicted',linewidth=3)
        plt.plot(w_plt, y_plt, 'g', label='Recovered',linewidth=3)
        plt.grid()
        plt.legend(loc='best')
        title1='c19 Cases-Recovery  LinearRegression  Date:' + date.today().strftime("%d/%m/%Y")
        plt.title(title1)
        pdf.savefig()
        plt.close()



        # C19 sprad Prediction Rate. 
        y_c19Cases_plt = mC19Cases  * w_plt + c_c19cases

        plt.figure(figsize=(7,7))
        plt.ylabel('No of people')
        plt.xlabel('No of days')
        plt.plot(w_plt, y_c19Cases_plt, 'r--', label='Predicted',linewidth=3)
        plt.plot(w_plt, x_plt, 'g', label='Actual',linewidth=3)
        plt.grid()
        plt.legend(loc='best')
        title1='c19 Cases-Weighted LInear Regression Date:' + date.today().strftime("%d/%m/%Y")
        plt.title(title1)
        pdf.savefig()
        plt.close()
        
        #plt.show()

    ####################################################
    # Process Data 
    #
    ####################################################
    def ProcessData(self):

        #load data 
        self.C19List = self.LoadC19CasesListData()

        with PdfPages('c19_stats.pdf') as pdf:
    
            # PLOT LINEAR REGRESSION DATA
            self.PlotLinearRegression(self.C19List, pdf)





### EOF

