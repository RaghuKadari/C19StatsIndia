#
#
# Data is processed in this file
#
#
import json
import numpy as np
import matplotlib.pyplot as plt
import urllib.request as ur
import tensorflow as tf
from tensorflow import keras
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
    def PlotLinearRegression(self,pdf):

        #set the type to numeric
        arr = np.array(self.C19List).astype(np.int)

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
        
        # Third figure Recovery rate Prediction
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


    ####################################################
    # Create Data windowing 
    #
    ####################################################

    def DsWindow(self,series, window_size, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        #dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset

    ####################################################
    # Simple neural network with Data scaled before
    # applying to NN
    #
    ####################################################
    def SimpleNN(self, pdf, findLR=False):
   
        # devide by 1e4 so that the Y value is not much deviation
        
        y = (self.ys)/10000
        x = self.xs
        split = 200
            
        x_train = x[:split]
        y_train = y[:split]
           
        x_valid = x[split:]   
        y_valid = y[split:]  


        window_size = 20
        batch_size = 10


        dataset = self.DsWindow(y_train, window_size, batch_size) 
        model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 
                    tf.keras.layers.Dense(10, activation="relu"), 
                        tf.keras.layers.Dense(1)
                        ])

        if (findLR == True):
            self.FindLearningRate(model, dataset, pdf)
        else:
            model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=3e-8, momentum=0.9),metrics='acc')
            history = model.fit(dataset, epochs=500)

            y_val = y_valid
            forecast = []
            i=0
            """
            for j in y_valid:
                forecast.append(model.predict(y_valid[i: i+ window_size][np.newaxis]))
                if((len(y_valid) -i) <= window_size):
                    break
                i=i+1
            """
            for j in y:
                forecast.append(model.predict(y[i: i+ window_size][np.newaxis]))
                if((len(y) -i) <= window_size):
                    break
                i=i+1

            #forecast = forecast[split-window_size:]
            results = np.array(forecast)
            y_pred = results[:,0,0]

            xlen=len(y_pred)
            
            plt.figure(figsize=(7,7))
            plt.ylabel('C19-Cases x 1e4')
            plt.xlabel('days')
            plt.plot(x[:xlen], y_pred, 'r--', label='Predicted',linewidth=3)
            plt.plot(x, y, 'g', label='Actual',linewidth=3)
            plt.grid()
            plt.legend(loc='best')
            title1='C19cases using Dense Neural network   Date: ' + date.today().strftime("%d/%m/%Y")
            plt.title(title1)
            pdf.savefig()
            plt.close()
            

    ####################################################
    # Find optimum learning rate of NN 
    #
    ####################################################
    def FindLearningRate(self,model, dataset, pdf, Loss="mse", Metrics="mse"):

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
        optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
        model.compile(loss=Loss, optimizer=optimizer, metrics=[Metrics])
        history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])


        lrs = 1e-8 * (10 ** (np.arange(100) / 20))
        plt.semilogx(lrs, history.history["loss"])
        plt.ylabel('LOSS')
        plt.xlabel('Learning Rate')
        plt.axis([1e-8, 1e-3, 0, 300])
        plt.plot(lrs, history.history["loss"] , 'b', label='Training Loss')
        plt.grid()
        plt.legend(loc='best')
        title1='C19case Neural network LR Rate  Date:' + date.today().strftime("%d/%m/%Y")
        plt.title(title1)
        pdf.savefig()
        plt.close()
    
    ####################################################
    # Recurrent neural network with Data scaled before
    # applying to RNN
    #
    ####################################################
    def RecurrentNN(self, pdf, findLR=False):
   
        # devide by 1e4 so that the Y value is not much deviation
        y = (self.ys)/10000
        #y = self.ys
        x = self.xs
        split = 200
            
        x_train = x[:split]
        y_train = y[:split]
           
        x_valid = x[split:]   
        y_valid = y[split:]  

        window_size = 20
        batch_size = 10

        dataset = self.DsWindow(y_train, window_size, batch_size) 

        model = tf.keras.models.Sequential([
              tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                        input_shape=[None]),
                tf.keras.layers.SimpleRNN(40, return_sequences=True),
                tf.keras.layers.SimpleRNN(40, return_sequences=True),
                tf.keras.layers.SimpleRNN(40, return_sequences=True),
                  tf.keras.layers.SimpleRNN(40),
                    tf.keras.layers.Dense(1)])

        if (findLR == True):
            self.FindLearningRate(model, dataset, pdf)
        else:
            model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=7e-4, momentum=0.9),metrics='mae')
            #model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(lr=7e-4, momentum=0.9),metrics='mae')
            history = model.fit(dataset, epochs=500)

            y_val = y_valid
            forecast = []
            
            for time in range(len(y) - window_size):
                  forecast.append(model.predict(y[time:time + window_size][np.newaxis]))
    

            #forecast = forecast[split-window_size:]
            results = np.array(forecast)
            y_pred = results[:,0,0]
    
            xlen=len(y_pred)
            
            plt.figure(figsize=(7,7))
            plt.ylabel('C19-Cases x 1e4')
            plt.xlabel('days')
            plt.plot(x[:xlen], y_pred, 'r--', label='Predicted',linewidth=3)
            plt.plot(x, y, 'g', label='Actual',linewidth=3)
            plt.grid()
            plt.legend(loc='best')
            title1='C19cases using Recurrent Neural network   Date:' + date.today().strftime("%d/%m/%Y")
            plt.title(title1)
            pdf.savefig()
            plt.close()
    ####################################################
    # LSTMs with Data scaled
    # 
    ####################################################
    def LstmNN(self, pdf, findLR=False):
   
        # Get Standard Deviatin and Mean
        yStd = self.ys.std()
        yMean = self.ys.mean()

        # Normalize Y
        y = (self.ys - yMean)/yStd;
            
        x = self.xs

        split = 200
            
        x_train = x[:split]
        y_train = y[:split]
           
        x_valid = x[split:]   
        y_valid = y[split:]  

        window_size = 20
        batch_size = 10

        #Clear Session
        tf.keras.backend.clear_session()

        dataset = self.DsWindow(y_train, window_size, batch_size) 

        model = tf.keras.models.Sequential([
              tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                        input_shape=[None]),
                  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
                  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                  tf.keras.layers.Dense(1)])

        if (findLR == True):
            self.FindLearningRate(model, dataset, pdf)
        else:
            model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9),metrics='mae')
            #model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(lr=7e-4, momentum=0.9),metrics='mae')
            history = model.fit(dataset, epochs=500)

            y_val = y_valid
            forecast = []
            
            for time in range(len(y) - window_size):
                  forecast.append(model.predict(y[time:time + window_size][np.newaxis]))
    

            #forecast = forecast[split-window_size:]
            results = np.array(forecast)
            y_pred = results[:,0,0]

            #plt.figure(figsize=(10, 6))
            xlen=len(y_pred)
            
            plt.figure(figsize=(7,7))
            plt.ylabel('C19-Cases - Normalized')
            plt.xlabel('days')
            plt.plot(x[:xlen], y_pred, 'r--', label='Predicted',linewidth=3)
            plt.plot(x, y, 'g', label='Actual',linewidth=3)
            plt.grid()
            plt.legend(loc='best')
            title1='C19cases using LSTMs Date:' + date.today().strftime("%d/%m/%Y")
            plt.title(title1)
            pdf.savefig()
            plt.close()

    ####################################################
    # Process Data 
    #
    ####################################################
    def ProcessData(self):

        #load data 
        self.C19List = self.LoadC19CasesListData()
        
        self.xs=self.C19List[:,0];
        self.ys=self.C19List[:,1];
        self.xs=np.asfarray(self.xs,int)
        self.ys=np.asfarray(self.ys,int)

        with PdfPages('c19_stats.pdf') as pdf:
    
            # PLOT LINEAR REGRESSION DATA
            self.PlotLinearRegression(pdf)

            #Dense Neural Network first find Optimum Learning rate
            self.SimpleNN(pdf, True)
            self.SimpleNN(pdf)

            #Recurrent Neural Network
            self.RecurrentNN(pdf)
            
            #using LSTMs
            self.LstmNN(pdf)



### EOF

