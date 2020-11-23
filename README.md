# C19StatsIndia
 Brief: Statistical analysis on Covid 19 Data sets - India.

# 1. Simple Linear Regression: 
  - Computing mean and slope of the data sets and plotting 
  - Below is the Graph, where we can see that the End prediction is closer to the actual rate.
  - similarly same plot for recovery cases for C19 cases. 
  ![image](https://user-images.githubusercontent.com/68960324/99973747-18eea680-2dc6-11eb-947b-5c916be3f22f.png)
  ![image](https://user-images.githubusercontent.com/68960324/99974025-69fe9a80-2dc6-11eb-8e3a-dac379c8b0a8.png)
  
  
# 2 Dense Neural Networks - TensorFlow:
  - Before feeding it to a Neural netweek we need to Normalize data since there is a lot of deviation end to end. 
  - To get best results it is best to find the optimum learning rate and use that learning rate and run the model. 
  - Below is a 3 layer dense neural network. The output from the graph we can see that its not bad but definatley 
    can be improved further.
    ## Learning Rate
    ![image](https://user-images.githubusercontent.com/68960324/99974806-72a3a080-2dc7-11eb-9577-c22a87165cde.png)
    ## C19 cases using Dense Neural Network - TensorFlow. 
    ![image](https://user-images.githubusercontent.com/68960324/99974915-95ce5000-2dc7-11eb-8412-2e991ddd288d.png)
    
 # 3 Using Recurrent Neural Networks - TensorFlow:
   - If you look at the graph of C19 cases, there is no seasonality or trend and for such scenarios applying previous
     interconnection models may not be fruitful. 
     ## C19 cases using Recurrent Neural Networks - TensorFlow. 
     ![image](https://user-images.githubusercontent.com/68960324/99975205-f52c6000-2dc7-11eb-8388-cb1c0129d112.png)
  # 4 Using LSTM's.
  - Simlar to RNN's, LSTM's also resulted in high loss. 
    ## C19 cases using LSTM's - TensorFlow: 
    ![image](https://user-images.githubusercontent.com/68960324/99975876-c367c900-2dc8-11eb-97cb-603718f9da1d.png)

  
   
