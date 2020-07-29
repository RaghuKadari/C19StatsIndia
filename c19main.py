
#################################################
# Licence: GPL 
# Author : Raghu Kadari
# This is Main file 
#
#
#################################################
import sys
import cv2    

#import files 
from c19_processdata import *

def main():

    # get the instance
    pc19 = c19()

    #Process data 
    pc19.ProcessData()



if __name__== "__main__":
      main()


































