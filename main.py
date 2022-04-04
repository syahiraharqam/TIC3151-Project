import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import silhouette_visualizer
import plotly.express as px

import constraint
import math as m

from random import randint
import random

st.title("TIC3151 Project")

# Drop down list for questions
question = st.sidebar.selectbox("Select question", ("Question 1", "Question 2", "Question 3"))

# ----- QUESTION 1 -----

# ----- QUESTION 2 -----

# To find out how many days did Vaccine A been finish distributed
def Day_of_Vaccine_Distribution_A(Population,day_to_done,maximum_Capacity_for_Per_Day):
  
  # Set the value
  Total_Vaccine_Needed = day_to_done * maximum_Capacity_for_Per_Day
  Last_Day_Vaccine_Needed = Population - Total_Vaccine_Needed
  Last_Day = day_to_done
  Vaccine_Type = 0
  Previous_Vaccine_Needed=0

  # if the population divide with maximum capacity per day no remainder, will go into this condition
  if(Population%maximum_Capacity_for_Per_Day==0):
      st.write("Day 1 until Day",day_to_done,":",Total_Vaccine_Needed) # display the day to done and how many vaccine been distributed
      Same_Day = day_to_done
      same_day = 1
      return Previous_Vaccine_Needed,Same_Day,same_day   #return back the value
  
  # if the population divide with maximum capacity per day got remainder, will go into this condition
  else:
      Total_Vaccine_Needed = (day_to_done-1) * maximum_Capacity_for_Per_Day 
      st.write("Day 1 until Day",day_to_done-1,":",Total_Vaccine_Needed) 
      Last_Day_Vaccine_Needed = Population - Total_Vaccine_Needed 
      
    
      if(Last_Day_Vaccine_Needed>maximum_Capacity_for_Per_Day and Last_Day_Vaccine_Needed%maximum_Capacity_for_Per_Day!=0): #If the last day is over the maximum capacity per day, it will go into this condition
         Next_day = Last_Day_Vaccine_Needed - maximum_Capacity_for_Per_Day
         st.write("Day",Last_Day-1,":",maximum_Capacity_for_Per_Day)
         st.write("Day",Last_Day,":",Next_day)
         Previous_Vaccine_Needed = Next_day
         Same_Day = Last_Day
         same_day = 3
      else: # else this part will calculate the last day and return the value
         st.write("Day",Last_Day,":",Last_Day_Vaccine_Needed)
         Previous_Vaccine_Needed = Last_Day_Vaccine_Needed 
         Same_Day = Last_Day
         same_day = 2
         
      return Previous_Vaccine_Needed,Same_Day,same_day

  
# To find out how many days did Vaccine B been finish distributed
def Day_of_Vaccine_Distribution_B(Population,day_to_done,maximum_Capacity_for_Per_Day,Same_Day,Previous_Vaccine_Needed):
  
  # Set the value
  Total_Vaccine_Needed = day_to_done * maximum_Capacity_for_Per_Day
  Last_Day_Vaccine_Needed = Population - Total_Vaccine_Needed
  Last_Day = day_to_done + Same_Day
  Vaccine_Type = 0
  Total_Vaccine_Needed_B = 0
  Previous_Vaccine_Needed_B = 0
  Same_Day1 = 0

  # If Previous vaccine needed get from the previous function haven't reach the maximum capacity per day, it will go into this condition
  if(Previous_Vaccine_Needed>0 and Previous_Vaccine_Needed<maximum_Capacity_for_Per_Day):
       
       Same_Day_Vaccination_Needed = maximum_Capacity_for_Per_Day - Previous_Vaccine_Needed # To calculate how many vaccine distribution in the same day 

       st.write('Day',Same_Day,':',Same_Day_Vaccination_Needed)
       
       Total_Vaccine_Needed = (day_to_done-1) * maximum_Capacity_for_Per_Day
       Total_Vaccine_Needed_B = Population - Same_Day_Vaccination_Needed
       Larger = maximum_Capacity_for_Per_Day*(Same_Day+day_to_done-1-Same_Day)

   # if the population divide with maximum capacity per day got remainder, will go into this condition
       if(Larger<Total_Vaccine_Needed_B):
                 
                 Reminder = m.floor(Total_Vaccine_Needed_B%maximum_Capacity_for_Per_Day)
                 st.write("Day",Same_Day+1,"until Day",Same_Day+day_to_done-1,":",Larger) # display the day to done and how many vaccine been distributed
                 Last_Day_Vaccine_Needed_B = Population - Same_Day_Vaccination_Needed - Total_Vaccine_Needed_B + Reminder
                 
                #If the last day is over the maximum capacity per day, it will go into this condition
                 if( Last_Day_Vaccine_Needed_B>(maximum_Capacity_for_Per_Day*Same_Day+day_to_done-1-Same_Day) and Last_Day_Vaccine_Needed_B%maximum_Capacity_for_Per_Day!=0):
                     Next_day = Last_Day_Vaccine_Needed_B - maximum_Capacity_for_Per_Day
                     st.write("Day",Last_Day-1,":",maximum_Capacity_for_Per_Day)
                     st.write("Day",Last_Day,":",Next_day)
                     Previous_Vaccine_Needed_B = Next_day
                     Same_Day1 = 0
                     Same_Day = Last_Day
                     same_day = 2
                 else: # else this part will calculate the last day and return the value
                     st.write("Day",Last_Day,":",Last_Day_Vaccine_Needed_B)
                     Previous_Vaccine_Needed_B = Last_Day_Vaccine_Needed_B
                     Same_Day1 = 0
                     Same_Day = Last_Day
                     same_day = 1
                     

                 return Previous_Vaccine_Needed_B,Same_Day,same_day,Same_Day1 #return back the value
  
       else: # else if the population divide with maximum capacity per day no remainder, will go into this condition
                 Total_Vaccine_Needed_B = Population - Same_Day_Vaccination_Needed
                 st.write("Day",Same_Day+1,"until Day",Same_Day+day_to_done-1,":",Total_Vaccine_Needed_B)
                 Previous_Vaccine_Needed = Total_Vaccine_Needed_B
                 Last_Day_Vaccine_Needed_B = Population - Same_Day_Vaccination_Needed - Total_Vaccine_Needed_B 
                 Same_Day1 = Same_Day+1
                 Same_Day = Same_Day+day_to_done-1
                 same_day = 1

                 #If the last day is over the maximum capacity per day, it will go into this condition
                 if(Last_Day_Vaccine_Needed_B==0):
                    st.write('')

                 elif(Last_Day_Vaccine_Needed_B>0):
                     st.write("Day",Last_Day,":",Last_Day_Vaccine_Needed_B)
                     Previous_Vaccine_Needed_B = Last_Day_Vaccine_Needed_B
                     Same_Day = Last_Day
                     same_day = 1

                 elif(Last_Day_Vaccine_Needed>maximum_Capacity_for_Per_Day and Last_Day_Vaccine_Needed_B%maximum_Capacity_for_Per_Day!=0):
                     Next_day = Last_Day_Vaccine_Needed_B - maximum_Capacity_for_Per_Day
                     st.write("Day",Last_Day-1,":",maximum_Capacity_for_Per_Day)
                     st.write("Day",Last_Day,":",Next_day)
                     Previous_Vaccine_Needed_B = Next_day
                     Same_Day = Last_Day
                     same_day = 2
                 elif(Last_Day_Vaccine_Needed>maximum_Capacity_for_Per_Day): # else this part will calculate the last day and return the value
                     st.write("Day",Last_Day,":",Last_Day_Vaccine_Needed_B)
                     Previous_Vaccine_Needed_B = Last_Day_Vaccine_Needed_B
                     Same_Day = Last_Day
                     same_day = 1
                 else:
                     Same_Day = Same_Day+day_to_done-1
                     same_day = 1

                 
                 return Previous_Vaccine_Needed,Same_Day,same_day,Same_Day1
  else: # If Previous vaccine needed get from the previous function had reach the maximum capacity per day, it will go into this condition

       # if the population divide with maximum capacity per day no remainder, will go into this condition
       if(Population%maximum_Capacity_for_Per_Day==0):
               st.write("Day",Same_Day+1,"until Day",day_to_done+Same_Day,":",Total_Vaccine_Needed)
               Same_Day1 = 0
               Same_Day = day_to_done+Same_Day
               same_day = 1
               return Previous_Vaccine_Needed_B,Same_Day,same_day,Same_Day1
       else:
               Total_Vaccine_Needed = (day_to_done-1) * maximum_Capacity_for_Per_Day 
               st.write("Day",Same_Day+1,"until Day",day_to_done-1,":",Total_Vaccine_Needed) 
               Last_Day_Vaccine_Needed = Population - Total_Vaccine_Needed 
               Same_Day1 = 0
    
               if(Last_Day_Vaccine_Needed>maximum_Capacity_for_Per_Day and Last_Day_Vaccine_Needed%maximum_Capacity_for_Per_Day!=0): #If the last day is over the maximum capacity per day, it will go into this condition
                  Next_day = Last_Day_Vaccine_Needed - maximum_Capacity_for_Per_Day
                  st.write("Day",Last_Day-1,":",maximum_Capacity_for_Per_Day)
                  st.write("Day",Last_Day,":",Next_day)
                  Previous_Vaccine_Needed = Next_day
                  Same_Day = Last_Day
                  same_day = 3
               else: # else this part will calculate the last day and return the value
                  st.write("Day",Last_Day,":",Last_Day_Vaccine_Needed)
                  Previous_Vaccine_Needed = Last_Day_Vaccine_Needed 
                  Same_Day = Last_Day
                  same_day = 2
              
               return Previous_Vaccine_Needed,Same_Day,same_day,Same_Day1

  
# To find out how many days did Vaccine C been finish distributed
def Day_of_Vaccine_Distribution_C(Population,day_to_done,maximum_Capacity_for_Per_Day,Same_Day,Previous_Vaccine_Needed,Same_Day1):
  
  # Set the value
  Total_Vaccine_Needed = day_to_done * maximum_Capacity_for_Per_Day
  Last_Day_Vaccine_Needed = Population - Total_Vaccine_Needed
  Last_Day = day_to_done + Same_Day
  Vaccine_Type = 0
  Total_Vaccine_Needed_C = 0 
  Previous_Vaccine_Needed_C = 0
  Larger1 = maximum_Capacity_for_Per_Day*(Same_Day-(Same_Day1-1))

  # If Previous vaccine needed get from the previous function haven't reach the maximum capacity per day, it will go into this condition
  if(Previous_Vaccine_Needed>0 and Previous_Vaccine_Needed<maximum_Capacity_for_Per_Day):
       
       Same_Day_Vaccination_Needed = maximum_Capacity_for_Per_Day - Previous_Vaccine_Needed # To calculate how many vaccine distribution in the same day 

       st.write('Day',Same_Day,':',Same_Day_Vaccination_Needed)
       
       Total_Vaccine_Needed = (day_to_done-1) * maximum_Capacity_for_Per_Day
       Total_Vaccine_Needed_C = Population - Same_Day_Vaccination_Needed  
       Larger = maximum_Capacity_for_Per_Day*(Same_Day+day_to_done-1-Same_Day)
 
   # if the population divide with maximum capacity per day got remainder, will go into this condition
       if(Larger<Total_Vaccine_Needed_C):
                 
                 Reminder = m.floor(Total_Vaccine_Needed_C%maximum_Capacity_for_Per_Day)
                 st.write("Day",Same_Day+1,"until Day",Same_Day+day_to_done-1,":",Total_Vaccine_Needed_C - Reminder) # display the day to done and how many vaccine been distributed
                 Last_Day_Vaccine_Needed_C = Population - Same_Day_Vaccination_Needed - Total_Vaccine_Needed_C  + Reminder
                 
                #If the last day is over the maximum capacity per day, it will go into this condition
                 if(Last_Day_Vaccine_Needed_C>(maximum_Capacity_for_Per_Day*Same_Day+day_to_done-1-Same_Day) and Last_Day_Vaccine_Needed_C%maximum_Capacity_for_Per_Day!=0):
                     Next_day = Last_Day_Vaccine_Needed_C - maximum_Capacity_for_Per_Day
                     st.write("Day",Last_Day-1,":",maximum_Capacity_for_Per_Day)
                     st.write("Day",Last_Day,":",Next_day)
                     
                 else: # else this part will calculate the last day and return the value
                     st.write("Day",Last_Day,":",Last_Day_Vaccine_Needed_C)
                     
                
  
       else: # else if the population divide with maximum capacity per day no remainder, will go into this condition
                 
                 st.write("Day",Same_Day+1,"until Day",Same_Day+day_to_done-1,":",Total_Vaccine_Needed_C)
                 Last_Day_Vaccine_Needed_C = Population - Same_Day_Vaccination_Needed - Total_Vaccine_Needed_C
                 
                 #If the last day is over the maximum capacity per day, it will go into this condition
                 if(Last_Day_Vaccine_Needed_C==0):
                    st.write('')
                 elif(Last_Day_Vaccine_Needed_C>(maximum_Capacity_for_Per_Day*Same_Day+day_to_done-1-Same_Day) and Last_Day_Vaccine_Needed_C%maximum_Capacity_for_Per_Day!=0):
                     Next_day = Last_Day_Vaccine_Needed_C - maximum_Capacity_for_Per_Day
                     st.write("Day",Last_Day-1,":",maximum_Capacity_for_Per_Day)
                     st.write("Day",Last_Day,":",Next_day)
                     
                 else: # else this part will calculate the last day and return the value
                     st.write("Day",Last_Day,":",Last_Day_Vaccine_Needed_C)
                     
  elif(Larger1!=Last_Day_Vaccine_Needed):
               Same_Day_Vaccination_Needed = Larger1 - Previous_Vaccine_Needed # To calculate how many vaccine distribution in the same day 

               st.write('Day',Same_Day,':',Same_Day_Vaccination_Needed)
               Last_Day_Vaccine_Needed_C = Population - Same_Day_Vaccination_Needed
               Larger2 = (Same_Day+day_to_done-1 - Same_Day) * maximum_Capacity_for_Per_Day
               if(Last_Day_Vaccine_Needed_C>Larger2):
                 Last_Day_VaccineC = Last_Day_Vaccine_Needed_C - Larger2
                 st.write("Day",Same_Day+1,"until Day",Same_Day+day_to_done-1,":",Larger2)
                 st.write('Day',Same_Day+day_to_done,':',Last_Day_VaccineC)
               else:
                 st.write("Day",Same_Day+1,"until Day",Same_Day+day_to_done-1,":",Last_Day_Vaccine_Needed_C)
                    
  else: # If Previous vaccine needed get from the previous function had reach the maximum capacity per day, it will go into this condition

       # if the population divide with maximum capacity per day no remainder, will go into this condition
       if(Population%maximum_Capacity_for_Per_Day==0):
               st.write("Day",Same_Day+1,"until Day",day_to_done+Same_Day,":",Total_Vaccine_Needed)
               
       else:
               Total_Vaccine_Needed = (day_to_done-1) * maximum_Capacity_for_Per_Day 
               st.write("Day",Same_Day+1,"until Day",Same_Day+day_to_done-1,":",Total_Vaccine_Needed) 
               Last_Day_Vaccine_Needed = Population - Total_Vaccine_Needed 
      
    
               if(Last_Day_Vaccine_Needed>maximum_Capacity_for_Per_Day and Last_Day_Vaccine_Needed%maximum_Capacity_for_Per_Day!=0): #If the last day is over the maximum capacity per day, it will go into this condition
                  Next_day = Last_Day_Vaccine_Needed - maximum_Capacity_for_Per_Day
                  st.write("Day",Last_Day-1,":",maximum_Capacity_for_Per_Day)
                  st.write("Day",Last_Day,":",Next_day)
                  
               else: # else this part will calculate the last day and return the value
                  st.write("Day",Last_Day,":",Last_Day_Vaccine_Needed)


def vaccine_csp(maximum_Capacity_for_Per_Day, x, y, z, num_cr, max_cap_cr, rent):
  # How many days that each age need to take for finish Vaccination
  Needed_day_for_Age_below_30 = x/maximum_Capacity_for_Per_Day 
  Needed_day_for_Age_between_30_and_60 = y/maximum_Capacity_for_Per_Day 
  Needed_day_for_Age_above_60 = z/maximum_Capacity_for_Per_Day 

  #Total Day that the whole State 1 population finish the vaccination
  Total = int(x)+int(y)+int(z)

  # Display the value that calculated at the top
  st.write('Day that each age need to took for finish the vaccination')
  st.write('\nAge above 60          (Vac-A):',m.ceil(Needed_day_for_Age_above_60),'days')

  # Send The variable to function Day_of_Vaccine_Distribution_A to find how many days did Vaccine A been finish distributed
  Population = z
  max = maximum_Capacity_for_Per_Day
  day_to_done = m.ceil(Needed_day_for_Age_above_60)

  # Pass the Variable to the function 
  Previous_Vaccine_Needed,Same_Day,same_day= Day_of_Vaccine_Distribution_A(Population,day_to_done,max) # get return value to do calculation at function Day_of_Vaccine_Distribution_B 

  if(same_day==0):
      Add_Day = 0
  else: 
      Add_Day = same_day

  st.write('\nAge Between 30 and 60 (Vac-B):',int(Needed_day_for_Age_between_30_and_60) + Add_Day ,'days')

  # Send The variable to function Day_of_Vaccine_Distribution_B to find how many days did Vaccine B been finish distributed
  Population = y
  max = maximum_Capacity_for_Per_Day
  day_to_done = m.ceil(Needed_day_for_Age_between_30_and_60)

  # Pass the Variable to the function 
  Previous_Vaccine_Needed_B,Same_Day_B,same_day_b,Same_Day1  = Day_of_Vaccine_Distribution_B(Population,day_to_done,max,Same_Day,Previous_Vaccine_Needed) # get return value to do calculation at function Day_of_Vaccine_Distribution_C 

  if(same_day_b==0):
      Add_Day_B = 0
  else: 
      Add_Day_B = same_day_b

  st.write('\nAge lower than 30     (Vac-C):',int(Needed_day_for_Age_below_30)+Add_Day_B,'days') 
  # Send The variable to function Day_of_Vaccine_Distribution_C to find how many days did Vaccine C been finish distributed
  Population = x
  max = maximum_Capacity_for_Per_Day
  day_to_done = m.ceil(Needed_day_for_Age_below_30)

  # Pass the Variable to the function 
  Day_of_Vaccine_Distribution_C(Population,day_to_done,max,Same_Day_B,Previous_Vaccine_Needed_B,Same_Day1)

  #Calculate the Total Day that finish whole process of vaccine distribution
  Total_Day = Needed_day_for_Age_below_30 + Needed_day_for_Age_between_30_and_60 + Needed_day_for_Age_above_60
  st.write('\nTotal Day:',m.ceil(Total_Day))

  # Add the Variable to constraint
  problem = constraint.Problem()
  problem.addVariable('CR1',range(num_cr[0]))
  problem.addVariable('CR2',range(num_cr[1]))
  problem.addVariable('CR3',range(num_cr[2]))
  problem.addVariable('CR4',range(num_cr[3]))
  problem.addVariable('CR5',range(num_cr[4]))

  # Add constraint to CSP
  problem.addConstraint(constraint.ExactSumConstraint(maximum_Capacity_for_Per_Day,[max_cap_cr[0],max_cap_cr[1],max_cap_cr[2],max_cap_cr[3],max_cap_cr[4]]),['CR1','CR2','CR3','CR4','CR5'])

  # Get the soultions from CSP 
  solutions = problem.getSolutions()

  # Calculate the cheapest way for per day cost

  minimum_Cost_Per_Day = 10000
  solution_found = {}


  for s in solutions:
      Rental_Per_Day = s['CR1']*rent[0] + s['CR2']*rent[1] + s['CR3']*rent[2] + s['CR4']*rent[3] + s['CR5']*rent[4]
      if Rental_Per_Day < minimum_Cost_Per_Day:
          minimum_Cost_Per_Day = Rental_Per_Day
          solution_found = s

  st.write("""  
  Rental for Per Day Day at each Vaccine Centre
  CR 1 (RM{}  per day) {}
  CR 2 (RM{}  per day) {}
  CR 3 (RM{}  per day) {}
  CR 4 (RM{}  per day) {}
  CR 5 (RM{} per day) {}
  """.format(rent[0], solution_found['CR1'], rent[1], solution_found['CR2'], rent[2], solution_found['CR3'], rent[3], solution_found['CR4'], rent[4], solution_found['CR5']))

  # Find out the Amount of population for vaccination in last day
  LastDay = Total-(int(Total/maximum_Capacity_for_Per_Day)*maximum_Capacity_for_Per_Day)
  Lastday = (LastDay/100)
  Last_Day = m.ceil(Lastday)*100

  # Add the Variable to constraint
  problem2 = constraint.Problem()
  problem2.addVariable('CR1',range(num_cr[0]))
  problem2.addVariable('CR2',range(num_cr[1]))
  problem2.addVariable('CR3',range(num_cr[2]))
  problem2.addVariable('CR4',range(num_cr[3]))
  problem2.addVariable('CR5',range(num_cr[4]))

  # Add constraint to CSP
  problem2.addConstraint(constraint.ExactSumConstraint(Last_Day,[max_cap_cr[0],max_cap_cr[1],max_cap_cr[2],max_cap_cr[3],max_cap_cr[4]]),['CR1','CR2','CR3','CR4','CR5'])

  # Get the soultions from CSP 
  solutions2 = problem2.getSolutions() 

  # Calculate the cheapest way for last day cost

  minimum_Cost_Per_Day2 = 8000
  solution2_found = {}

  for s in solutions2:
      Rental_Per_Day = s['CR1']*rent[0] + s['CR2']*rent[1] + s['CR3']*rent[2] + s['CR4']*rent[3] + s['CR5']*rent[4]
      if Rental_Per_Day < minimum_Cost_Per_Day2:
          minimum_Cost_Per_Day2 = Rental_Per_Day
          solution2_found = s

  st.write(""" 
  Rental for Last Day at each Vaccine Centre 
  CR 1 (RM{}  per day) {}
  CR 2 (RM{}  per day) {}
  CR 3 (RM{}  per day) {}
  CR 4 (RM{}  per day) {}
  CR 5 (RM{} per day) {}
  """.format(rent[0], solution2_found['CR1'], rent[1], solution2_found['CR2'], rent[2], solution2_found['CR3'], rent[3], solution2_found['CR4'], rent[4], solution2_found['CR5']))

  #Total cost for per day(Not include last day)
  total = solution_found['CR1']*rent[0] + solution_found['CR2']*rent[1] + solution_found['CR3']*rent[2] + solution_found['CR4']*rent[3] + solution_found['CR5']*rent[4]
  st.write("Total cost for per day day(Not include last day): RM",total)

  #Total cost for last day
  Total = solution2_found['CR1']*rent[0] + solution2_found['CR2']*rent[1] + solution2_found['CR3']*rent[2] + solution2_found['CR4']*rent[3] + solution2_found['CR5']*rent[4]
  st.write("\nTotal cost for last day: RM",Total)


  #Total cost 
  TTL = ((m.ceil(Total_Day)-1)*total) + Total
  st.write("Total Cost: RM:",TTL)


# ----- QUESTION 3 -----

# Return data
def get_dataset():
    # Read data from csv file
    data = pd.read_csv('Bank_CreditScoring.csv')

    return data

# Return X and y data
def get_XY(data):
    # Get categorical columns
    cat_cols = []
    for col in data:
        if (data.dtypes[col] == 'object') & (col != 'Decision'):
            cat_cols.append(col)

    # Drop categorical columns in X
    num_df = data.drop(cat_cols, axis=1)

    # Get dummy variable from categorical columns
    dummy_df = pd.get_dummies(data[cat_cols])

    # Combine the dummy dataframe with X
    X_encoded = pd.concat([num_df, dummy_df], axis=1, join='inner')

    # Find X and y
    X = X_encoded.drop('Decision', axis=1)
    y = data['Decision']
    return X, y


if(question=="Question 1"):
    st.write()
elif(question=="Question 2"):
    state = st.sidebar.selectbox("Select state", ("State 1", "State 2", "State 3", "State 4", "State 5"))

    st.header('Cheapest Way to Distribute Vaccine at', state)

    max_cap_cr = [200,500,1000,2500,4000]
    rent = [100,250,500,800,1200]

    if(state=="State 1"):
        #Maximum capacity per day at State 1
        maximum_Capacity_for_Per_Day_1 = 5000

        # Population at State 1
        x = 115900 # Age lower than 30
        y = 434890 # Age Between 30 and 60
        z = 15000  # Age above 60

        num_cr_1 = [21,16,11,22,6]
        
        vaccine_csp(maximum_Capacity_for_Per_Day_1, x, y, z, num_cr_1, max_cap_cr, rent)

    elif(state=="State 2"):
        #Maximum capacity per day at State 2
        maximum_Capacity_for_Per_Day_2 = 10000

        # Population at State 2
        x = 100450 # Age lower than 30 
        y = 378860 # Age Between 30 and 60 37.886 
        z = 35234  # Age above 60 

        num_cr_2 = [31,17,16,11,3]

        vaccine_csp(maximum_Capacity_for_Per_Day_2, x, y, z, num_cr_2, max_cap_cr, rent)

    elif(state=="State 3"):
        #Maximum capacity per day at State 3
        maximum_Capacity_for_Per_Day_3 = 7500

        # Population at State 3
        x = 223400 # Age lower than 30
        y = 643320 # Age Between 30 and 60
        z = 22318  # Age above 60

        num_cr_3 = [23,16,12,13,4]

        vaccine_csp(maximum_Capacity_for_Per_Day_3, x, y, z, num_cr_3, max_cap_cr, rent)

    elif(state=="State 4"):
        #Maximum capacity per day at State 4
        maximum_Capacity_for_Per_Day_4 = 8500

        # Population at State 4
        x = 269300 # Age lower than 30
        y = 859900 # Age Between 30 and 60
        z = 23893  # Age above 60

        num_cr_4 = [17,17,17,16,2]

        vaccine_csp(maximum_Capacity_for_Per_Day_4, x, y, z, num_cr_4, max_cap_cr, rent)

    else:
        #Maximum capacity per day at State 5
        maximum_Capacity_for_Per_Day_5 = 9500

        # Population at State 5
        x = 221100 # Age lower than 30
        y = 450500 # Age Between 30 and 60
        z = 19284  # Age above 60

        num_cr_5 = [20,11,21,16,2]

        vaccine_csp(maximum_Capacity_for_Per_Day_5, x, y, z, num_cr_5, max_cap_cr, rent)
    
else:
    # Description of dataset

    st.header('Description of Bank Credit Scoring dataset')

    # Get data
    data_model = get_dataset()

    st.write('No. of columns:', data_model.shape[1])
    st.write('No. of rows:', data_model.shape[0])
    
    # Display descriptive statistics of loan amount column
    ds_data = pd.DataFrame({'Descriptive statistics of loan amount': data_model['Loan_Amount'].describe()})
    st.dataframe(ds_data)

    data_model.boxplot(by ='Employment_Type', column =['Score'], grid = False)
    # st.pyplot(bp_fig)

    # Data preprocessing

    # Get X and y data
    X, y = get_XY(data_model)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    # Initialize parameter dictionary
    params = dict()


    # Naive Bayes model

    st.header("Naive Bayes")
    st.sidebar.write("Naive Bayes")
    
    # NB parameters
    var_smoothing = st.sidebar.slider("var_smoothing", 1e-9, 0.1)
    params["var_smoothing"] = var_smoothing

    # Build NB model
    nb_model = GaussianNB(var_smoothing=params["var_smoothing"])

    # Train NB model
    nb_model.fit(X_train, y_train)

    # Predict using NB model
    nb_y_pred = nb_model.predict(X_test)

    # Evaluate NB model
    nb_acc = accuracy_score(y_test, nb_y_pred)*100

    st.write('Parameters:')
    st.write('var_smoothing', params['var_smoothing'])

    st.write('Accuracy:', nb_acc)


    # Decision Tree model

    st.header('Decision Tree')
    st.sidebar.write("Decision Tree")

    # DTC parameters
    criterion = st.sidebar.selectbox("Select criterion", ('entropy', 'gini'))
    splitter = st.sidebar.selectbox("Select splitter", ('best', 'random'))
    max_depth = st.sidebar.slider("max_depth", 1, 10)
    params["criterion"] = criterion
    params["splitter"] = splitter
    params["max_depth"] = max_depth

    # Build DTC model
    dtc_model = DecisionTreeClassifier(criterion=params["criterion"], splitter=params["splitter"], max_depth=params["max_depth"], random_state=0)
    
    # Train DTC model
    dtc_model.fit(X_train, y_train)

    # Predict using DTC model
    dtc_y_pred = dtc_model.predict(X_test)

    # Evaluate DTC model
    dtc_acc = accuracy_score(y_test, dtc_y_pred)*100

    st.write('Parameters:')
    st.write('criterion', params['criterion'])
    st.write('splitter', params['splitter'])
    st.write('max_depth', params['max_depth'])

    st.write('Accuracy:', dtc_acc)

    # Visualize decision tree
    fn=X.columns
    cn=['Accept', 'Reject']
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=3000)
    tree.plot_tree(dtc_model,
                feature_names=fn,
                class_names=cn,
                filled=True)

    st.pyplot(fig)


    # Model performance comparison

    st.header('Accuracy Comparison between Naive Bayes and Decision Tree')

    # Plot barcharts comparing accuracies of the models
    df_results = pd.DataFrame({
        'Model': ['Naive Bayes', 'Decision Tree'],
        'Accuracy': [nb_acc, dtc_acc]
    })
    df_bar_results = px.bar(df_results, x="Model", y="Accuracy")
    st.plotly_chart(df_bar_results)

    
    # K-Means Cluster

    st.sidebar.write("K-Means Cluster")
    
    st.header('K-Means Cluster')

    data = get_dataset()
    
    # Get categorical columns
    cl_cat_cols = []
    for col in data:
        if (data.dtypes[col] == 'object'):
            cl_cat_cols.append(col)

    # Drop categorical columns in X
    cl_num_df = data.drop(cl_cat_cols, axis=1)

    # Get dummy variable from categorical columns
    cl_dummy_df = pd.get_dummies(data[cl_cat_cols])

    # Combine the dummy dataframe with X
    X_encoded_cluster = pd.concat([cl_num_df, cl_dummy_df], axis=1, join='inner')

    # Finding the best k value using the elbow method
    st.write('Finding the best k-value using the elbow method')

    distortions = []
    for i in range(2,11):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(X_encoded_cluster)
        distortions.append(km.inertia_)

    # Plot
    plt.plot(range(2,11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot() 

    st.write('The optimal k-value is 5.')

    # KMeans cluster parameters
    n_clusters = st.sidebar.slider("n_clusters", 2, 11)
    params["n_clusters"] = n_clusters

    # Build KMeans cluster
    km_model = KMeans(n_clusters=params["n_clusters"], random_state=0)

    # Train KMeans cluster 
    km_model.fit(X_encoded_cluster.values)

    # Append clusters to dataframe
    df_clusters = data.copy()
    df_clusters['Cluster'] = km_model.labels_

    # Silhouette score
    sil_score = silhouette_score(X_encoded_cluster, km_baseline.labels_)
    st.write('Silhouette score (n =)', params["n_clusters"], ':', sil_score)

    # Plot silhouette visualizer
    sil_fig = silhouette_visualizer(km_model, X_encoded_cluster, colors='yellowbrick')
    st.pyplot(sil_fig)

    # Plot scatter
    sns.scatterplot(x='Loan_Amount', y='Total_Sum_of_Loan', hue='Cluster', data=df_clusters)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot() 

