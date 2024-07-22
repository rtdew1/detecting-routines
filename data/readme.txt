# README: Data

To run the routine detection model, you must have train and test data saved
in this directory as train.csv and test.csv respectively.

Due to non-disclosure agreements, we do NOT provide the original data used in 
the paper. However, we provide a script that generates simulated data, that can 
be used to run the code. 

To generate simulated data, simply run:
python generate_sim_data.py --n {num_custs} --w {num_weeks} --t {num_test_weeks}

To modify the settings of the simulation, see: sim_config.py

To use the code with your own data, train.csv and test.csv must have the 
following columns:
 - id = a person id, starting at 1, and continuously indexed such that 
        max(id) = the number of customers
 - week = a week index, for the data as a whole (how many weeks have elapsed
          since the start of the data)
 - iweek = individual's week in the data, i.e., how many weeks since the id
           entered the data (NOT USED IN OUR MODEL EXCEPT FOR FILTERING)
 - dayhour = the combination of day of the week and hour of the day for the 
             observation (168 possible options; Sunday 12am = 0)
 - y = number of requests at that week + dayhour （always 1 or greater）