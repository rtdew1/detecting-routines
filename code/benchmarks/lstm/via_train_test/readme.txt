# README

These are the train/test files corresponding to the data used in the paper. They have been temporarily removed, pending company approval. 

Columns:

 - id is the person id
 - week_index is the raw week number from the original data, where there were originally 100 weeks; we only used 48 of them
 - week is the week index we actually used
 - iweek tracks how many weeks since the person entered the data (NOT USED IN OUR MODEL EXCEPT FOR FILTERING)
 - wday tracks which day of the week it is
 - hour tracks which hour of the day it is
 - dayhour is the combination of wday and hour (168 possible options)
 - y tracks how many requests were made at that time; it is always 1 or greater