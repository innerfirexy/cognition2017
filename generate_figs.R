# Generate the figures in Cognition 2017 paper
# Author: Yang Xu
# Time: 2/22/2017

library(data.table)
library(ggplot2)


##
# Plot Figure 1
dt.swbd = fread('data/SWBD_entropy.csv')
