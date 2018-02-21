library(dplyr)
library(magrittr)
library(lubridate)
library(tidyr)
library(ggplot2)
library(plotly)
library(shiny)
library(DT)
library(reshape2)
library(splitstackshape)
library(stringr)
library(corrplot)
library(corrr)
library(ggthemes)
library(zoo)
options(scipen=10000)

dat <- read.csv("Shanghai license plate price.csv")

dat$Date <- as.Date(as.yearmon(dat$Date, "%Y/%m"))

datCor <- subset(dat, select = -c(Date))
datCor <- cor(datCor)
colnames(datCor) <- c("Total Number Issued","Lowest Price","Average Price","Total Applicants")
rownames(datCor) <- c("Total Number Issued","Lowest Price","Average Price","Total Applicants")
corrplot(datCor, method= "number", type="upper")


