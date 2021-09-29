install.packages("dplyr")
install.packages("broom")
install.packages("lme4")
install.packages("ltm")
install.packages("mlbench")
install.packages("glmmLasso")
install.packages("MASS")
install.packages("MuMIn")
install.packages("margins")
install.packages("remotes")
install.packages("MLmetrics")
install.packages("lmmen")
install.packages("ppcor")
install.packages("psych")
install.packages("ggplot2")
install.packages("corrplot")

library(MASS)
library(glmmLasso)
library(mlbench)
library(tidyverse)
library(broom)
library(glmnet)
library(ltm)
library(dplyr)
library(lme4)
library(car)
library(MuMIn)
library(margins)
library(remotes)
library(MLmetrics)
library(lmmen)
library(ppcor)
library(psych)
library(ggplot2)
library(corrplot)

set.seed(222)

data <- read.csv(file= choose.files(),header =TRUE )
df <- as.data.frame(data)
names(df)[1] <- "Y"

#descriptive statistic
#biserial.cor
biserial.cor(df$Tavg.Tair, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$median.Tair, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$Tmin.Tair, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$perc10.Tair, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$Tmax.Tair, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$perc90.Tair, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$MTD, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$STD, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$IQR, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$MAD, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$Cv, df$Y, use = c("all.obs", "complete.obs"), level = 2)
biserial.cor(df$CWSI2, df$Y, use = c("all.obs", "complete.obs"), level = 2)

# feature selection
Y <- as.numeric(df$Y)

# scaling
# df_scaled <- df
# df_scaled$Tavg.Tair<-scale(df$Tavg.Tair)
# df_scaled$median.Tair<-scale(df$median.Tair)
# df_scaled$Tmin.Tair<-scale(df$Tmin.Tair)
# df_scaled$perc10.Tair<-scale(df$perc10.Tair)
# df_scaled$Tmax.Tair<-scale(df$Tmax.Tair)
# df_scaled$perc90.Tair<-scale(df$perc90.Tair)
# df_scaled$MTD<-scale(df$MTD)
# df_scaled$STD<-scale(df$STD)
# df_scaled$IQR<-scale(df$IQR)
# df_scaled$MAD<-scale(df$MAD)
# df_scaled$Tavg.Tair<-scale(df$Tavg.Tair)
# df_scaled$Cv<-scale(df$Cv)
# df_scaled$CWSI2<-scale(df$CWSI2)

# glm
log.reg <- glm(Y~.,data = df,family = binomial)
summary(log.reg)

#stepwise- backward
stepwise <-  step(log.reg, direction="both") # Backwards selection is the default
summary(stepwise)
pred <- ifelse(stepwise$fitted.values < 0.5, 0, 1)
F1_Score(pred,Y,positive = "1")
Accuracy(pred,Y)

#stepwise- forward
log.reg.min <- glm(Y~1,data = df,family = binomial)
biggest <- formula(glm(Y~.,data = df,family = binomial))
forward <-  step(log.reg.min, direction="forward", scope=biggest)
summary(forward)
pred <- ifelse(forward$fitted.values < 0.5, 0, 1)
F1_Score(pred,Y,positive = "1")
Accuracy(pred,Y)

log.reg1 <- glm(Y~MTD + STD + Cv + perc90.Tair + CWSI2 ,data = df,family = binomial)
summary(log.reg1)
pred <- ifelse(log.reg1$fitted.values < 0.5, 0, 1)
F1_Score(pred,Y,positive = "1")
Accuracy(pred,Y)

log.reg1 <- glm(Y~ Cv + STD + CWSI2 + perc90.Tair + perc10.Tair ,data = df,family = binomial)
summary(log.reg1)
pred <- ifelse(log.reg1$fitted.values < 0.5, 0, 1)
F1_Score(pred,Y,positive = "1")
Accuracy(pred,Y)




#polr

df$Y <- as.factor(df$Y)
m <- polr(Y~.,data = df, Hess=TRUE)
summary(m)

#stepwise- backward
stepwise <-  step(m, direction="both") # Backwards selection is the default
summary(stepwise)


#stepwise- forward
log.reg.min <- polr(Y~1,data = df, Hess=TRUE)
biggest <- formula(polr(Y~.,data = df, Hess=TRUE))
forward <-  step(log.reg.min, direction="forward", scope=biggest)
summary(forward)
