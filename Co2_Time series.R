df<-read.csv("CO2 dataset.csv")
df<-na.omit(df)
#convert to time series
data<-ts(df[,-1],start=1800,end=2014,frequency=1)
#detailed metrics
class(data)
cat("Start year of co2 emission obsevatation \n",start(data))
cat("\nEnd year of CO2 emission observation",end(data))
plot(data)
abline(reg=lm(data~time(data)))
summary(data)

#split the data set
train_df<-ts(data[1:191],start=1800,end=1990,frequency=1)
test_df<-ts(data[192:215],start=1991,end=2014,frequency=1)

#detect trend
library(forecast)
trend<-ma(data,order=1,centre=TRUE)
plot(trend,col="blue")

#detrend of time series
plot(as.ts(data-trend),col="red")

#dickey fuller test
library(tseries)
adf.test(diff(log(data)),alternative="stationary",k=0) #d=1

#ACF-q value and PACF-p value
acf(diff(data)) #q-15,16,18
pacf(diff(data)) #p-7,15,16,24

#timeseries forecasting

#single exponential smoothing
ses.data<-ses(train_df,alpha = 0.2,h=24)
autoplot(ses.data)
ses.diff.data<-ses(diff(train_df),alpha = 0.2,h=24)
autoplot(ses.diff.data)
accuracy(ses.diff.data,diff(test_df))
#identify optimal alpha parameter
alpha<-seq(0.01,0.99,by=0.01)
RMSE<-NA
for(i in seq_along(alpha)) {
  fit<-ses(diff(train_df),alpha=alpha[i],h=24)
  RMSE[i]<-accuracy(fit,diff(test_df))[2,2]
}

alpha.fit<-data.frame(alpha,RMSE)
library(dplyr)
alpha.min<-filter(alpha.fit,RMSE == min(RMSE))
library(ggplot2)
ggplot(alpha.fit,aes(alpha,RMSE)) + geom_line() + geom_point(data=alpha.min,aes(alpha,RMSE),size=2,color="blue")

#refit the model with alpha=0.08
ses.diff.opt<-ses(diff(train_df),alpha=0.08,h=24)
accuracy(ses.diff.opt,diff(test_df))
p1<-autoplot(ses.diff.opt) + theme(legend.position = "bottom")
p2<-autoplot(diff(test_df)) + autolayer(ses.diff.opt,alpha=0.08) + ggtitle("Predicted vs Actuals for test data")

gridExtra::grid.arrange(p1,p2,nrow=1)
models<-list()
models["ses"]<-accuracy(ses.diff.opt,diff(test_df))[2,2]

#Holt's Method
holt.data<-holt(train_df,h=23)
autoplot(holt.data)
holt.data$model
accuracy(holt.data,test_df)

#identify optimal beta parameter
beta<-seq(0.0001,0.5,by=0.001)
RMSE<-NA
for(i in seq_along(beta)) {
  fit<-holt(train_df,beta=beta[i],h=24)
  RMSE[i]<-accuracy(fit,test_df)[2,2]
}

beta.fit<-data.frame(beta,RMSE)
library(dplyr)
beta.min<-filter(beta.fit,RMSE == min(RMSE))
library(ggplot2)
ggplot(beta.fit,aes(beta,RMSE)) + geom_line() + geom_point(data=beta.min,aes(beta,RMSE),size=2,color="blue")

#refit the model with beta=0.0631
holt.data.opt<-holt(train_df,beta=0.0631,h=24)
accuracy(holt.data.opt,test_df)
autoplot(holt.data.opt)

models["holt"]<-accuracy(holt.data,test_df)[2,2]

#damping methods
#holt's linear (additive) method
fit1<-ets(train_df,model="ZAN",alpha=0.8,beta=0.2)
pred1<-forecast(fit1,h=24)
fit2<-ets(train_df,model="ZAN",damped=TRUE,alpha=0.8,beta=0.2,phi=0.85)
pred2<-forecast(fit2,h=24)
#holt's exponential (mutliplicative) model
fit3<-ets(train_df,model="ZMN",alpha=0.8,beta=0.2)
pred3<-forecast(fit3,h=24)
fit4<-ets(train_df,model="ZMN",damped=TRUE,alpha=0.8,beta=0.2,phi=0.85)
pred4<-forecast(fit4,h=24)

autoplot(train_df) + autolayer(pred1$mean,color="blue") +autolayer(pred2$mean,color="blue",linetype="dashed") + autolayer(pred3$mean,color="red") + autolayer(pred4$mean,color="red",linetype="dashed") + autolayer(test_df,color="grey")

#ARIMA Model
#q-15,16,18
#p-7,15,16,24
q<-c(15,16,18)
p<-c(7,15,16,24)
fit<-arima(train_df,order=c(15,1,15),optim.control =list(maxit=1000))
pred<-predict(fit,n.ahead = 24)
autoplot(train_df) + autolayer(pred$pred,color="blue") + autolayer(test_df,color="red")

#optimal hyperparameter for arima model
p_val<-NULL
q_val<-NULL
RMSE<-NULL
k<-0
for(i in p)
{
  for(j in q)
  {
    p_val[k]<-i
    q_val[k]<-j
    fit<-arima(train_df,order=c(i,1,j),optim.control =list(maxit=1000),method="ML")
    pred<-predict(fit,n.ahead = 24)
    RMSE[k]<-accuracy(pred$pred,test_df)[2]
    k<-k+1
    
  }
}
arima.opt<-data.frame(p_val[1:6],q_val[1:6],RMSE)
arima.min<-filter(arima.opt,RMSE==min(RMSE))

#refitting the model with value q-18 and p-15
arima.opt.fit<-arima(train_df,order=c(15,1,18),optim.control =list(maxit=1000),method="ML")
pred<-predict(arima.opt.fit,n.ahead = 24)
autoplot(train_df) + autolayer(pred$pred,color="blue") + autolayer(test_df,color="red")
models["arima"]<-accuracy(pred$pred,test_df)[2]

#Auto-arima model
fit<-auto.arima(train_df)

future_pred <- forecast(fit,24)

head(future_pred)

plot(future_pred, main = "Forecasted emmision rates",
     col.main = "green")

models["auto-armia"]<-accuracy(future_pred$upper[1:24],test_df)[c(2,5)]

#since arima has less RMSE and MAPE we will save the arima model
saveRDS(arima.opt.fit,"arima_model.RDA")
