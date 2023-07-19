library(forecast)

emmision_data <- read.csv("CO2_data/emmision.csv")
head(emmision_data)

emmision_ts <- ts(emmision_data$average,
                  start     = c(1958, 3),
                  frequency = 12)
head(emmision_ts)

# Now the Time series data consists of co2 emmision data from march of 1958 to 2nd last month, ie October 2022
# We will take just a subset of this as it will better reflect the current trends of emmision rates which are quite different compared to the 1950s
# we will consider emmision from 1980

emmision_ts <- window(emmision_ts,
                      start = c(1970, 1))
plot(emmision_ts)

# Fitting the ARIMA (Auto regressive integrated moving average model)
model <- auto.arima(emmision_ts)

# Predicting next 5 years (60 months) emmisions

future_pred <- forecast(model, 60)

head(future_pred)

plot(future_pred, main = "Forecasted emmision rates",
     col.main = "green")

# To find prediction of specific date

preds_df <- as.data.frame(future_pred)
query_date <- 'May 2024'
preds_df[query_date, 1]
