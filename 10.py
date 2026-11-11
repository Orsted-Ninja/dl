import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#
df = yf.download('^NSEI', start='2010-01-01', end='2022-01-01', progress=False)
df = df[['Close']].dropna().reset_index()
df.columns = ['Date', 'Close']
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

train, test = train_test_split(df, test_size=0.2, shuffle=False)
X_train, y_train = train[['Days']], train['Close']
X_test, y_test = test[['Days']], test['Close']


model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")


plt.figure(figsize=(10, 5))
plt.plot(test['Date'], y_test, label='Actual')
plt.plot(test['Date'], y_pred, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('NIFTY 50 Time Series Forecasting - Linear Regression')
plt.legend()
plt.show()
