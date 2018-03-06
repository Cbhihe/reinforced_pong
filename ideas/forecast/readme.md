# Ideas for the ML project

## Cryptocurrency forecast

### Introduction

The number of crypto currencies is growing, but there are some more popular than 
others. Some analysis [1][1],[2][2] have revealed that they may be correlated. 
This correlation doesn't neccesary have to be instantaneous. It may happen that 
the effect of some high influent currencies affect others with some delay [3][3].

If those situations exists and can be identified with some confidence guarantee, 
a bid can be placed in order to exploit the delay, and to obtain a positive 
return in the long run.

### Objective

The objective is twofold, to predict the future values of a crypto-currency, and 
to determine what confident the value is. For example in the long run, a 95% 
predicted value x for t+1 will be at least x the 95% of the times.

Note that we may use different time scales, allowing t+d with d > 0.

### Data

The training data can be obtained from different sources. A test scenario can be 
simulated with a dataset of the previous market values in a fixed interval. Then 
we simulate the evolution of the time, with knowledge of the future value. With 
this scheme we can evaluate the accuracy and precision of the forecasted value.

For high frecuency datasets, we can use for example:

	% curl https://api.bitfinex.com/v1/pubticker/BTCUSD | jq .
	{
	  "mid": "11441.5",
	  "bid": "11441.0",
	  "ask": "11442.0",
	  "last_price": "11441.0",
	  "low": "11050.0",
	  "high": "11497.3552016",
	  "volume": "28616.53373061",
	  "timestamp": "1520194527.112287"
	}

Note that there is a limit of 30 req/min, so a frecuency at most 0.5 Hz.

[1]: https://www.sifrdata.com/cryptocurrency-correlation-matrix/
[2]: https://steemit.com/cryptocurrency/@karabrick/bitcoin-and-altcoin-correlations
[3]: https://www.researchgate.net/publication/224385407_A_Data_mining_algorithm_to_analyse_stock_market_data_using_lagged_correlation
