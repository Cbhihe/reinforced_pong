import requests
import json
import time

api_url = 'https://api.bitfinex.com/v1'

def symbols(s):

	res = s.get(api_url + '/symbols')
	return json.loads(res.content)

def ticket(s, sym):
	res = s.get(api_url + '/pubticker/' + sym)
	return json.loads(res.content)

def watch_tickets(s, sym):
	while True:
		data = ticket(s, sym)
		print("At {:.2f}, last price: {}".format(
			float(data['timestamp']), float(data['last_price'])))
		time.sleep(5)

s = requests.Session()
#syms = symbols(s)

sym = 'btcusd'
watch_tickets(s, sym)



#https://api.bitfinex.com/v1/pubticker/BTCUSD
