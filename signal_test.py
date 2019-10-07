import pycurl, json


last_file = "201909305409.RES"

c = pycurl.Curl()
c.setopt(pycurl.URL, '220.149.235.55:3000/api/postSignal')
c.setopt(pycurl.HTTPHEADER, ['Content-Type:application/json'])
data = json.dumps({"key": "C0D58A096417AE8A9677DEA74064C8BA",
                   "signal": last_file})
c.setopt(pycurl.POST, 1)
c.setopt(pycurl.POSTFIELDS, data)
c.setopt(pycurl.VERBOSE, 1)
c.perform()
c.close()
