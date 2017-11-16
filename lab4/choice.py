import numpy as np
import hashlib, sys

if len(sys.argv) != 2:
	print('Usage: {} <Last 5 digits of ID number>'.format(sys.argv[0]))

ID = sys.argv[1] # Last 5 digits of id number
OPTIONS = ['degree 2nd moment', 'mean length'] # The options

str_options = ';'.join(OPTIONS + [ID])
hash_object = hashlib.md5(str_options.encode('utf-8'))
i = hash_object.digest()[0] % len(OPTIONS)

print(OPTIONS[i])
