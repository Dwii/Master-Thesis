# Author: Jonas Latt, jonas.latt@flowkit.com
import sys
import getopt
import json

def userparam(constants, dictionary):
    opt, arg = getopt.getopt(sys.argv[1:], "", [i + "=" for i in constants])
    if arg:
        raise getopt.GetoptError('invalid option(s) {0}'.format(str(arg)))
    for key, value in opt:
        var = key.lstrip("-")
        T = type(dictionary[var])
        if T is list:
            dictionary[var] = json.loads(value)
        else:
            dictionary[var] = T(value)


