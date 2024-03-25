# load json module
import json
import os
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    sns.set(style="white", color_codes=True)
    sns.set_context(
        rc={"font.family": 'sans', "font.size": 24, "axes.titlesize": 24,
            "axes.labelsize": 24})
    # python dictionary with key value pairs
    dict = {'Python' : 1, 'C++' : '.cpp', 'Java' : True}
    write = False
    print(os.getcwd())

    if write:
        # create json object from dictionary
        json = json.dumps(dict)

        # open file for writing, "w"
        f = open("oldies/experiments/dict.json", "w")

        # write json object to file
        f.write(json)

        # close file
        f.close()
    else:
        # Opening JSON file
        f = open('oldies/experiments/dict.json')

        # returns JSON object as
        # a dictionary
        data = json.load(f)
        print(data)
        print(type(data['Python']))
