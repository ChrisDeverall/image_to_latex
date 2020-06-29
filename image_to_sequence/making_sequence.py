import json
import pandas as pd
import sys 
from replacing_equals import *
from replacing_fraction import *
from sequence_helper import *

in_json_filename = sys.argv[1]
out_json_filename = sys.argv[2]

def sequencify(my_df):
    fraction_dict = {}

    frac_df = my_df[my_df["label"]=="frac"].sort_values(by = "w")
    non_frac_df = my_df[my_df["label"]!="frac"]
    if len(frac_df)==0:
        return left_to_right(non_frac_df, fraction_dict)
    else:

        for i in range(len(frac_df)):
            frac_sequence = []
            numerator_mask = numerator_symbols(frac_df.iloc[0], non_frac_df)
            denom_mask = denom_symbols(frac_df.iloc[0], non_frac_df)

            numerator_seq = left_to_right(non_frac_df[numerator_mask], fraction_dict)
            denom_seq = left_to_right(non_frac_df[denom_mask], fraction_dict)
            frac_sequence.extend(numerator_seq)
            
            my_dict = frac_df.iloc[0].to_dict()
            my_dict["x"]=int(my_dict["x"])
            my_dict["y"]=int(my_dict["y"])
            my_dict["w"]=int(my_dict["w"])
            my_dict["h"]=int(my_dict["h"])
            
            frac_sequence.extend([my_dict])
            frac_sequence.extend(denom_seq)
            total_mask = numerator_mask | denom_mask

            used_symbol_df = non_frac_df[total_mask]
            #find box around all used symbols
            #most left 
            x = used_symbol_df["x"].min()
            #most right
            x_max = (used_symbol_df["x"] + used_symbol_df["w"]).max()
            #most up 
            y = used_symbol_df["y"].min()
            #most down
            y_max = (used_symbol_df["y"] + used_symbol_df["h"]).max()
            #then do:
            w = x_max - x   
            h = y_max - y

            frac_label = "seq" + str(i)
            fraction_dict.update({frac_label:frac_sequence})
            non_frac_df = non_frac_df[~total_mask]
            non_frac_df = non_frac_df.append({"x":int(x), "y":int(y), "w":int(w),"h":int(h),"label":frac_label}, ignore_index = True)
            frac_df = frac_df.drop(frac_df.index[0])

        return left_to_right(non_frac_df, fraction_dict)


with open(in_json_filename) as my_file:
    data = json.load(my_file)

list_of_sequences = []
counter = 0 
for i in data:
    my_df = pd.DataFrame(i)
    my_df = replace_equals(my_df)
    my_df = replace_fraction(my_df)
    list_of_sequences.append(sequencify(my_df))
    if counter %1000 ==0:
        print("Just finished equation "+str(counter))
    counter +=1

with open(out_json_filename, 'w') as out_file:
    json.dump(list_of_sequences, out_file)