import pandas as pd

def left_to_right(subset_df, fraction_dict):
    sequence_list = []
    for i in range(len(subset_df)):
        least_x_row = subset_df[subset_df["x"] ==subset_df["x"].min()].iloc[0]
        if least_x_row["label"][:3] == "seq":
            sequence_list.extend(fraction_dict[least_x_row["label"]])
        else:
            my_dict = least_x_row.to_dict()
            my_dict["x"]=int(my_dict["x"])
            my_dict["y"]=int(my_dict["y"])
            my_dict["w"]=int(my_dict["w"])
            my_dict["h"]=int(my_dict["h"])
            sequence_list.append(my_dict)
        subset_df = subset_df.drop([least_x_row.name])
    return sequence_list

def numerator_symbols(fraction, non_fraction_df):
    
    final_bool_mask= pd.Series(False, index = non_fraction_df.index)
    lowest_y_series = non_fraction_df["y"] + non_fraction_df["h"]
    middle_x_series = non_fraction_df["x"] + (non_fraction_df["w"]/2).astype(int)

    first_mask = fraction["x"] < middle_x_series
    second_mask = (fraction["x"]+fraction["w"]) > middle_x_series
    third_mask = (fraction["y"] - (2/3) * fraction["w"]) < lowest_y_series
    fourth_mask = fraction["y"] > non_fraction_df["y"]
    combined_mask = first_mask & second_mask & third_mask & fourth_mask

    return combined_mask

def denom_symbols(fraction, non_fraction_df):

    final_bool_mask= pd.Series(False, index = non_fraction_df.index)
    highest_y_series = non_fraction_df["y"]
    middle_x_series = non_fraction_df["x"] + (non_fraction_df["w"]/2).astype(int)

    minus_bottom = fraction["y"] + fraction["h"]
    first_mask = fraction["x"] < middle_x_series
    second_mask = (fraction["x"]+fraction["w"]) > middle_x_series
    third_mask = (minus_bottom + (2/3) * fraction["w"]) > highest_y_series
    fourth_mask = minus_bottom < non_fraction_df["y"]
    combined_mask = first_mask & second_mask & third_mask & fourth_mask

    return combined_mask