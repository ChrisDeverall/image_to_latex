import pandas as pd 



def above_fraction(minus_df, non_minus_df):
    final_bool_mask= pd.Series(False, index = minus_df.index)
    lowest_y_series = non_minus_df["y"] + non_minus_df["h"]
    middle_x_series = non_minus_df["x"] + (non_minus_df["w"]/2).astype(int)
    for index, row, in minus_df.iterrows():

        first_mask = row["x"] < middle_x_series
        second_mask = (row["x"]+row["w"]) > middle_x_series
        third_mask = (row["y"] - (2/3) * row["w"]) < lowest_y_series
        fourth_mask = row["y"] > non_minus_df["y"]
        combined_mask = first_mask & second_mask & third_mask & fourth_mask
        if combined_mask.sum()>0:
            final_bool_mask[index] = True
    return final_bool_mask

def below_fraction(minus_df, non_minus_df):

    final_bool_mask= pd.Series(False, index = minus_df.index)
    highest_y_series = non_minus_df["y"]
    middle_x_series = non_minus_df["x"] + (non_minus_df["w"]/2).astype(int)

    for index, row, in minus_df.iterrows():
        minus_bottom = row["y"] + row["h"]
        first_mask = row["x"] < middle_x_series
        second_mask = (row["x"]+row["w"]) > middle_x_series
        third_mask = (minus_bottom + (2/3) * row["w"]) > highest_y_series
        fourth_mask = minus_bottom < non_minus_df["y"]
        combined_mask = first_mask & second_mask & third_mask & fourth_mask
        if combined_mask.sum()>0:
            final_bool_mask[index] = True
    return final_bool_mask

def replace_fraction(bounding_boxes_df):
    minus_mask = bounding_boxes_df["label"]=="-"
    minus_df = bounding_boxes_df[minus_mask]
    non_minus_df = bounding_boxes_df[~minus_mask]
    above_mask = above_fraction(minus_df,non_minus_df)
    below_mask = below_fraction(minus_df,non_minus_df)
    final_mask = above_mask & below_mask
    if final_mask.sum()>0:
        frac_indeces = list(final_mask[final_mask].index)
        bounding_boxes_df["label"][frac_indeces] = "frac"
    return bounding_boxes_df