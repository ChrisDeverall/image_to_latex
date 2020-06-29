import pandas as pd

def same_x_coords(minus_df):
    final_bool_mask = pd.Series(False, index = minus_df.index)
    for i in list(minus_df["x"]):
        x_subtracted = abs(minus_df["x"] - i)
        bool_mask = (x_subtracted <= 1)
        if bool_mask.sum()>1: # thus there exists a match
            final_bool_mask = (final_bool_mask) | (bool_mask)
    return final_bool_mask

def same_width(minus_df):
    final_bool_mask = pd.Series(False, index = minus_df.index)
    for i in list(minus_df["w"]):
        w_subtracted = abs(minus_df["w"] - i)
        bool_mask = (w_subtracted <= 1)
        if bool_mask.sum()>1: # thus there exists a match
            final_bool_mask = (final_bool_mask) | (bool_mask)
    return final_bool_mask

def nearby_y_coords(minus_df):
    final_bool_mask = pd.Series(False, index = minus_df.index)
    for index,row in minus_df.iterrows():
        y_subtracted = abs(minus_df["y"] - row["y"])
        bool_mask = y_subtracted < (6 * row["h"])
        if bool_mask.sum() > 1:
            final_bool_mask = (final_bool_mask) | bool_mask
    return final_bool_mask

def replace_equals(bounding_boxes_df):
    minus_df = bounding_boxes_df[bounding_boxes_df["label"]=="-"]
    first_mask = same_x_coords(minus_df)
    second_mask = nearby_y_coords(minus_df)
    third_mask = same_width(minus_df)
    final_mask = first_mask & second_mask & third_mask
    
    if final_mask.sum()==2: #when 2 lines pass the criteria
        [index_1, index_2] = list(final_mask[final_mask].index[:2])
        x = min(bounding_boxes_df["x"][index_1], bounding_boxes_df["x"][index_2])
        y = min(bounding_boxes_df["y"][index_1], bounding_boxes_df["y"][index_2])
        w = max (bounding_boxes_df["w"][index_1], bounding_boxes_df["w"][index_2])
        bot_point = max(bounding_boxes_df["y"][index_1]+bounding_boxes_df["h"][index_1], bounding_boxes_df["y"][index_2]+bounding_boxes_df["h"][index_2])
        h = bot_point - y
        equals_dictionary = {"x" :x ,"y":y,"w":w,"h":h,"label":"="}
        bounding_boxes_df = bounding_boxes_df.drop([index_1, index_2])
        bounding_boxes_df = bounding_boxes_df.append(equals_dictionary, ignore_index = True)
        bounding_boxes_df = bounding_boxes_df.reset_index(drop=True)
        return bounding_boxes_df
    else:
        return bounding_boxes_df
    return df_with_equals