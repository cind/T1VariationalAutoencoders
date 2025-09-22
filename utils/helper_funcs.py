import os
import numpy as np
import pandas as pd
import datetime as dt

def within_timeperiod(date1, date1_format, date2, date2_format, ndays):
	"""Checks that date1 and date2 are no more than ndays apart"""
	date1=dt.datetime.strptime(date1, date1_format)
	date2=dt.datetime.strptime(date2, date2_format)
	backward = date1 - dt.timedelta(days=ndays)
	forward = date1 + dt.timedelta(days=ndays)
	if date2 >= backward and date2 <= forward:
		return True
	else:
		return False	

def find_shortest_date_pair_idx(date_dict):
    """Find index with shortest # days between pair of dates
       date_dict must have keys as indices and values are tuples of date string pairs"""
    min_diff=None
    closest_idx=None
    for idx, (date1_str, date2_str) in date_dict.items():
        date1=dt.datetime.strptime(date1_str, '%Y-%m-%d').date()
        date2=dt.datetime.strptime(date2_str, '%Y-%m-%d').date()
        diff=abs((date2 - date1).days)
        if min_diff is None or diff < min_diff:
            min_diff=diff
            closest_idx=idx
    return closest_idx        

