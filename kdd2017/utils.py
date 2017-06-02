#! /usr/bin/env python
# -*- coding: utf-8 -*-

# import necessary modules
import math
from datetime import datetime
from sklearn import preprocessing
import numpy as np
import copy
from datetime import timedelta
import collections
from sklearn.decomposition import IncrementalPCA
from scipy.stats import boxcox
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import itertools
import collections
from multiprocessing import Pool, TimeoutError


def findsubsets(S,m):
    return set(itertools.combinations(S, m))

def findsubsets2(S, add_origin=True):
    ret = set()
    for i in range(1, len(S)):
        ret = ret.union(findsubsets(S, i))
    print("S=", S)
    if add_origin:
        ret.add(tuple(S))
    return ret


def generate_final_volumes(volumes):
    volumes_final = {}
    tollgate_id_dirs = set()
    for start_time_window in volumes:
        for tollgate_id in volumes[start_time_window]:
            for direction in volumes[start_time_window][tollgate_id]:
                tollgate_id_dirs.add((tollgate_id, direction))

    tollgate_id_dirs = list(tollgate_id_dirs)
    finnal_predict_times1 = [
                                (datetime(2016,10,i,8), datetime(2016,10,i,10))
                                for i in range(25, 32)
                            ]
    finnal_predict_times2 = [
                                (datetime(2016,10,i,17), datetime(2016,10,i,19))
                                for i in range(25, 32)
                            ]
    finnal_predict_times = finnal_predict_times1 + finnal_predict_times2
    for tollgate_id, direction in tollgate_id_dirs:
        predict_datetimes = []
        for time_range in finnal_predict_times:
            start_datetime = time_range[0]
            end_datetime = time_range[1]
            cur_datetime = start_datetime
            while cur_datetime < end_datetime:
                predict_datetimes.append(cur_datetime)
                cur_datetime = cur_datetime + timedelta(minutes=20)
        for predict_datetime in predict_datetimes:
            if predict_datetime not in volumes_final:
                volumes_final[predict_datetime] = {}
            if tollgate_id not in volumes_final[predict_datetime]:
                volumes_final[predict_datetime][tollgate_id] = {}
            if direction not in volumes_final[predict_datetime][tollgate_id]:
                volumes_final[predict_datetime][tollgate_id][direction] = 1
    return volumes_final


def extract_is_work_day(cur_date):
    if cur_date.year == 2015:
        # http://news.sina.com.cn/c/2014-12-16/154731291679.shtml
        if cur_date.month == 1:
            if cur_date.day >= 1 and cur_date.day <= 3:
                return 0
            if cur_date.day == 4:
                return 1
        if cur_date.month == 2:
            if cur_date.day == 15 or cur_date.day == 28:
                return 1
            if cur_date.day >= 18 and cur_date.day <= 24:
                return 0
        if cur_date.month == 4:
            if cur_date.day >= 4 and cur_date.day <= 6:
                return 0
        if cur_date.month == 5:
            if cur_date.day >= 1 and cur_date.day <= 3:
                return 0
        if cur_date.month == 6:
            if cur_date.day >= 20 and cur_date.day <= 22:
                return 0
        if cur_date.month == 9:
            if cur_date.day >= 26 and cur_date.day <= 27:
                return 0
        if cur_date.month == 10:
            if cur_date.day >= 1 and cur_date.day <= 7:
                return 0
            if cur_date.day == 10:
                return 1
    if cur_date.year == 2016:
        # http://news.qq.com/cross/20151211/xK0R05S8.html
        if cur_date.month == 1:
            if cur_date.day >= 1 and cur_date.day <= 3:
                return 0
        if cur_date.month == 2:
            if cur_date.day >= 7 and cur_date.day <= 13:
                return 0
            if cur_date.day == 6 or cur_date.day == 4:
                return 1
        if cur_date.month == 4:
            if cur_date.day >= 2 and cur_date.day <= 4:
                return 0
        if cur_date.month == 5:
            if cur_date.day >= 1 and cur_date.day <= 2:
                return 0
        if cur_date.month == 6:
            if cur_date.day >= 9 and cur_date.day <= 11:
                return 0
            if cur_date.day == 12:
                return 1
        if cur_date.month == 9:
            if cur_date.day >= 15 and cur_date.day <= 17:
                return 0
            if cur_date.day == 18:
                return 1
        if cur_date.month == 10:
            if cur_date.day >= 1 and cur_date.day <= 7:
                return 0
            if cur_date.day == 8 or cur_date.day == 9:
                return 1
    if cur_date.weekday() == 6:
        return 0
    if cur_date.weekday() == 5:
        return 0
    return 1
    #return cur_date.weekday != 6 and cur_date.weekday != 5


def load_links(path_links):
    lines = open(path_links, "r").readlines()
    link_data = {}
    for line in lines[1:]:
        #line = line.replace('"', '')
        line = line.strip()
        words = line.split('","')
        link_id = int(words[0].replace('"', ''))
        length = int(words[1])
        width = int(words[2])
        lanes = int(words[3])
        in_top = words[4].split(",")
        out_top = words[5].split(",")
        lane_width = int(words[6].replace('"', ''))
        link_data[link_id] = {}
        link_data[link_id]["length"] = length
        link_data[link_id]["width"] = width
        link_data[link_id]["lanes"] = lanes
        link_data[link_id]["in_top"] = in_top
        link_data[link_id]["out_top"] = out_top
        link_data[link_id]["lane_width"] = lane_width
    return link_data


def load_routes(path_routes):
    lines = open(path_routes, "r").readlines()
    routes_data = {}
    for line in lines[1:]:
        line = line.strip()
        line = line.replace('"', '')
        words = line.split(",")
        intersection_id = words[0]
        tollgate_id = int(words[1])
        link_seq = [int(link_id) for link_id in words[2:]]
        routes_data[(intersection_id, tollgate_id)] = link_seq
    return routes_data

def load_weather_info(path_weather_infos):
    datetime_weather = {}
    for path_weather_info in path_weather_infos:
        is_first_line = True
        for line in open(path_weather_info, "r"):
            if is_first_line:
                is_first_line = False
                continue
            line = line.replace('"', '')
            words = line.split(",")
            trace_start_time = words[0]
            trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d")
            hour = int(words[1])
            trace_start_time = datetime(trace_start_time.year,
                                        trace_start_time.month,
                                        trace_start_time.day,
                                        hour)
            datetime_weather[trace_start_time] = {}
            datetime_weather[trace_start_time]["pressure"] = float(words[2])
            datetime_weather[trace_start_time]["sea_pressure"] = float(words[3])
            datetime_weather[trace_start_time]["wind_direction"] = float(words[4])
            datetime_weather[trace_start_time]["wind_speed"] = float(words[5])
            datetime_weather[trace_start_time]["temperature"] = float(words[6])
            datetime_weather[trace_start_time]["rel_humidity"] = float(words[7])
            datetime_weather[trace_start_time]["precipitation"] = float(words[8])
    return datetime_weather


def load_volumes_info(in_file_names):

    if not isinstance(in_file_names, list):
        in_file_names = [in_file_names]

    volumes = {}
    for in_file_name in in_file_names:
        # Step 1: Load volume data
        fr = open(in_file_name, 'r')
        fr.readline()  # skip the header
        vol_data = fr.readlines()
        fr.close()

        # Step 2: Create a dictionary to caculate and store volume per time window
        # volumes = {}  # key: time window value: dictionary
        for i in range(len(vol_data)):
            each_pass = vol_data[i].replace('"', '').split(',')
            tollgate_id = each_pass[1]
            direction = each_pass[2]

            pass_time = each_pass[0]
            pass_time = datetime.strptime(pass_time, "%Y-%m-%d %H:%M:%S")
            time_window_minute = int(math.floor(pass_time.minute / 20) * 20)
            #print pass_time
            start_time_window = datetime(pass_time.year, pass_time.month,
                                         pass_time.day,
                                         pass_time.hour, time_window_minute, 0)

            if start_time_window not in volumes:
                volumes[start_time_window] = {}
            if tollgate_id not in volumes[start_time_window]:
                volumes[start_time_window][tollgate_id] = {}
            if direction not in volumes[start_time_window][tollgate_id]:
                volumes[start_time_window][tollgate_id][direction] = 1
            else:
                volumes[start_time_window][tollgate_id][direction] += 1
    return volumes




def load_travel_times_from_trajectories(path_trajectorieses, skip_date_ranges,
                                        load_frequent_info=False,
                                        frequent_threshold=1):

    if isinstance(path_trajectorieses, basestring):
        path_trajectorieses = [path_trajectorieses]
    elif isinstance(path_trajectorieses, list):
        path_trajectorieses = path_trajectorieses
    else:
        raise ValueError("unknown format...")

    travel_times = {}
    for path_trajectories in path_trajectorieses:
        # Step 1: Load trajectories
        fr = open(path_trajectories, 'r')
        fr.readline()  # skip the header
        traj_data = fr.readlines()
        fr.close()
        # print(traj_data[0])

        vehicle_id_f = collections.defaultdict(lambda : 0)
        all_route_ids = set()
        ## compute the vehicule
        for i in range(len(traj_data)):
            each_traj = traj_data[i].replace('"', '').split(',')
            intersection_id = each_traj[0]
            tollgate_id = each_traj[1]
            vehicle_id = each_traj[2]
            route_id = intersection_id + '-' + tollgate_id
            all_route_ids.add(route_id)
            vehicle_id_f[(route_id, vehicle_id)] += 1

        frequent_route_ids = set()
        for key in vehicle_id_f:
            if vehicle_id_f[key] > 1:
                route_id = key[0]
                frequent_route_ids.add(route_id)
                #print vehicle_id_f[key]
        if all_route_ids == frequent_route_ids:
            print("enough paramters")
        else:
            raise ValueError("not info...")

        # Step 2: Create a dictionary to store travel time for each route per time window
        # travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
        # travel_times_avg = collections.defaultdict(list)
        for i in range(len(traj_data)):
            each_traj = traj_data[i].replace('"', '').split(',')
            intersection_id = each_traj[0]
            tollgate_id = each_traj[1]
#            vehicle_id = each_traj[2]

            route_id = intersection_id + '-' + tollgate_id

            if load_frequent_info:
                key = (route_id, vehicle_id)
                if key not in vehicle_id_f or vehicle_id_f[key] <= frequent_threshold:
                    continue

            if route_id not in travel_times.keys():
                travel_times[route_id] = {}

            trace_start_time = each_traj[3]
            trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
            time_window_minute = math.floor(trace_start_time.minute / 20) * 20
            start_time_window = datetime(trace_start_time.year,
                                         trace_start_time.month,
                                         trace_start_time.day,
                                         trace_start_time.hour,
                                         int(time_window_minute),
                                         0)
            is_need_skip = False
            for skip_date_range in skip_date_ranges:
                if start_time_window >= skip_date_range[0] and \
                        start_time_window < skip_date_range[1]:
                    is_need_skip = True
                    break
            if is_need_skip:
                continue
            tt = float(each_traj[-1]) # travel time
            if start_time_window not in travel_times[route_id].keys():
                travel_times[route_id][start_time_window] = [tt]
            else:
                travel_times[route_id][start_time_window].append(tt)
            if trace_start_time.hour >= 8 and trace_start_time.hour <= 10 \
                 or trace_start_time.hour >= 17 and trace_start_time.hour <= 19:
                key = (route_id, trace_start_time.hour, int(time_window_minute))
                #travel_times_avg[key].append(tt)

#    for key in travel_times_avg:
#        print(key)
#        print(np.median(travel_times_avg[key]))
#
#    exit(0)

    return travel_times

def search_closest_date_weather_info(date, datetime_weather):
    closest_date = None
    for cur_date in datetime_weather:
        if closest_date is None or (cur_date <= date and closest_date < cur_date):
            closest_date = cur_date
    #print("closest_date=", closest_date)
    return datetime_weather[closest_date]



def convert_date_to_x_volumes(date, datetime_weather, tollgate_id_x):
    weather_info = search_closest_date_weather_info(
        date, datetime_weather)
    x = []
    #x.append(date.year)
    x.append(date.month) ## 2
    #x.append(date.day)
    x.append(date.hour) ## 3
    x.append(date.minute) ## 4
    x.append(date.weekday()) ## 5
    x.append(not extract_is_work_day(date)) # 6
    x.append(float(weather_info["wind_speed"])) ## 7
    x.append(float(weather_info["temperature"])) ## 8
    x.append(float(weather_info["rel_humidity"])) ## 9
    x.append(float(weather_info["precipitation"])) ## 10
    wod = [0] * 7
    wod[date.weekday()] = 1
    x += wod ## 11 - 17
    x += list(tollgate_id_x) # 18 - 41 (include)
    x += [date.weekday()==5 or date.weekday()==6] ## weekend 42
    return x


def transform_data(x_i, le, with_fit=True):
    if isinstance(le, preprocessing.MinMaxScaler) or isinstance(le, preprocessing.Binarizer):
        x_i = x_i.astype(np.float)
        if with_fit:
            le.fit(x_i.reshape((-1, 1)))
        x_i = le.transform(x_i.reshape((-1, 1)))
    elif isinstance(le, preprocessing.LabelEncoder):
        if with_fit:
            le.fit(x_i)
        x_i = le.transform(x_i.reshape((-1, 1)))
    else:
        raise ValueError("unknow transform")
    return x_i.reshape((-1)), le

def compute_tollgate_id_to_link_ids_x(routes_data,):
    link_ids = []

    tollgate_id_to_link_ids = collections.defaultdict(list)
    tollgate_id = set()
    for it in routes_data:
        tollgate_id_to_link_ids[it[1]] += routes_data[it]
        tollgate_id_to_link_ids[it[1]] = list(set(tollgate_id_to_link_ids[it[1]]))
        link_ids += routes_data[it]

    all_link_ids = list(set(link_ids))
    tollgate_id_to_x = {}
    for tollgate_id in tollgate_id_to_link_ids:
        x = [0] * len(all_link_ids)
        x = np.asarray(x)
        link_ids = tollgate_id_to_link_ids[tollgate_id]
        for link_id in link_ids:
            #print("all_link_ids=", all_link_ids)
            #print("link_id=", link_id)
            pos = all_link_ids.index(link_id)
            #print("pos=", pos)
            #print("x=", x)
            x[pos] = 1
        tollgate_id_to_x[tollgate_id] = x
    return tollgate_id_to_x

def convert_volumes_into_X_y(volumes, datetime_weather, link_data, routes_data,
                            les_train=None,
                            is_ret_raw_info=False, verbose=False):

    tollgate_id_to_x = compute_tollgate_id_to_link_ids_x(routes_data)
    tollgate_id_to_x_len = len(tollgate_id_to_x[tollgate_id_to_x.keys()[0]])
    print("tollgate_id_to_x_len=", tollgate_id_to_x_len)
    X = []
    y = []
    les = [
           preprocessing.LabelEncoder(), ## tollgate_id
           preprocessing.LabelEncoder(), ## direction
           preprocessing.LabelEncoder(), ## month
           preprocessing.LabelEncoder(), ## hour
           preprocessing.LabelEncoder(), ## minute
           preprocessing.LabelEncoder(), ## weekday
           preprocessing.LabelEncoder(), ## work day
           preprocessing.MinMaxScaler(), ## wind
           preprocessing.MinMaxScaler(), ## temperature
           preprocessing.MinMaxScaler(), ## rel_humidity
           preprocessing.MinMaxScaler(), ## precipitation
          ]
    wod_les = [preprocessing.LabelEncoder()] * 7
    les += wod_les
    les += [preprocessing.LabelEncoder()] * tollgate_id_to_x_len
    les += [preprocessing.LabelEncoder(),]
    les += [preprocessing.MinMaxScaler(), ] ## future hour feature
    dates = []
    raw_info = []
    for start_time_window in volumes:
        for tollgate_id in volumes[start_time_window]:
            #tollgate_id = int(tollgate_id)
            for direction in volumes[start_time_window][tollgate_id]:
                dates.append(start_time_window)
                x = []
                x.append(tollgate_id)
                x.append(direction)
                #print("tollgate_id=", tollgate_id)
                x += convert_date_to_x_volumes(start_time_window, datetime_weather, tollgate_id_to_x[int(tollgate_id)])
                X.append(x)
                y.append(volumes[start_time_window][tollgate_id][direction])
                raw_info.append((tollgate_id, direction, start_time_window))
    X = np.asarray(X)
    y = np.asarray(y)

    X = add_prev_two_hour_ft(X, y, dates)
    print("X[:5,:]=", X[:5,:])
    print("y[:5]=", y[:5])
    for i in range(X.shape[1]):
        if les_train is None:
            le = les[i]
            x_i = X[:, i]
            x_i, le = transform_data(x_i, le, True)
            X[:, i] = x_i
        else:
            le = les_train[i]
            x_i = X[:, i]
            x_i, le = transform_data(x_i, le, False)
            X[:, i] = x_i
    X = np.asarray(X, np.float)


    dates = np.asarray(dates)
    sorted_indexes = sorted(range(len(dates)), key = lambda x: dates[x])
    X = X[sorted_indexes, :]
    y = y[sorted_indexes]
    dates = dates[sorted_indexes]
    if not is_ret_raw_info:
        if les_train is None:
            return X, y, dates, les
        else:
            return X, y, dates, les_train
    else:
        new_raw_info = []
        for i in sorted_indexes:
            new_raw_info.append(raw_info[i])
        raw_info = new_raw_info
        if les_train is None:
            return X, y, dates, les, raw_info
        else:
            return X, y, dates, les_train, raw_info


def convert_date_to_x(date, datetime_weather, link_ids_x):

    #print("len(link_ids_x)=", len(link_ids_x))
    weather_info = search_closest_date_weather_info(
        date, datetime_weather)

    x = []
    #x.append(date.year)
    x.append(date.month) ## 2
    x.append(float(date.day)) ## 3
    x.append(date.hour) ## 4
    x.append(date.minute) ## 5
    x.append(date.weekday()) ## 6
    x.append(not extract_is_work_day(date)) ## 7
    x.append(float(weather_info["wind_speed"])) ## 8
    x.append(float(weather_info["temperature"])) ## 9
    x.append(float(weather_info["rel_humidity"])) ## 10
    x.append(float(weather_info["precipitation"])) ## 11
    wod = [0] * 7
    wod[date.weekday()] = 1
    x += wod # 12 - 18
    x += list(link_ids_x) # 19 - 42 (include)
    return x


def compute_route_to_link_ids_x(routes_data,):
    link_ids = []
    tollgate_id = set()
    print("routes_data=", routes_data)
    for it in routes_data:
        link_ids += routes_data[it]
    all_link_ids = list(set(link_ids))

    route_to_x = {}
    for route_id in routes_data:
        x = [0] * len(all_link_ids)
        x = np.asarray(x)
        link_ids = routes_data[route_id]
        for link_id in link_ids:
            #print("all_link_ids=", all_link_ids)
            #print("link_id=", link_id)
            pos = all_link_ids.index(link_id)
            #print("pos=", pos)
            #print("x=", x)
            x[pos] = 1
        route_to_x[route_id] = x
    return route_to_x


def search_prev_two_hour_y(cur_date, y, dates):
    prev_cur_date = cur_date - timedelta(hours=2)
    for i, date in enumerate(dates):
        if date == prev_cur_date:
            return y[i]
    return 0

def add_prev_two_hour_ft(X, y, dates):
    addition_x = []
    for i, date in enumerate(dates):
        addition_x.append(search_prev_two_hour_y(date, y, dates))
    addition_x = np.asarray(addition_x).reshape((-1, 1))
    print("X.shape=", X.shape)
    print("addition_x.shape=", len(addition_x))
    X = np.hstack([X, np.asarray(addition_x)])
    return X

def convert_into_X_y(travel_times, datetime_weather, link_data, routes_data,
                     les_train=None, is_ret_raw_info=False, is_skip_not_trainning_hours=False, verbose=False):

    route_to_link_ids_x = compute_route_to_link_ids_x(routes_data)
    route_to_link_ids_x_len = len(route_to_link_ids_x[route_to_link_ids_x.keys()[0]])
    X = []
    y = []
    les = [
           preprocessing.LabelEncoder(),
           preprocessing.LabelEncoder(),
           preprocessing.LabelEncoder(),
           preprocessing.LabelEncoder(),
           preprocessing.LabelEncoder(),
           preprocessing.LabelEncoder(),
           preprocessing.LabelEncoder(),
           preprocessing.LabelEncoder(),
           preprocessing.MinMaxScaler(), ## wind speed
           preprocessing.MinMaxScaler(), ## temperature
           preprocessing.MinMaxScaler(), ## rel_humidity
           preprocessing.MinMaxScaler(), ## precipitation
          ]
    wod_les = [preprocessing.LabelEncoder()] * 7
    les += wod_les
    les += [preprocessing.MinMaxScaler(), ] * route_to_link_ids_x_len
    les += [preprocessing.MinMaxScaler(), ]
    dates = []
    raw_info = []
    for route_id in travel_times:
        if verbose:
            print("route_id=", route_id)
        for start_time_window in travel_times[route_id]:
            is_trainning_hour = False
            start_year = start_time_window.year
            start_month = start_time_window.month
            start_day = start_time_window.day
            if start_time_window >= datetime(start_year, start_month, start_day, 8) and \
                    start_time_window < datetime(start_year, start_month, start_day, 10):
                is_trainning_hour = True
            if start_time_window >= datetime(start_year, start_month, start_day, 17) and \
                    start_time_window < datetime(start_year, start_month, start_day, 19):
                is_trainning_hour = True
            if (not is_trainning_hour) and is_skip_not_trainning_hours:
                continue
            if verbose:
                print("start_time_window=", start_time_window)
            
            raw_info.append((route_id, start_time_window))
            A, B = route_id.split('-')
            x = []
            #x.append(route_id)
            x.append(A)
            x.append(B)
            route_id_for_link_id = (A, int(B))
            #print(route_to_link_ids_x[route_id_for_link_id])
            x += convert_date_to_x(start_time_window, datetime_weather, route_to_link_ids_x[route_id_for_link_id])
            #print("x=", x)
            if len(travel_times[route_id][start_time_window]) >= 1:
                X.append(x)
                y.append(np.average(travel_times[route_id][start_time_window]))
                dates.append(start_time_window)
    X = np.asarray(X)
    y = np.asarray(y)
    X = add_prev_two_hour_ft(X, y, dates)
    print("X=", X)
    print("X[:5,:]=", X[:5,:])
    print("y[:5]=", y[:5])
    for i in range(X.shape[1]):
        if les_train is None:
            le = les[i]
            x_i = X[:, i]
            x_i, le = transform_data(x_i, le, True)
            X[:, i] = x_i
        else:
            le = les_train[i]
            x_i = X[:, i]
            x_i, le = transform_data(x_i, le, False)
            X[:, i] = x_i
    X = np.asarray(X, np.float)
    dates = np.asarray(dates)
    sorted_indexes = sorted(range(len(dates)), key = lambda x: dates[x])
    X = X[sorted_indexes, :]
    y = y[sorted_indexes]
    dates = dates[sorted_indexes]

#    if les_train is None:
#        print("LabelEncoder........info")
#        for le in les:
#            print(list(le.classes_))

    if not is_ret_raw_info:
        if les_train is None:
            return X, y, dates, les
        else:
            return X, y, dates, les_train
    else:
        new_raw_info = []
        for i in sorted_indexes:
            new_raw_info.append(raw_info[i])
        raw_info = new_raw_info
        if les_train is None:
            return X, y, dates, les, raw_info
        else:
            return X, y, dates, les_train, raw_info


#Function
def invboxcox(y,ld):
   if ld == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(ld*y+1)/ld))


def mape_loss(y, y_predict):
    loss = np.sum(np.abs((y_predict - y) / y)) / float(len(y))
    if np.isnan(loss):
        return 100.0
    else:
        return loss

def inv_mape_loss(estimator, y, y_predict):
    return 1.0 - mape_loss(y, y_predict)

def remove_outliers(X, y, dates, m=3.0):
    ret_keep_indexes = np.asarray([], dtype=int)
    route_ids = set(X[:, 0])
    keep_indexes = np.asarray(range(len(y)))
    for route_id in route_ids:
        route_items = X[:, 0] == route_id
        y_route = y[route_items]
        keep_indexes_route = keep_indexes[route_items]
        avg_y_route = np.average(y_route)
        m_diff_y = y_route - avg_y_route
        stable_indexes = keep_indexes_route[
            abs(m_diff_y) < m * np.std(m_diff_y)]
        ret_keep_indexes = np.append(ret_keep_indexes, stable_indexes)
    ret_keep_indexes.sort()
    return X[ret_keep_indexes], y[ret_keep_indexes], dates[ret_keep_indexes]


def remove_outliers2(X, y, dates, m=3.0):
    ret_keep_indexes = np.asarray([], dtype=int)
    route_ids = set(X[:, 0])
    keep_indexes = np.asarray(range(len(y)))
    time_keys = set()
    for start_time_window in dates:
        key_str = "%02d%02d" % (
            start_time_window.hour,
            start_time_window.minute)
        time_keys.add(key_str)
    for time_key in time_keys:
        for route_id in route_ids:
            route_items = X[:, 0] == route_id
            X_route = X[route_items,:]
            y_route = y[route_items]
            dates_route = dates[route_items]
            keep_indexes_route = keep_indexes[route_items]
            inner_time_items = np.asarray([False] * len(dates_route))
            for i, date in enumerate(dates_route):
                key_str = "%02d%02d" % (
                    start_time_window.hour,
                    start_time_window.minute)
                if key_str == time_key:
                    inner_time_items[i] = True
            X_route_time = X_route[inner_time_items]
            y_route_time = y_route[inner_time_items]
            keep_indexes_route_time = keep_indexes_route[inner_time_items]
            avg_y_route_time = np.average(y_route_time)
            m_diff_y = y_route_time - avg_y_route_time
            stable_indexes = keep_indexes_route_time[
                abs(m_diff_y) < m * np.std(m_diff_y)]
            ret_keep_indexes = np.append(ret_keep_indexes, stable_indexes)
    ret_keep_indexes.sort()


def remove_outliers3(X, y, dates, m=3.0, key=[0,1], is_avg_or_median=True):
    if len(key) == 0:
        return X, y, dates

    key = np.asarray(key)
    key = key[key < X.shape[1]]

    ret_keep_indexes = np.asarray([], dtype=int)
    entities = set()
    for entity in X[:, key]:
        entities.add(tuple(list(entity)))
    keep_indexes = np.asarray(range(len(y)))
    for entity in entities:
        entity_items = np.all(X[:, key] == entity, axis=1)
        y_entities = y[entity_items]
        keep_indexes_route = keep_indexes[entity_items]
        if is_avg_or_median == 1:
            avg_y_entity = np.average(y_entities)
            m_diff_y = y_entities - avg_y_entity
            stable_indexes = keep_indexes_route[
                abs(m_diff_y) < m * np.std(m_diff_y)]
        elif is_avg_or_median == 0:
            d = np.abs(y_entities - np.median(y_entities))
            mdev = np.median(d)
            if mdev != 0:
                s = d/mdev
            else:
                s = [0.0,] * len(d)
            s = np.asarray(s)
            stable_indexes = keep_indexes_route[s<m]
        elif is_avg_or_median == 2:
            len_y_entities = len(y_entities)
            sorted_indexes = sorted(range(len_y_entities), key = lambda x: y_entities[x])
            edge = int(len_y_entities*m/10.0/2.0)
            #print("len_y_entities=", len_y_entities)
            #print("edge=", edge)
            if edge < len_y_entities / 2 and edge > 0:
                sorted_indexes = sorted_indexes[edge:-edge]
                stable_indexes = keep_indexes_route[sorted_indexes]
            elif edge == 0:
                stable_indexes = keep_indexes_route[sorted_indexes]
            else:
                stable_indexes = np.asarray([])
        elif is_avg_or_median == 3:
            avg_y_entity = np.average(y_entities)
            m_diff_y = y_entities - avg_y_entity
            stable_indexes = keep_indexes_route[
                m_diff_y < m * np.std(m_diff_y)]
        else:
            raise ValueError("is_avg_or_median unknown....")
        ret_keep_indexes = np.append(ret_keep_indexes, stable_indexes)
    ret_keep_indexes.sort()
    return X[ret_keep_indexes], y[ret_keep_indexes], dates[ret_keep_indexes]


#def remove_outliers_by_classifier(X, y, dates, ):
    
def compute_loss(input_compute_loss):

    Model = input_compute_loss["Model"]
    config = input_compute_loss["config"]
    X_train = input_compute_loss["X_train"]
    y_train = input_compute_loss["y_train"]
    dates_train = input_compute_loss["dates_train"]
    X_test = input_compute_loss["X_test"]
    y_test = input_compute_loss["y_test"]
    is_y_log = input_compute_loss["is_y_log"]
    is_boxcox = input_compute_loss["is_boxcox"]
    loss_func = input_compute_loss["loss_func"]

    model = Model(**config)
    if hasattr(model ,"dates_train"):
        model.dates_train = dates_train
    if is_y_log:
        model.fit(X_train, np.log(y_train))
        predict_y_test = np.exp(model.predict(X_test))
    elif is_boxcox:
        model.fit(X_train, boxcox(y_train, boxcox_lambda))
        predict_y_test = invboxcox(model.predict(X_test), boxcox_lambda)
    else:
        model.fit(X_train, y_train)
        predict_y_test = model.predict(X_test)
    if loss_func is None:
        loss = mape_loss(y_test, predict_y_test)
    else:
        loss = loss_func(y_test, predict_y_test)
    return (repr(config), config, loss)

def compute_losses(input_compute_losses):
    result = []
    for input_compute_loss in input_compute_losses:
        result.append(compute_loss(input_compute_loss))
    return result
 
def GridSearchCVDates(Model, tuned_parameters, X, y, dates,
                      days_to_test=7, cv=5, loss_func=None,
                      is_y_log=False, is_boxcox=False, boxcox_lambda=1.0,
                      X_val=None, y_val=None, dates_val=None, skip_cvs=[],
                      is_estimate_val=True, estimate_val_w=2.0,
                      is_include_future_training=False,
                      test_date_ranges=None, n_cores=2):
    global_configs = []
    for tuned_parameter in tuned_parameters:
        configs = []
        init_config = {}
        for key in tuned_parameter:
            init_config[key] = tuned_parameter[key][0]
        configs.append(copy.copy(init_config))
        for key in tuned_parameter:
            tmp_configs = []
            for i in range(1, len(tuned_parameter[key])):
                for iconfig in configs:
                    tmp_config = copy.copy(iconfig)
                    tmp_config[key] = tuned_parameter[key][i]
                    tmp_configs.append(tmp_config)
            configs += tmp_configs
        global_configs += configs
    if len(global_configs) == 0:
        global_configs.append({})
    max_date = dates[-1] + timedelta(days=1)
    report = collections.defaultdict(list)
    mem_config = {}
    nrepeat = 1

    if test_date_ranges is None:
        test_date_ranges = []
        for p, i in enumerate(range(cv) * nrepeat):
            if i in skip_cvs:
                continue
            max_train_date = max_date - timedelta(days=(i + 1) * days_to_test)
            max_test_date = max_date - timedelta(days=(i) * days_to_test)
            test_date_ranges.append((max_train_date, max_test_date))
            print("range=", max_train_date, max_test_date)
    else:
        for max_train_date, max_test_date in test_date_ranges:
            print("range=", max_train_date, max_test_date)

    start_time = datetime.now()
    for p, date_time_range_item in enumerate(test_date_ranges*nrepeat):
        max_train_date, max_test_date = date_time_range_item
        train_items = dates < max_train_date
        test_items = np.asarray([False] * len(dates))
        inc_one_step = timedelta(days=1)
        cur_datetime = max_train_date
        while cur_datetime < max_test_date:
            ## training
            if is_include_future_training:
                tmp_range_items = np.logical_and(
                    dates >= datetime(
                        cur_datetime.year, cur_datetime.month, cur_datetime.day,
                        hour=6),
                    dates < datetime(
                        cur_datetime.year, cur_datetime.month, cur_datetime.day,
                        hour=8))
                train_items = np.logical_or(
                    train_items,
                    tmp_range_items
                    )
                tmp_range_items = np.logical_and(
                    dates >= datetime(
                        cur_datetime.year, cur_datetime.month, cur_datetime.day,
                        hour=15),
                    dates < datetime(
                        cur_datetime.year, cur_datetime.month, cur_datetime.day,
                        hour=17))
                train_items = np.logical_or(
                    train_items,
                    tmp_range_items
                    )
            ## testing
            tmp_range_items = np.logical_and(
                dates >= datetime(
                    cur_datetime.year, cur_datetime.month, cur_datetime.day,
                    hour=8),
                dates < datetime(
                    cur_datetime.year, cur_datetime.month, cur_datetime.day,
                    hour=10))
            test_items = np.logical_or(
                test_items,
                tmp_range_items
                )
            tmp_range_items = np.logical_and(
                dates >= datetime(
                    cur_datetime.year, cur_datetime.month, cur_datetime.day,
                    hour=17),
                dates < datetime(
                    cur_datetime.year, cur_datetime.month, cur_datetime.day,
                    hour=19))
            test_items = np.logical_or(
                test_items,
                tmp_range_items
                )
            cur_datetime += inc_one_step

        X_train = X[train_items]
        y_train = y[train_items]
        dates_train = dates[train_items]

        X_test = X[test_items]
        y_test = y[test_items]
        dates_test = dates[test_items]

        if len(y_test) == 0:
            continue

        input_compute_losses = []
        for ic, config in enumerate(global_configs):
            input_compute_loss = {}
            input_compute_loss["Model"] = Model
            input_compute_loss["config"] = config
            input_compute_loss["X_train"] = X_train
            input_compute_loss["y_train"] = y_train
            input_compute_loss["dates_train"] = dates_train
            input_compute_loss["X_test"] = X_test
            input_compute_loss["y_test"] = y_test
            input_compute_loss["is_y_log"] = is_y_log
            input_compute_loss["is_boxcox"] = is_boxcox
            input_compute_loss["loss_func"] = loss_func
            input_compute_losses.append(input_compute_loss)
        print("len(input_compute_losses)=", len(input_compute_losses))
        print("n_cores=", n_cores)
        ic = 0
        need_print_last_time = datetime.now()
        #pool = Pool(processes=n_cores)
        #for ret_val in pool.imap_unordered(compute_loss, input_compute_losses):
        for input_compute_loss in input_compute_losses:
            is_need_print = False
            diff_time_print = datetime.now() - need_print_last_time
            if diff_time_print.seconds > 30:
                is_need_print = True
                need_print_last_time = datetime.now()
            ret_val = compute_loss(input_compute_loss)
            if is_need_print:
                print("p=", p)
                print("ic %d/%d" % (ic, len(global_configs)))
            diff_time = datetime.now() - start_time
            cur_count = ic + p * len(global_configs) + 1
            total_count = len(global_configs) * nrepeat * len(test_date_ranges)

            config_key = ret_val[0]
            config = ret_val[1]
            loss = ret_val[2]

            report[repr(config)].append(loss)
            mem_config[repr(config)] = config
            if is_need_print:
                print("progress %d/%d" % (cur_count, total_count))
                remaining_seconds = diff_time.seconds / float(cur_count) * float(total_count - cur_count)
                print("remaining minutes = %d" % int(remaining_seconds/60))
                print("loss=", loss)
            ic += 1


    if is_estimate_val and not is_include_future_training:
        start_time = datetime.now()
        for nr in range(nrepeat):
            for ic, config in enumerate(global_configs):
                print("val part")
                print("ic %d/%d" % (ic, len(global_configs)))
                print("nr = ", nr)
                print("config=", config)
                diff_time = datetime.now() - start_time
                total_count = ic + nr * len(global_configs) + 1
                remaining_seconds = diff_time.seconds / float(total_count) * float(len(global_configs) * nrepeat - total_count)
                print("remaining minutes = %d" % int(remaining_seconds/60))
                model = Model(**config)
                if hasattr(model ,"dates_train"):
                    model.dates_train = copy.deepcopy(dates)
                if is_y_log:
                    model.fit(X, np.log(y))
                    predict_y_val = np.exp(model.predict(X_val))
                elif is_boxcox:
                    model.fit(X, boxcox(y, boxcox_lambda))
                    predict_y_val = invboxcox(model.predict(X_val), boxcox_lambda)
                else:
                    model.fit(X, y)
                    predict_y_val = model.predict(X_val)
                if loss_func is None:
                    loss = mape_loss(y_val, predict_y_val)
                else:
                    loss = loss_func(y_val, predict_y_val)
                for i in range(int(estimate_val_w)):
                    report[repr(config)].append(loss)
                print("loss=", loss)

    all_loss_configs = []
    for config in report:
        end_pos = len(report[config]) / 6 * 5
        all_loss_configs.append((
                                  np.average(report[config][:end_pos]), config, report[config], np.average(report[config]), np.std(report[config])
                               ))
    all_loss_configs = sorted(all_loss_configs, key=lambda x: x[0])
    print("summary:")
    for loss_config in all_loss_configs[::-1]:
        #print("loss_config=", loss_config)
        loss_val = loss_config[0]
        config_str = str(loss_config[1])
        avg_val = float(loss_config[3])
        std_val = float(loss_config[4])
        print("key loss=", loss_val , "avg loss=", avg_val, " std_val=", std_val, " all losses=", loss_config[2])
        print(config_str)
    print("end summary")
    best_config = all_loss_configs[0][1]
    best_loss = all_loss_configs[0][0]
    return mem_config[best_config], best_loss



def GridSearchCVDatesWithVal(Configurations,
                             X_train, y_train, dates_train,
                             X_val, y_val, dates_val,
                             X_final,
                             is_y_log=False, is_boxcox=False, boxcox_lambda=1.0,
                             is_include_val_loss_for_eval=True, cv=5, skip_cvs=[], days_to_test=7,
                             X_frequent_train=None, y_frequent_train=None, is_estimate_val=True,
                             estimate_val_w=2.0,
                             is_include_future_training=False, remove_future_training_test=True,
                             test_date_ranges=None, n_cores=2):
    ## search for best meta parameters
    report = []
    for config in Configurations:
        print("="*60)
        Model = config["model"]
        print(type(Model()).__name__)
        tuned_parameters = config["tuned_parameters"]
        best_config, best_loss = GridSearchCVDates(
            Model, tuned_parameters,
            X_train, y_train, copy.deepcopy(dates_train), loss_func=None,
            is_boxcox=is_boxcox, boxcox_lambda=boxcox_lambda,
            cv=cv, skip_cvs=skip_cvs,
            X_val=X_val, y_val=y_val, days_to_test=days_to_test,
            is_estimate_val=is_estimate_val, estimate_val_w=estimate_val_w,
            is_include_future_training=is_include_future_training, test_date_ranges=test_date_ranges, n_cores=n_cores)
        if is_include_val_loss_for_eval and (not is_include_future_training):
            cur_Model = Model
            cur_config = copy.deepcopy(best_config)
            cur_model = cur_Model(**cur_config)
            if hasattr(cur_model, "dates_train"):
                cur_model.dates_train = copy.deepcopy(dates_train)
            if is_y_log:
                cur_model.fit(X_train, np.log(y_train))
                predict_y_val = np.exp(cur_model.predict(X_val))
            elif is_boxcox:
                cur_model.fit(X_train, boxcox(y_train, boxcox_lambda))
                predict_y_val = invboxcox(cur_model.predict(X_val), boxcox_lambda)
            else:
                cur_model.fit(X_train, y_train)
                predict_y_val = cur_model.predict(X_val)
            val_loss = mape_loss(y_val, predict_y_val)
            best_loss = (best_loss + val_loss)/2.0
        report.append((best_loss, Model, best_config))

    report = sorted(report, key = lambda x: x[0])
    for item in report[::-1]:
        print("model=", type(item[1]()).__name__,
              " loss=", item[0],
              " best_config=", item[2])

    best_Model = report[0][1]
    best_config = report[0][2]
    best_model = best_Model(**best_config)

    if (not is_include_future_training) or remove_future_training_test:
        X_train_all = X_train
        y_train_all = y_train
        if hasattr(best_model, "dates_train"):
            #print("debug best_model has dates_train")
            best_model.dates_train = copy.deepcopy(dates_train)
    else:
        print("include future value for fitting")
        X_train_all = np.vstack([X_train, X_val])
        y_train_all = np.hstack([y_train, y_val])
        if hasattr(best_model, "dates_train"):
            dates_train_all = np.hstack([dates_train, dates_val])
            best_model.dates_train = copy.deepcopy(dates_train_all)
            if not all(dates_train_all[i] <= dates_train_all[i+1]
                    for i in xrange(len(dates_train_all)-1)):
                raise ValueError("train dates are not sorted...")
        #print("X_train_all=", X_train_all[-10:, :])
        #print("best_model.dates_train=", best_model.dates_train[-60:])
        #print("X_train_all.shape=", X_train_all.shape)
        #print("y_train_all.shape=", y_train_all.shape)
        #print("best_model.dates_train.shape=", best_model.dates_train.shape)

    if is_y_log:
        best_model.fit(X_train_all, np.log(y_train_all))
        predict_y_val = np.exp(best_model.predict(X_val))
        predict_y_final = np.exp(best_model.predict(X_final))
    elif is_boxcox:
        best_model.fit(X_train_all, boxcox(y_train_all, boxcox_lambda))
        predict_y_val = invboxcox(best_model.predict(X_val), boxcox_lambda)
        predict_y_final = invboxcox(best_model.predict(X_final), boxcox_lambda)
    else:
        best_model.fit(X_train_all, y_train_all)
        predict_y_val = best_model.predict(X_val)
        predict_y_final = best_model.predict(X_final)

    loss = mape_loss(y_val, predict_y_val)
    print("X_train_all[0]=", X_train_all[0])
    print("X_train_all[-1]=", X_train_all[-1])
    print("best_model=", type(best_model).__name__)
    print("best_config=", best_config)
    print("best model loss on val=", loss)
    return predict_y_final

#    for config in global_configs:
#        print config

def plot_travel_times_fix_hour(working_dir, traval_times):
    out_dir = "travel_times_fix_hour"
    abs_out_dir = os.path.join(working_dir, out_dir)
    if not os.path.isdir(abs_out_dir):
        os.makedirs(abs_out_dir)
    time_keys = set()
    for route_id in traval_times:
        for start_time_window in traval_times[route_id]:
            key_str = "%02d%02d" % (start_time_window.hour,
                                start_time_window.minute)
            time_keys.add(key_str)

    for time_key in time_keys:
        for route_id in traval_times:
            x = []
            y = []
            for start_time_window in traval_times[route_id]:
                key_str = "%02d%02d" % (start_time_window.hour,
                                    start_time_window.minute)
                if time_key == key_str:
                    x.append(start_time_window)
                    y.append(np.average(traval_times[route_id][start_time_window]))

            path_figure = os.path.join(abs_out_dir, "%s_%s.svg" % (route_id, time_key))
            plt.clf()
            plt.scatter(x, y, color="black", marker=".")
            plt.gcf().autofmt_xdate()
            plt.draw()
            plt.savefig(path_figure)

            path_figure = os.path.join(abs_out_dir, "%s_%s_log.svg" % (route_id, time_key))
            plt.clf()
            plt.scatter(x, np.log(y), color="black", marker="o")
            plt.gcf().autofmt_xdate()
            plt.draw()
            plt.savefig(path_figure)

def compute_harmonic_mean(predict_vals):
    predict_vals = np.vstack(predict_vals)
    predict_vals = 1.0/predict_vals
    print("predict_vals.shape=", predict_vals.shape)
    n_res = predict_vals.shape[0]
    print("n_res=", n_res)
    predict_vals = np.sum(predict_vals, axis=0)
    predict_vals = n_res / predict_vals
    return predict_vals.flatten()

def plot_travel_times_fix_date(working_dir, traval_times):
    out_dir = "travel_times_fix_date"
    abs_out_dir = os.path.join(working_dir, out_dir)
    if not os.path.isdir(abs_out_dir):
        os.makedirs(abs_out_dir)
    time_keys = set()
    for route_id in traval_times:
        for start_time_window in traval_times[route_id]:
            key_str = "%04d%02d%02d" % (
                int(start_time_window.year),
                int(start_time_window.month),
                int(start_time_window.day), )
            time_keys.add(key_str)

    for time_key in time_keys:
        for route_id in traval_times:
            x = []
            y = []
            for start_time_window in traval_times[route_id]:
                key_str = "%04d%02d%02d" % (
                    start_time_window.year,
                    start_time_window.month,
                    start_time_window.day, )
                if time_key == key_str:
                    x.append(int("%02d%02d" % (start_time_window.hour, start_time_window.minute)))
                    y.append(np.average(traval_times[route_id][start_time_window]))
            if len(y) == 0:
                continue
            sorted_indexes = sorted(range(len(y)), key=lambda i: x[i])
            x = np.asarray(x)
            y = np.asarray(y)
            x = x[sorted_indexes]
            y = y[sorted_indexes]
            #x = range(len(y))

            path_figure = os.path.join(abs_out_dir, "%s_%s.svg" % (route_id, time_key))
            plt.clf()
            plt.scatter(x, y, color="black", marker=".")
            #plt.gcf().autofmt_xdate()
            plt.draw()
            plt.savefig(path_figure)

            path_figure = os.path.join(abs_out_dir, "%s_%s_log.svg" % (route_id, time_key))
            plt.clf()
            plt.scatter(x, np.log(y), color="black", marker="o")
            #plt.gcf().autofmt_xdate()
            plt.draw()
            plt.savefig(path_figure)

if __name__ == "__main__":
    from kdd2017.utils import load_weather_info
    from kdd2017.utils import search_closest_date_weather_info
    from datetime import datetime
    weather_infos = ["/ssd/jinpeng/dataSets/dataSets/training/weather (table 7)_training_update.csv", "/ssd/jinpeng/dataSets/dataSets/testing_phase1/weather (table 7)_test1.csv"]

    datetime_weather = load_weather_info(weather_infos)

    cur_date = datetime(2016,10,18,12)
    weather = search_closest_date_weather_info(cur_date, datetime_weather)

