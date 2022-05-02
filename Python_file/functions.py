# -*- coding: utf-8 -*-
# ======================================================================================================================
# Copyright 2021 School of Environmental Science and Engineering, Shanghai Jiao Tong University, Shanghai, PR. China.

# This script is used to solve the optimal design solution of the rural sewage treatment system (contain the collection
# system and wastewater treatment system) within a considered area (small area, about 2000 households). This script used
# the A-star and Prim algorithm to calculate the optimal pipeline.

# This file is the function part of RuST-DM (Rural Sewage Treatment optimal Design Model).

# ----------------------------------------------------------------------------------------------------------------------
# GNU General Public License
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License # along with this program.
# If not, see <http://www.gnu.org/licenses/>

# ----------------------------------------------------------------------------------------------------------------------
# Author: Ysh Huang
# Contact: ysh_huang@foxmail.com
# Date: 22-10-2021
# Version: 1.1.1

# References: https://doi.org/10.1016/j.jenvman.2021.113673
# ======================================================================================================================

# general import
from __future__ import print_function  # only for python 2.7.x

import os
import math

import arcpy  # for ArcGIS
import numpy as np

from scipy.spatial import ConvexHull
from copy import deepcopy


class MyException(Exception):
    def __init__(self, message):
        Exception.__init__(self)
        self.message = message


# ----------------------------------------------------------------------------------------------------------------------
# Readout input data
# ----------------------------------------------------------------------------------------------------------------------

# Read built-in data


def read_standard(file_path):
    """
    用于读取当地《农村污水处理设施污水排放标准》。

    :param file_path: txt文件名及路径

    :return:
    discharge_standard: nested list [['id', scale_min,  scale_max, water_level, discharge_level], [], ...]
                        子列表中0为str，1-4为int
    """
    with open(file_path, 'r') as f:
        standard = [j.strip('\n').split(',') for j in f.readlines()[1:]]
    return [[i[0], int(i[1]), int(i[2]), int(i[3]), int(i[4])] for i in standard]


def read_technology_dataset(file_path):
    """
    用于读取农村污水处理工艺数据。

    :param file_path: txt文件名及路径

    :return:
    technology_dataset: nested list [['id', 'name', scale_min,  scale_max, water_level, discharge_level], [], ...]
                        子列表中0-1为str，2-5为int，6为float
    """
    with open(file_path, 'r') as f:
        technology = [j.strip('\n').split(',') for j in f.readlines()[1:]]
    return [[i[0], i[1], int(i[2]), int(i[3]), int(i[4]), int(i[5]), float(i[6])] for i in technology]


def read_parameters(file_path):
    """
    用于读取设计参数，如污水管网、覆盖率、预期造价等。泵站、检查井等价格。此函数用于读取

    :param file_path: txt文件名及路径

    :return:
    design_para: dictionary {pop：xx, seg: xx, slope:xx, dep_min:xx...}，key为str，value为float
    """
    with open(file_path, 'r') as f:
        pipe_type = [j.strip('\n').split(',') for j in f.readlines()[1:]]
    return dict([(i[0], float(i[1])) for i in pipe_type])


def read_pipe_type(file_path):
    """
    用于读取各规格管道的价格参数

    :param file_path: 参数数据存储路径及文件名称

    :return:
    type_dict：dictionary。{100: 30, ...)，其中key-value均为int
    """
    with open(file_path, 'r') as f:
        pipe_type = [j.strip('\n').split(',') for j in f.readlines()[1:]]
    return dict([(int(i[0]), int(i[1])) for i in pipe_type])


# Read input data


def read_household(point_feature, target_field, Q=0.085, P=None):
    """
    读取农村住房空间点模式数据。

    :param point_feature: point vector data
    :param target_field. str 聚类标识所在字段
    :param P: population per household, float, None as default, 使用默认值时表示采用属性表提供的P值
    :param Q: sewage discharge per day per capita, 0.085 m3 as default

    :return:
    household: nested list. [[id_h, X, Y, H, P * Q, M, 0, 0, FID, [None, None, 0, 0, 0, 0],
                             [None, None, None], [id_h, None, 0, 0, 0]], [], ...]
                             其中列表元素表示的意义分别为：id_h为ID号，X/Y/H分别为坐标和高程；p、Q为人数和人均污水排放量；M为聚类标识；
                             0为当地地表水功能等级，取值0-5，0表示无数据；0空间障碍，0为无，1为有；fid:GIS属性表FID号；
                             [None, None, 0, 0, 0, 0]管网参数，汇入节点、管材规格；埋深、跌水井、泵站、wtp，bool型(0/1)
                             [None, None, None]当地地表水功能等级/污水排放等级/满足条件的造价最低设备(工艺)
                             [id_h, None, 0, 0, 0]]当前节点ID/父节点ID/a*算法中的F、G、H值
                             [0, 0, 0, 0]设备造价/设备运维、管网造价/运维
    """
    rows, fields = arcpy.SearchCursor(point_feature), arcpy.ListFields(point_feature)
    spatial_ref = arcpy.Describe(point_feature).spatialReference
    household = []
    if P is None:  # 使用属性表中P值, 同过GIS数据指定P值（适用于小尺度）
        ID, X, Y, p, H, M, x, y, id_h, id_n, DID = 10000, 0, 0, 0, 0, 0, 0, None, None, None, None
        for row in rows:
            for field in fields:
                if field.name == "POINT_X":
                    X = row.getValue(field.name)
                if field.name == "POINT_Y" or field.name == "point_y":
                    Y = row.getValue(field.name)
                if field.name == "DID" or field.name == "did":  # 用于节点删除
                    DID = row.getValue(field.name)
                if field.name == "P" or field.name == "p":
                    p = row.getValue(field.name)
                if field.name == "dem" or field.name == "DEM":
                    H = row.getValue(field.name)
                if field.name == target_field:  # todo 不同情况下值不同，考虑改为用户输入？？！！
                    M = row.getValue(field.name)
                    id_h = "H%d" % ID  # 编号，起始点，H10000
            household.append([id_h, X, Y, H, p * Q, M, 0, 0, DID, [None, None, 0.5, 0, 0, 0],
                              [None, None, None], [id_h, None, 0, 0, 0], [0, 0, 0, 0]])
            ID += 1
    else:  # 使用函数接收的P值，大尺度，使用均值
        ID, X, Y, H, M, x, y, id_h, id_n, DID = 10000, 0, 0, 0, 0, 0, None, None, None, None
        for row in rows:
            for field in fields:
                if field.name == "POINT_X" or field.name == "point_x":
                    X = row.getValue(field.name)
                if field.name == "POINT_Y" or field.name == "point_y":
                    Y = row.getValue(field.name)
                if field.name == "DID" or field.name == "did":
                    DID = row.getValue(field.name)
                if field.name == "dem" or field.name == "DEM":
                    H = row.getValue(field.name)
                if field.name == target_field:  # 不同案例该值不同，考虑改为用户输入？？!!  todo
                    M = row.getValue(field.name)
                    id_h = "H%d" % ID  # 编号，起始点，H10000
            household.append([id_h, X, Y, H, P * Q, M, 0, 0, DID, [None, None, 0.5, 0, 0, 0],
                              [None, None, None], [id_h, None, 0, 0, 0], [0, 0, 0, 0]])
            ID += 1
    # check
    if len(household) == 0:
        raise Exception("EMPTY LIST: YOU GOT AN EMPTY LIST, PLEASE CHECK YOUR INPUT FILE!!!")
    return household, spatial_ref


def read_road(polyline_feature):
    """
    读取输入的路网数据，该数据在输入前需做好预处理工作。后续算法中对路段数据的使用、以及特性有所需求。
    管段长度尽可能保持相等，值控制在栅格大小的一半左右，如栅格为30米，则管段长度最好在15米左右。

    :param polyline_feature:

    :return:
    road_node: 路网中所有节点，nested list, 数据结构同household。 [[], [], ...]
    id_list: nested list。[[ID1,ID2], [ID2,ID3], ...]
    """
    # read out
    rows, fields = arcpy.SearchCursor(polyline_feature), arcpy.ListFields(polyline_feature)
    start_node_list, end_node_list, pair_list, id_list = [], [], [], []
    ID, X, Y, x, y, id_s, id_e = 10000, 0, 0, 0, 0, None, None
    for row in rows:
        for field in fields:
            if field.name == "START_X":
                X = row.getValue(field.name)
            if field.name == "START_Y":
                Y = row.getValue(field.name)
            if field.name == "END_X":
                x = row.getValue(field.name)
            if field.name == "END_Y":
                y = row.getValue(field.name)
        start_node_list.append((1, X, Y))
        end_node_list.append((1, x, y))
        pair_list.append([(X, Y), (x, y)])

    # reconstruct pipeline pair nodes。 统一设定管段节点ID号，以便于路径查找
    road_nodes = list(set(start_node_list + end_node_list))
    road_nodes = [list(i) for i in road_nodes]
    for node in road_nodes:
        id_r = "R%d" % ID  # 编号，起始点，R10000
        node[0] = id_r
        ID += 1

    node_dict = dict(zip([tuple(i[1:]) for i in road_nodes], road_nodes))  # {(x,y): [id,xy], ...}
    for s, e in pair_list:
        id_list.append([node_dict[s][0], node_dict[e][0]])
    road_nodes = [list(i) + [None, 0, None, 0, 0, None, [None, None, 0.5, 0, 0, 0], [None, None, None],
                             [i[0], None, 0, 0, 0], [0, 0, 0, 0]] for i in road_nodes]  # i 包含id、x/y
    # check

    if len(road_nodes) == 0 or len(id_list) == 0:
        raise Exception('EMPTY LIST!!!')
    return road_nodes, id_list


def read_raster(point_feature):
    """
    This function is used to read DEM dataset.

    :param point_feature:

    :return:
    DEM_list: nested list, 数据结构同household。 [[], [], ...]
    ENVI_list: nested list,  [[id_l, X, Y, L], [], ...]
    """
    rows, fields = arcpy.SearchCursor(point_feature), arcpy.ListFields(point_feature)
    DEM_list, ENVI_list = [], []
    ID, X, Y, T, L, id_t, id_l = 100000, 0, 0, 0, 0, None, None
    for row in rows:
        for field in fields:
            if field.name == "POINT_X" or field.name == "point_x":
                X = row.getValue(field.name)
            if field.name == "POINT_Y" or field.name == "point_y":
                Y = row.getValue(field.name)
            if field.name == "DEM" or field.name == "dem":
                T = row.getValue(field.name)
            if field.name == "ENVI" or field.name == "envi":
                L = row.getValue(field.name)
                id_t = "D%d" % ID  # 编号，起始点，H10000
                id_l = "E%d" % ID  # 编号，起始点，H10000
        DEM_list.append([id_t, X, Y, T, 0, None, 0, 0, None, [None, None, 0.5, 0, 0, 0],
                         [None, None, None], [id_t, None, 0, 0, 0], [0, 0, 0, 0]])
        ENVI_list.append([id_l, X, Y, L])
        ID += 1
    # check
    if len(DEM_list) == 0 or len(ENVI_list) == 0:
        raise Exception("EMPTY LIST: YOU GOT AN EMPTY LIST, PLEASE CHECK YOUR INPUT FILE!!!")
    return DEM_list, ENVI_list


def read_connector(point_feature):
    """
    用于读取邻近的市政/场镇污水管网接口，数据结构中与其他不同的时，本函数中的Q表示允许汇入的最大污水量。

    :param point_feature:

    :return:
    connector: nested list, 数据结构同household。 [[], [], ...]

    """
    rows, fields = arcpy.SearchCursor(point_feature), arcpy.ListFields(point_feature)
    connector = []
    ID, X, Y, Q, id_c = 100, 0, 0, 0, None
    for row in rows:
        for field in fields:
            if field.name == "POINT_X" or field.name == "point_x":
                X = row.getValue(field.name)
            if field.name == "POINT_Y" or field.name == "point_y":
                Y = row.getValue(field.name)
            if field.name == "Q" or field.name == "q":
                Q = row.getValue(field.name)
                id_c = "C%d" % ID  # 编号，起始点，H10000
        connector.append([id_c, X, Y, None, Q, None, 0, 0, None, [None, None, 0.5, 0, 0, 0],
                          [None, None, None], [id_c, None, 0, 0, 0]])
        ID += 1
    # check
    if len(connector) == 0:
        raise Exception("EMPTY LIST: YOU GOT AN EMPTY LIST, PLEASE CHECK YOUR INPUT FILE!!!")
    return connector


# ----------------------------------------------------------------------------------------------------------------------
# Data pretreatment
# ----------------------------------------------------------------------------------------------------------------------


def check_folder(folder_path):
    """
    检查文件夹是否存在，如不存在则创建这个文件夹

    :param folder_path: folder path， str

    :return: None
    """
    if os.path.exists(folder_path) is False:  # 存在
        os.makedirs(folder_path)
    return


def list_to_dict(point_list, index_num=0):
    """
    convert list to dictionary, setting ID as key (index_num = 0).

    :param point_list: nested list
    :param index_num: index number in nested list   在子列表中的索引号

    :return:
    point_dict：字典中每个key对应的value为point_list中的子列表
    """
    return dict(zip([i[index_num] for i in point_list], point_list))


def unfold_list(inputs):
    """
    展开嵌套列表，只展开一级。如管段ID对

    :param inputs: nested list， e.g. [[ID1,ID2],[ID2,ID3], ... ]

    :return:
    out_list: [ID1,ID2, ...]
    """
    return list(set([i for j in inputs for i in j]))


def get_distance(source, sink):
    """
    This function is used to calculate the three-dimensional Euclidean distance of two points.

    :param source: start point, list. road/household/DEM nodes[id,x,y,z ...]
    :param sink: end point, list. road/household/DEM nodes

    :return:
    distance: 3D Euclidean distance of the given two points, float
    distance_2D: 2D Euclidean distance of he given two points, float
    """
    height = abs(source[3] - sink[3])
    distance_2D = np.hypot((source[1] - sink[1]), (source[2] - sink[2]))
    distance = np.hypot(height, distance_2D)
    return distance, distance_2D


def get_nearest_road(road_nodes, household_nodes):
    """
    用于计算每个household，最邻近的road节点。为确保后续算法的有效性，road数据在输入前需基于DEM点进行打断。

    :param road_nodes: 路网节点，原始路网数据中无H值，需根据DEM数据计算。
    :param household_nodes:

    :return:
    nearest_dict: dictionary, id {H123: R111, ...}
    """
    nearest_dict = {}
    for household in household_nodes:
        candidates = []
        for node in road_nodes:
            distance, _ = get_distance(household, node)
            candidates.append([distance, node[0]])
        nearest_dict[household[0]] = sorted(candidates)[0][1]
    return nearest_dict


def get_nearest_node(source, sewer_nodes):
    """
    用于计算source节点中至污水管网最近的一个节点

    :param source: list. detail information of source node
    :param sewer_nodes: nested list, detail information of each sewer node

    :return:
    nearest_node: detail information of a sewer node
    """
    candidates = []
    for node in sewer_nodes:
        distance, _ = get_distance(source, node)
        candidates.append([distance, node, source])
    return sorted(candidates)[0]


def get_nearest_distance(point_list):
    """
    用于计算目标节点限制距离，取点簇内最邻近距离的均值

    :param point_list:
    :return:
    """
    near_distance = []  # list
    for PNT in point_list:
        distance_list = []
        for pnt in point_list:
            if pnt[0] != PNT[0]:
                distance, _ = get_distance(PNT, pnt)
                distance_list.append(distance)
            else:
                continue
        near_distance.append(sorted(distance_list)[0])
    return np.max(near_distance)


def get_nearest_plus(point_list, checked_s, pnt_list, near_distance=999.0):
    """
    用于计算余下节点中，至目标点集最邻近的节点P集

    :param point_list: household point within a cluster，目标点集
    :param checked_s: list, ID of checked point, marked as source nodes
    :param pnt_list: 管网节点
    :param near_distance: 最邻近距离，用于限定最邻近节点。

    :return:
    nearest: 完整节点信息
    """
    candidates = []
    for pnt in pnt_list:
        candidates_list = []
        for PNT in point_list:
            if pnt[0] != PNT[0] and PNT[0] not in checked_s:
                distance, _ = get_distance(pnt, PNT)
                candidates_list.append([PNT, pnt, distance])  # pnt为污水管网中节点
            else:
                continue
        candidates.append(sorted(candidates_list, key=lambda x: x[2])[0])
    result = [i for i in candidates if i[2] <= near_distance]
    if result:
        return result
    else:
        return False


def get_household_cluster(household_point):
    """
    this function used to convert household point list into a dictionary according the cluster marker, and the marker
    'C0' indicates the scatted household points.

    :param household_point: nested list. out put of function read_household

    :return:
    household_dict: dictionary. {'C1': [pnt1, pnt2,...], 'C2': [...], ...} pnt的数据结构同household_list中的子列表
    """
    marker = list(set([i[5] for i in household_point]))
    cluster_list = []
    for m in marker:
        cluster = []
        for i in household_point:
            if i[5] == m:
                cluster.append(i[0])  # only ID number
            else:
                continue
        cluster_list.append(cluster)
    return dict(zip(marker, cluster_list))


def recursive_del(household_cluster, household_num, C_RuST=1, err=0.1):  # TODO
    """
    采用递归的方法逐个删除满足条件的键值对，直至满足预期条件。该函数需配合household_filter函数使用。

    :param household_cluster: nested list, 每个点簇/散点构成一个子列表，每个子列表代表这个点簇中点的信息,按子列表长度逆序排列
    :param household_num number of household
    :param C_RuST: 农污处理覆盖率,1 as default 。当C_RuST=1时不会调用该函数
    :param err: 允许的误差, 0.1 as default

    :return:
    household_cluster：删除部分子列表后的嵌套列表，数据结构与输入相同
    rest_num: 剩余household数目
    deleted: nested list.删除掉的节点
    """
    del_num = 0
    percent = float(household_num - del_num) / household_num  # 实际值
    target = (percent - C_RuST) / C_RuST  # target error
    deleted = []
    while target > 0:
        del_item = household_cluster[-1]
        del_num += len(del_item)
        # renew
        percent_new = float(household_num - del_num) / household_num
        target_new = (percent_new - C_RuST) / C_RuST
        if target_new > 0 and abs(target_new) <= err:  # 不足，继续删除
            household_cluster = household_cluster[:-1]
            deleted.extend(del_item)
            target = target_new
        elif target_new > 0 and abs(target_new) > err:  # 满足条件，跳出循环
            break
        elif target_new < 0 and abs(target_new) <= err:  # 删除过多，但还在误差范围内，跳出循环
            break
        elif target_new < 0 and abs(target_new) > err:  # 防止删除过多，超出误差范围。不删除，保留上轮迭代结构
            del_num -= len(del_item)
            break
        else:
            raise Exception("UNEXPECTED ERROR!!!")
    rest_num = household_num - del_num
    return household_cluster, rest_num, deleted


def household_filter(household_dict, marker, C_RuST=1, err=0.1):
    """
    This function is used to filter household point data according required RuST coverage.

    :param household_dict: clustered household point pattern。
    :param marker: marker of scattered points
    :param C_RuST: 农污处理覆盖率,1 as default
    :param err: 允许的误差, 0.1 as default

    :return:
    filtered_household_dict: 数据结构同输入的 household_dict，ONE scatter构建成一个key-value
    rest_num: int剩余户数
    rate: 实际污水处理覆盖率
    del_item: 删除掉的household节点。完整信息
    """
    # get household number
    household_num = 0
    for _, V in household_dict.items():
        household_num += len(V)

    if C_RuST == 1:
        return list(household_dict.values()), household_num, 1, None

    elif 0 < C_RuST < 1:
        household_cluster = []
        for k, v in household_dict.items():
            # 嵌套列表，一个子列表表示一个cluster/所有scatter
            if k == marker:
                household_cluster.extend([[i] for i in v])
            else:
                household_cluster.append(v)
        household_cluster = sorted(household_cluster, key=lambda x: len(x), reverse=True)  # 逆序，嵌套列表的长度递减
        household_cluster, rest_num, del_item = recursive_del(household_cluster, household_num, C_RuST, err)
        # construct dictionary
        scatter, cluster = [], []
        for pnt in household_cluster:
            if len(pnt) > 1:
                cluster.append(pnt)
            elif len(pnt) == 1:
                scatter.append(pnt[0])

        if scatter:  # not empty
            cluster.append(scatter)
        return cluster, rest_num, float(rest_num) / household_num, del_item

    else:
        raise Exception("C_RuST VALUE MUST BE RANGE FROM 0 TO 1 !!!")


def get_dictionary(household_dict, marker):
    """
    将指定marker（scatter）的对象，存储为独立的键值对

    :param household_dict: 聚类后的household点簇。dictionary
    :param marker: 聚类标识, str

    :return:
    household: 更新后的household_dict
    """
    household = {}
    for k, v in household_dict.items():
        if k != marker:
            household[k] = v
        else:  # k == marker
            cnt = 0
            for i in v:
                K = "%s_S%d" % (k, cnt)
                household[K] = [i]
                cnt += 1
    return household


def assign_value(point_list, value_list, value_index, index_num):
    """
    This function is used to renew elevation or environmental level of each point

    :param point_list: household/road/connector point list
    :param value_list: DEM/ENVI list
    :param value_index: value index
    :param index_num: index number，待更新字段所处索引位置

    :return:
    point_list: value assigned point list.
    """
    for pnt in point_list:
        candidates_list = []
        for value in value_list:
            distance_2d = np.hypot((pnt[1] - value[1]), (pnt[2] - value[2]))
            candidates_list.append([distance_2d, value])
        candidate = sorted(candidates_list)[0][1]  # 取最小值
        pnt[index_num] = candidate[value_index]  # 更新目标值
    return point_list


def get_convex_hull(point_list):
    """
    This function is used to obtain the convex hull vertex of given points.
    使用scipy.spatial库中的ConvexHull计算凸包节点，并获取各点的ID号，用于查找管网合并时的最佳汇入点。

    :param point_list: nested list. 数据结构同household

    :return:
    返回列表包含每个节点完整信息
    vertex_list: list. [[ID1, ...], [ID2, ...], ...]
    """
    point = [i[1:3] for i in point_list[:]]
    point_arr = np.array(point)
    hull = ConvexHull(point_arr)
    index_num, index_list = [], []
    for simplex in hull.simplices:
        index_num.extend(list(simplex))
    index_list = list(set(index_num))
    # check
    if point_list[index_list[0]][1:3] != point[index_list[0]]:
        print("To checking convex hull points!!!")
        coor_list = [point[i] for i in index_list]
        vertex_list = []
        for P in coor_list:
            for p in point_list:
                if p[1] == P[0] and p[2] == P[1]:
                    vertex_list.append(p)
        return vertex_list
    else:
        return [point_list[i] for i in index_list]


# ----------------------------------------------------------------------------------------------------------------------
# 扩展模块 （Expansion Module）
# ----------------------------------------------------------------------------------------------------------------------


def get_diameter(Q, type_dict):
    """
    This function is used to ge the diameter of pipeline. And converted to standard specifications. 本项目中使用管径规格如
    下：

    :param Q: Flow of sewage, float (m3).
    :param type_dict: dictionary. with standard specifications of pipe, (mm)

    :return:
    diameter: standard specifications of pipe       int
    """
    diameter = None
    for i in sorted(type_dict):  # 不包含’DN‘
        Q_max = 0.8 * 9 * 3600 * 3.14159 * (i ** 2) / 4000000  # m3/day, flow speed is 1 m/s
        if Q <= Q_max:
            diameter = i
            break
        else:
            continue
    if None == diameter:  # raise error, if the diameter in type_list do not meet the requirement of Q.
        raise Exception("Please input more larger pipe specification!!!")
    return diameter


def get_technology(facility, standard, technology_list, f_ls=30):
    """
    selecting the cheapest technology according to the scale of sewage treatment facility and local environmental demands
    facility要独立与其他类型节点，不能相互引用。

    :param facility: with ENVI data， 此外flow需明确/更新
    :param standard: 当地农村污水处理设施排放标准
    :param technology_list: nested list
    :param f_ls: 使用寿命 30年

    :return:
    facility: 更新工艺、以及造价。数据结构不变
    站点编号，技术编号，X/Y坐标，处理规模，排放标准，环境质量，【处理设施造价、处理设施运维、管网造价、管网运维】
    """
    # pretreatment
    flow = math.ceil(facility[4] * 1.2)  #
    facility[10][0] = facility[6]

    # update standard
    s_candidates = []
    for s in standard:
        if s[1] <= flow <= s[2] and s[3] <= facility[6]:  # 只选择与当地环境现状相同的，不选择高于当地环境现状的标准
            s_candidates.append(s)
        else:
            continue

    if len(s_candidates) == 0:
        facility[10][1] = 4  # get pollutants discharge standard  TODO
        print("default(4) discharge standard was set!!!")
    else:
        S = sorted(s_candidates, key=lambda x: x[4])[-1]
        facility[10][1] = S[4]  # get pollutants discharge standard

    # select technology
    candidates = []
    for t in technology_list:
        if t[2] <= facility[10][1] and t[3] <= flow <= t[4]:  # get
            candidates.append(t)
        else:
            continue
    if len(candidates) == 0:
        raise Exception("No suitable technology available!!!")
    else:  # taking the cheapest one
        candidates.sort(key=lambda x: x[5])
        candidate = candidates[0]
        facility[10][2] = candidate[0]
        capex = candidate[5] * flow
        opex = candidate[6] * candidate[4] * f_ls * 365

    facility[12] = [capex, opex, 0, 0]
    return facility


def get_earthwork(source, sink, diameter, depth_min=0.5, wide=0.5, depth_max=4.0, slope_d=0.005):  # 考虑输出节点 todo
    """
    This function is used to calculate the volume of earthwork. If the slope of the pipeline is greater than the default
    value a drop well is required. The flow direction is from source point to sink point.
    输入数据中的source和sink点是数据结构相同的各类point feature。该函数只要用于管网路径查找中造价的估算，各节点会被多次迭代。因此，计算过程中不对
    节点中的数据进行修改。

    :param wide: wide of trench, meter
    :param diameter: diameter of pipeline, mm
    :param source: start point of a pipeline;
           [id, X, Y, H, Q, M, 0, 0, None, [None, None, 0, 0, 0, 0], [None, None, None], [id, None, 0, 0, 0],[0,0,0,0]]
    :param sink: end point of a pipeline.
           [id, X, Y, H, Q, M, 0, 0, None, [None, None, 0, 0, 0, 0], [None, None, None], [id, None, 0, 0, 0],[0,0,0,0]]
    :param depth_min: the minimum depth of pipeline, the default value is 0.5 meter.
    :param depth_max: the maximum depth of pipeline, the default value is 4 meter.
    :param slope_d: the slope of pipeline and the default value is 0.005. 管道设计坡度

    :return:
    earthwork: the volume of earthwork, float type(m3).
    drop: 0 or 1. 1 means a drop well is required; 0 means no drop well.
    pump_station: 0 or 1. 1 means a  pump station is required; 0 means no  pump station.
    """
    earthwork, drop, pump_station, deep = None, 0, 0, depth_min
    w = diameter / 1000.0 + wide  # meter 沟槽宽度
    height = source[3] - sink[3]  # 高程差
    trench = sink[9][2]  # 终点当前埋深，如果该点有另外支路接入，需重点考虑。
    distance, distance_2d = get_distance(source, sink)
    depth = depth_min + distance_2d * slope_d - height  # 埋深，理论计算值。
    # calculate earthwork
    if height >= 0:  # 顺坡：①小于最小值，②范围内，③大于最大值
        if depth > depth_max:
            pump_station = 1
            earthwork = 2 * depth_min * distance_2d * w  # * 0.5  # 开挖和回填
            deep = depth_min
        elif depth <= depth_max:  # trench = 0.5-4
            if depth > trench:
                earthwork = 2 * 0.5 * (depth_min + depth) * distance_2d * w
                drop = 1
                deep = depth
            elif depth_min <= depth <= trench:
                earthwork = 2 * 0.5 * (depth_min + depth) * distance_2d * w
                drop = 1
                deep = trench
            elif depth < depth_min:
                earthwork = 2 * 0.5 * (depth_min + depth_min) * distance_2d * w
                drop = 1
                deep = depth_min
    else:  # height < 0:  逆坡：①范围内，②大于最大值
        if depth > depth_max:
            pump_station = 1
            earthwork = 2 * depth_min * distance_2d * w
            deep = depth_min
        elif depth <= depth_max:
            if depth > trench:
                drop = 1
                earthwork = 2 * 0.5 * (depth_min + depth) * distance_2d * w
                deep = depth
            elif depth_min <= depth <= trench:
                earthwork = 2 * 0.5 * (depth_min + depth) * distance_2d * w
                drop = 1
                deep = trench
    return earthwork, drop, pump_station, deep


def get_h_value(point, end, price=150, lifespan=30):
    """
    This function is used to estimate the cost from point x to the end point, use the Manhattan distance.
    A-star 启发式函数H值的计算，用于估计中间点至终点的代价。

    :param point: the coordinate of point x; eg [ID, x, y, z, q, ...].
    :param end: the coordinate of end point; eg [ID, x, y, z, q, ...].
    :param price: comprehensive cost of pipeline, default value is 150 CNY。不含检查井
    :param lifespan: life span of pipeline, default value is 30 year.

    :return:
    h_value: 中间点x至终点（目标点）的估计代价。
    """
    distance = abs(point[1] - end[1]) + abs(point[2] - end[2]) + abs(point[3] - end[3])  # Manhattan Distance
    con = distance * price + (1 + int(distance / 50)) * 3000
    h_value = con + 0.012 * con * lifespan
    return h_value


def get_neighbor(point, graph, coefficient=1.5, cellsize=30):
    """
    FOR A* ALGORITHMS. This function is used to get the neighbor points within the GRAPH of a given point P.
    用于获取底图中与目标点（当前点）相邻的各点（包括斜对角点）

    :param point: the coordinate of point x.
    :param graph: point list, points with coordinate，graph实际就是DEM point feature数据读取后的point list。
    :param cellsize: raster size, 30 meter is set as default value
    :param coefficient: used to calculate the scanning distance of the given point P, default value is 1.5.

    :return:
    neighbor_dict: dictionary with neighbor points of point P.
    """
    neighbor_dict = {}
    near_distance = cellsize * coefficient
    for i in graph:
        distance = np.hypot(point[1] - i[1], point[2] - i[2])
        if distance <= near_distance and distance != 0:
            neighbor_dict[i[0]] = i

    if len(neighbor_dict) == 0:
        raise ValueError("You got AN EMPTY neighbor list! Please check you inputs!!!")
    return neighbor_dict


def pipeline_cost(earthwork, pipeline, diameter, drop, pipe_type, pump, price_dict, lifespan=30):
    """
    This function is used to calculate the total cost of pipeline, contains construction cost( material and earthwork),
    operation and maintenance cost. 两节点间的管道造价，配合get_earthwork使用。
    不包括化粪池、隔油井的相关费用。

    :param lifespan: life span of pipeline, default value is 30 year.
    :param pump: number of pump station, int
    :param earthwork: excavation and back-filling, m3
    :param pipeline: 3D Euclidean length of pipeline, meter.
    :param diameter: pipeline diameter, mm.
    :param drop: 0 or 1, 1 means a drop well is required; 0 means no drop well.
    :param pipe_type: price of pipe. type is set as key, eg. {100: 34}
    :param price_dict: price of earthwork, pipe, drop well, inspection well, pump,eg[12,{type: 59, ...}, 32, 44, 2344]

    :return:
    父节点至当前节点的造价，不包括起始节点至父节点的造价
    """
    capex = earthwork * price_dict['earthwork'] + pipeline * pipe_type[diameter] + drop * price_dict["drop"] + pump * \
            price_dict['pump']
    opex = 0.012 * capex * lifespan
    return round(capex, 4), round(opex, 4)


def get_weight(start, point, end, flow, price_dict, pipe_type, depth_min=0.5, wide=0.5, depth_max=4,
               slope=0.005, price=150, ls=30):
    """
    This function is used to calculate or upgrade F, G and H value. 计算中间点的启发式函数权重。这个中间点为当前处理点的相邻点。

    :param price_dict: price of earthwork, pipe, drop well, inspection well, pump. dict
    :param flow: 起始节点的污水流量,start node
    :param pipe_type
    :param depth_min: the minimum depth of pipeline, the default value is 0.5 meter.
    :param depth_max: the maximum depth of pipeline, the default value is 4 meter.
    :param slope: the slope of pipeline and the default value is 0.005. 管道设计坡度
    :param wide: wide of trench, meter
    :param start: the front point of current, list. parent node
    :param point: point need to calculate F, G and H.
    :param end: next point of current, list.
    :param price: comprehensive cost of pipeline
    :param ls: ife span of pipeline, default value is 30 year.

    :return:
    point: updated (input point)
    """
    pipeline, _ = get_distance(start, point)
    diameter = get_diameter(flow, pipe_type)
    earthwork, drop, pump, _ = get_earthwork(start, point, diameter, depth_min, wide, depth_max, slope)
    results = pipeline_cost(earthwork, pipeline, diameter, drop, pipe_type, pump, price_dict, ls)
    g_value = sum(results)
    H_value = get_h_value(point, end, price, ls)
    point[11][2], point[11][3], point[11][4] = g_value + H_value, g_value, H_value  # 只包含父节点至当前点的造价
    point[11][0], point[11][1] = point[0], start[0]
    return point


def get_path(end, path_dic):
    """
    This function is used to get the path from start point to end point.
    返回值记录起点至终点最优路径所经过的每一个节点的ID号，并按经过的先后逆序排列。配合A*算法使用

    :param end: ID of end point.
    :param path_dic: point ID dictionary, {ID: [ID, parent],...}。路径字典，start节点无父节点（None）

    :return:
    path: pipeline path from source point to sink point. [start, id1, id2, ... source]
    """
    path, check = [end], end
    while check:  # or check is not None:
        for v in path_dic.values():
            if check == v[0]:
                if v[1] is None:
                    check = v[1]
                else:
                    path.append(v[1])
                    check = v[1]
            else:
                continue

    path.reverse()  # reverse, made path from source node to sink node
    return path


def a_star(source, sink, graph, price_dict, pipe_type, flow, coeff=1.5, depth_min=0.5, wide=0.5, depth_max=4,
           slope=0.005, cellsize=30, lifespan=30):  # todo
    """
    This function is used to calculated the path from start point to end point based on A* algorithm. The weight used to
    find the optimal path (most cost-effective) is the total cost of pipeline, rather the distance.

    Do not consider the loss of flow.

    :param source: point coordinate of start point.
             [id, X, Y, H, Q, M, 0, 0, None, [None, None, 0, 0, 0, 0], [None, None, None], [id, None, 0, 0, 0],[0,0,0,0]]
    :param sink: point coordinate of end point, [ID, x, y, z, q...].
    :param graph: point list，nested list
           [[id, X, Y, H, Q, M, 0, 0, None, [None, None, 0, 0, 0, 0], [None, None, None], [id, None, 0, 0, 0],[0,0,0,0]], [], ...]
    :param price_dict: dictionary price of earthwork, pipe, drop well, inspection well, pump
    :param pipe_type dictionary  管道造价，按规格分
    :param flow: 污水流量（source node）
    :param coeff: used to calculate the scanning distance of the given point P, default value is 1.5.
    :param depth_min: the minimum depth of pipeline, default value is 0.5 meter.
    :param wide: wide of trench, meter.
    :param depth_max: the maximum depth of pipeline, default value is 4 meter.
    :param slope: the slope of pipeline, default value is 0.005.
    :param cellsize: raster size, 30 meter is set as default value
    :param lifespan: ife span of pipeline, default value is 30 year.

    :return:
    path: [(start, end), cost, [ids, id1, id2, ..., idn, ide]]
          path中的cost不包括接户管、隔油池和化粪池的造价
    """
    # data pretreatment
    graph_node, source, sink = deepcopy(graph), deepcopy(source), deepcopy(sink)  # 引用传递，导致数据污染
    graph_node.append(source)
    graph_node.append(sink)

    H = get_h_value(source, sink)
    path, cost, close_dict = [], 0, {}
    open_list = [[source[0], None, H, 0, H]]  # POINT, PARENT, F, G, H
    graph_dict = list_to_dict(graph_node)
    # get neighbor points
    while open_list:
        open_list.sort()  # 确保在原位修改而非产生一个新对象（sorted())
        parent = open_list.pop(0)  # [POINT, PARENT, F, G, H]
        # 到达终点
        if parent[0] == sink[0]:
            close_dict[parent[0]] = parent
            cost = parent[2]
            break
        point = graph_dict[parent[0]]  # parent point, details
        close_dict[parent[0]] = parent
        neighbors = get_neighbor(point, graph_node, coeff, cellsize)  # details
        for k, P in neighbors.items():  # get F, H, G (当前点、父节点、终点，而非当前点、起点、终点）
            if k in close_dict:
                continue
            else:
                pnt = get_weight(point, P, sink, flow, price_dict, pipe_type, depth_min, wide, depth_max, slope, 150,
                                 lifespan)  # 更新管道造价（当前点P至父节点）
                pnt_G = pnt[11][3] + parent[3]  # 更新当前点G值
                pnt_F = pnt[11][2] + parent[3]  # 更新当前点F值
                if k in [j[0] for j in open_list]:  # 新节点的邻近节点在open_list中
                    open_dict = list_to_dict(open_list, 0)
                    # 核定是否更新节点 G/H/F值
                    if pnt_F < open_dict[k][2]:  # 当前路径造价更低，更新G/H/F值, 基于Python的应用功能实现open_list的更新
                        open_dict[k][1], open_dict[k][3] = pnt[11][1], pnt[11][3]  # shadow copy
                        open_dict[k][2], open_dict[k][2] = pnt_F, pnt_G
                    else:  # no need to update
                        continue
                else:
                    pnt[11][3] = pnt_G
                    pnt[11][2] = pnt_F
                    open_list.append(pnt[11][:])
    # get path
    point_dict = {}
    for k, v in close_dict.items():
        point_dict[k] = v[:3]
    path_list = get_path(sink[0], point_dict)

    pump_list, drop_list = [], []
    for node_k, node in graph_dict.items():
        if node_k in path_list:  # 是pipeline节点
            if node[9][3] == 1 and node[9][4] == 1:
                pump_list.append(node_k)
                drop_list.append(node_k)
            elif node[9][3] == 1 and node[9][4] != 1:
                drop_list.append(node_k)
            elif node[9][3] != 1 and node[9][4] == 1:
                pump_list.append(node_k)
            else:
                continue
    path = [(source[0], sink[0]), cost, path_list]
    return path, drop_list, pump_list


def connect_test(facility, from_node, to_node, standard, technology, price, pipe_type,
                 depth_min=0.5, depth_max=4.0, wide=0.5, slope=0.004, f_sl=30, s_fl=30):
    """
    用于计算待测试节点是否可连接至当前WTP。当用函数时，需确保facility, node已进行深度复制

    :param facility: 当前WTP，完整信息household节点，节点DEM，ENVI，flow数据均已更新，且深度复制
    :param from_node: 待测试节点，完整信息household节点
    :param to_node: sewer node, 完整信息household节点
    :param standard:
    :param technology:
    :param price:
    :param pipe_type:
    :param depth_min:
    :param depth_max:
    :param wide:
    :param slope:
    :param f_sl:
    :param s_fl:

    :return:
    Bool: True means merging option was selected, False means onsite option was selected
    facility: facility for next iteration
    """
    # scenario: on-site
    facility_t = get_technology(deepcopy(from_node), standard, technology, f_sl)
    cost_onsite = sum(facility_t[12]) + sum(facility[12])

    # scenario: merge[to wtp]
    facility_m = deepcopy(facility)
    facility_m[4] += from_node[4]  # update flow
    facility_m = get_technology(facility_m, standard, technology, f_sl)
    diameter = get_diameter(from_node[4], pipe_type)
    workload = get_earthwork(from_node, to_node, diameter, depth_min, wide, depth_max, slope)
    length, _ = get_distance(facility, from_node)
    capex = workload[0] * price["earthwork"] + workload[1] * price["drop"] + workload[2] * price["pump"] + length * \
            pipe_type[diameter]
    opex = capex * 0.012 * s_fl
    facility_m[12][2] += capex
    facility_m[12][3] += opex
    cost_merge = sum(facility_m[12])

    # scenario option and output
    if cost_onsite >= cost_merge:  # merging
        return True, facility_m
    elif cost_onsite < cost_merge:
        return False, facility_t


def Expansion_Module_cluster(house_dict, household_dict, standard, technology, price, pipe_type, POP, Q,
                             depth_min=0.5, depth_max=4.0, wide=0.5, slope=0.004, f_sl=30, s_fl=30):
    """
    用于计算农村污水处理设施的最佳处理模式：区域设备规模的配置。此处仅进行初略计算，明确每套设备的规模（汇入的排污点），不对管网的铺设方案进行计算。
    因此，该结果只是理想情景下的最优解。后续还需对污水管网的铺设方案进行修正，如沿道路铺设。
    函数所使用的实参需进行深度复制

    :param house_dict: dictionary。过滤后的household节点为value，ID为key。节点DEM，ENVI数据均已更新
    :param household_dict: dictionary。household cluster dictionary每个点簇为一键值对，scatters
    :param standard: nested list。目标区域农村污水处理设施污水排放标准
    :param technology: nested list。常用的农村污水处理装备，
    :param price: dictionary。化粪池、开挖方等价格
    :param pipe_type: dictionary。各类规格管材造价
    :param POP:
    :param Q:
    :param depth_min: float.管段最小埋深，默认值为0.5米
    :param depth_max: float。管段最大埋深，默认值为4米
    :param wide: int。开挖沟槽宽度，默认值为0.5米。
    :param slope: float。管段坡度，默认值为0.004
    :param f_sl: int。污水处理设施设计寿命，默认值为30年。
    :param s_fl: int。管网设计寿命，默认值为30年

    :return:
    facility_dict：dictionary。key为处理设施ID号；value为其详细信息（list），结构同household节点。数据需基于原始数据copy
    source_dict: dictionary。污水处理设施的污水源。key为facility之ID号，value为nested list。{id：[[id,x,...], [], ...], ...}
    """
    house_dict = deepcopy(house_dict)
    facility_dict, source_dict = {}, {}
    # 迭代cluster/scatter
    for f_id, pnt in household_dict.items():
        # scenarios
        PNT = [house_dict[i] for i in pnt]
        if len(pnt) == 1:  # scatter
            facility = get_technology(deepcopy(PNT[0]), standard, technology, f_sl)
            # for write out
            facility_dict[facility[0]] = facility
            source_dict[facility[0]] = PNT

        elif len(pnt) == 2:  # only two household
            node_l, node_h = sorted(PNT, key=lambda x: x[3])
            facility = get_technology(deepcopy(node_l), standard, technology, f_sl)
            switch, facility_t = connect_test(facility, node_h, node_l, standard, technology, price, pipe_type,
                                              depth_min, depth_max, wide, slope, f_sl, s_fl)
            if switch:  # True, merging
                facility_dict[facility_t[0]] = facility_t
                source_dict[facility_t[0]] = PNT
            else:  # False  onsite
                # first node
                facility_dict[facility[0]] = facility
                source_dict[facility[0]] = [node_l]
                # second node
                facility_dict[facility_t[0]] = facility_t
                source_dict[facility_t[0]] = [node_h]

        elif len(pnt) >= 3:  # 计算每个source
            # set initial value
            node = sorted(PNT, key=lambda x: x[3])[0]  # Python 2.7 不能拆包
            near_distance = math.ceil(get_nearest_distance(PNT) / 10) * 10  # 向上取整，10
            facility = get_technology(deepcopy(node), standard, technology, f_sl)
            checked_nodes, source_ID = [node[0]], [node[0]]  # only ID
            source_nodes = [house_dict[i] for i in source_ID]  # sewage source nodes of facility
            test = get_nearest_plus(PNT, checked_nodes, source_nodes, near_distance)
            while len(checked_nodes) < len(pnt):
                # get the nearest node to current sewer network (source node)
                checked, m_nodes = [], []  # only ID
                if test:  # Non False
                    for item in test:
                        test_node = item[0]
                        switch, facility_t = connect_test(facility, item[0], item[1], standard, technology, price,
                                                          pipe_type, depth_min, depth_max, wide, slope, f_sl, s_fl)
                        if switch:  # True, merging
                            if test_node[0] in checked_nodes:
                                checked.append(test_node[0])
                            else:
                                checked.append(test_node[0])
                                checked_nodes.append(test_node[0])
                                source_ID.append(test_node[0])
                                m_nodes.append(test_node[0])
                        else:  # False 不合并
                            checked.append(test_node[0])

                        if len(test) == len(checked):  # 潜在合并点迭代完成  # do while for-loop complete
                            source_nodes = [house_dict[i] for i in source_ID]
                            wtp = deepcopy(sorted(source_nodes, key=lambda x: x[3])[0])
                            wtp[4] = len(source_nodes) * POP * Q
                            facility = get_technology(wtp, standard, technology, f_sl)
                            # all nodes iteration completed, output results
                            if len(checked_nodes) == len(pnt):  # iteration completed
                                facility_dict[facility[0]] = facility
                                source_dict[facility_t[0]] = source_nodes
                            else:  # 区分test中节点是否都不能合并
                                # for next iteration
                                if m_nodes:  # not empty, 继续使用当前facility进行合并测试
                                    test = get_nearest_plus(PNT, checked_nodes, source_nodes, near_distance)
                                else:  # empty, 当前facility没有可合并对象。输出当前wtp，并更换新wtp
                                    # output
                                    facility_dict[facility[0]] = facility
                                    source_dict[facility[0]] = [house_dict[i] for i in source_ID]
                                    # next iteration
                                    node = sorted([I for I in PNT if I[0] not in checked_nodes], key=lambda x: x[3])[0]
                                    facility = get_technology(deepcopy(node), standard, technology, f_sl)
                                    checked_nodes.append(node[0])
                                    source_ID = [node[0]]  # only ID
                                    if len(checked_nodes) == len(pnt):  # completed, output
                                        facility_dict[facility[0]] = facility
                                        source_dict[facility[0]] = [house_dict[i] for i in source_ID]
                                    else:  # uncompleted, for next iteration
                                        sewer_nodes = [house_dict[i] for i in source_ID]
                                        test = get_nearest_plus(PNT, checked_nodes, sewer_nodes, near_distance)
                        else:
                            continue

                else:  # false
                    # output current facility
                    facility_dict[facility[0]] = facility
                    source_dict[facility[0]] = [house_dict[i] for i in source_ID]
                    # for next iteration
                    node = sorted([I for I in PNT if I[0] not in checked_nodes], key=lambda x: x[3])[0]
                    facility = get_technology(deepcopy(node), standard, technology, f_sl)
                    checked_nodes.append(node[0])
                    source_ID = [node[0]]  # only ID
                    if len(checked_nodes) == len(pnt):  # completed, output
                        facility_dict[facility[0]] = facility
                        source_dict[facility[0]] = [house_dict[i] for i in source_ID]
                    else:  # uncompleted, for next iteration
                        sewer_nodes = [house_dict[i] for i in source_ID]
                        test = get_nearest_plus(PNT, checked_nodes, sewer_nodes, near_distance)

    return facility_dict, source_dict


def get_index(index):
    """
    获取列表中另一个元素所对应的index，用于两个元素的列表[a, b]

    :param index: 当前元素的index号，0或1

    :return:
    0或1
    """
    if index == 0:
        return 1
    if index == 1:
        return 0


def revise_wtp_id(wtp_new, sewer_dict, pipe_nodes, source_dict):
    """
    随着不断根据更新后的管网数据更新，wtp位点也随着改变。此时

    :param wtp_new: dictionary, 更新后的WTP，主要是坐标（节点）发生变化
    :param sewer_dict: dictionary,
    :param pipe_nodes: dictionary,
    :param source_dict: dictionary,

    :return:
    new_wtp:
    new_sewer:
    new_pipe_nodes:
    new_source:
    """
    new_wtp, new_sewer, new_pipe_nodes, new_source = {}, {}, {}, {}
    new_key_v = {K: V[0] for K, V in wtp_new.items()}  # key: old id, value: new id
    for ke, va in new_key_v.items():
        new_wtp[va] = wtp_new[ke]
        new_sewer[va] = sewer_dict[ke]
        new_pipe_nodes[va] = pipe_nodes[ke]
        new_source[va] = source_dict[ke]
    return new_wtp, new_sewer, new_pipe_nodes, new_source


def built_sewer_network_direct(source_nodes, facility_ID):
    """
    用于求解管网路径，该函数仅适用于3-4个节点。此时管网不沿道路铺设，按节点间的直线距离进行计算

    :param source_nodes: nested list。当前WTP的污水源节点，数据结构同household
    :param facility_ID: str。当前WTP的ID号

    :return:
    sewer：nested list。[[id1, id2], [id2, id3], ...]
    """
    sewer = []  # for output
    nodes_dict = list_to_dict(source_nodes)
    sink_list, checked = [nodes_dict[facility_ID]], [facility_ID]
    while len(checked) < len(source_nodes):
        candidates = []
        to_check = [i for i in source_nodes if i[0] not in checked]
        for s in sink_list:
            candidates.append(get_nearest_node(s, to_check))
        _, nearest, sink = sorted(candidates)[0]  # distance node sink
        sewer.append([nearest[0], sink[0]])
        checked.append(nearest[0])
        sink_list.append(nodes_dict[nearest[0]])
    return sewer


def get_pipe_path(road_pair, start, end, sewer_nodes):  # expansion, todo
    """
    求解两点间的管网沿道路铺设的路径，只计算两点间的路径。路网数据需根据DEM点进行打断，并且路网中路段的长度不能过短。输入的管道节点对数据
    必须确保是相连的，中间没有断点，否则该算法无效（无法跳出while循环）。

    算法中假定，先找到的路径（迭代次数少）即为最短路径，因此，为确保结果的有效性road节点间的距离需尽可能的保持一致。

    :param road_pair: nested list [[id1,id2],...]
    :param start: id
    :param end: id
    :param sewer_nodes: 目标管网节点ID列表

    :return:
    path: nested list [(start, end), 0, [start, ....., end]]
          只输出路径节点（from start to end)，不输出管段。
    """
    node_dict = {}
    key = sorted(list(set([i[0] for i in road_pair] + [i[1] for i in road_pair])))  # 所有road节点ID号
    for k in key:
        node_id = []  # 存储与节点k相邻的所有节点
        for nodes in road_pair:  # [id1, id2], 获取与节点k相邻的其他所有节点
            if k in nodes:
                index_num = get_index(nodes.index(k))  # 两个节点中的另一个
                node_id.append(nodes[index_num])
            else:
                continue
        node_dict[k] = node_id

    # get path
    path_nodes, checked = [], []
    next_nodes, check_list = [start], [start]
    while end not in next_nodes:  # BFS 宽度优先
        check = check_list.pop(0)  # 从尾部取值，出
        if check not in checked:  # and check in node_dict:
            next_nodes = node_dict[check]
            path_nodes.append([check, [i for i in node_dict[check] if i not in checked]])
            # 将新增节点ID添加至列表前端，以确保先进先出。
            check_list.extend(next_nodes)
            checked.append(check)
        elif check not in node_dict:
            raise KeyError
        else:
            continue

        # change end nodes
        for node in next_nodes:
            if node in sewer_nodes:
                end = node  # todo
            else:
                continue

    # for output
    parent = end
    check = []
    path_nodes_list = [parent]  # end to start, sequence
    while parent is not start:  # does this algorithm effective ?  todo
        for item in path_nodes:
            if parent in item[1] and item not in check:
                check.append(item)
                parent = item[0]
                path_nodes_list.append(parent)
            else:
                continue
    return [(start, end), 99999, path_nodes_list[::-1]]


def del_copy_pipe(branch_pipe, sewer):  # cost-effective ? todo
    """
    用于删除支管路径中与已有管网中重复的部分

    :param branch_pipe: 需检查的支管路径, nested list [[start, next1], [next1, id3],...],
    :param sewer: 已有管道路径, nested list [[start, next1], [next1, id3],...]

    :return:
    branch: 删除重复部分的支管
    interface: ID
    """
    branch, interface = [], None  # 留意重复元素
    sewer_nodes = unfold_list(sewer)
    if sewer:  # 非空
        for b in branch_pipe:
            if b in sewer or b[::-1] in sewer:  # 重复管段
                continue
            elif b[1] in sewer_nodes:
                branch.append(b)
                interface = b[1]
            elif b[0] in sewer_nodes:
                branch.append(b)
                interface = b[0]
            elif b[0] not in sewer_nodes and b[1] not in sewer_nodes:
                branch.append(b)
            else:  # non-copy
                raise Exception("ERROR: delete copy pipeline!!!")
        return branch, interface
    else:  # 空
        return branch_pipe, None


def check_nodes(path, near_nodes):
    """
    用于检测当前污水管网铺设路径中包含多少near nodes。

    :param path: nested list [[ID1,ID2], [ID2, ID3], ...]
    :param near_nodes: nested list。数据结构同household

    :return:
    num：int
    """
    path_nodes = unfold_list(path)
    num = 0
    for node in near_nodes:
        if node[0] in path_nodes:
            num += 1
    return num


def built_sewer_network_road(source_nodes, road_dict, road_pair, near_road, price, pipe_type,
                             coeff=3.0, depth_min=0.5, depth_max=4.0, wide=0.5, slope=0.004, s_ls=30):
    """
    用于求解有道路铺设时的污水管网沿道路铺设最佳方案（只给出管道的铺设方案）。并且，将高程最低的near road节点设置为新的WTP建设位置。
    当wtp所属污水源节点>=5，且有路网数据时，调用该函数。并且在本模型中我们假设所有的聚集规模超过4户的聚集区都有道路数据。
    在计算中，可能采取①直接铺设管网（直线）或②沿道路铺设管网两种情况，其中①仅起始两个节点；②中生成路劲除起始点外，还可能包含其他带连接的near node
    因此，计算二者按平均造价进行核算。

    将所有的source节点沿着管网连接起来。

    :param source_nodes:nested list，detail information on nodes. 数据结构同household。当前WTP的污水源节点
    :param road_dict: dictionary，数据结构同household。 目标区域路网节点
    :param road_pair: nested list，[[ID1,ID2], [ID2, ID3], ...]
    :param near_road: dictionary，{H1：R1，...}
    :param price:
    :param pipe_type:
    :param coeff:
    :param depth_min:
    :param depth_max:
    :param wide:
    :param slope:
    :param s_ls:

    :return:
    sewer: nested list. [[ID1,ID2], [ID2, ID3], ...]
    """
    # pretreatment
    sewer = []
    nodes_dict = list_to_dict(source_nodes)
    nodes_dict.update(road_dict)  # 所有节点
    near_nodes = []  # nearest road node of each source node. is copy???
    for i in [i[0] for i in source_nodes]:
        near = road_dict[near_road[i]]
        if near not in near_nodes:
            near_nodes.append(near)
        else:
            continue
    root = sorted(near_nodes, key=lambda x: x[3])[0]

    # pipeline, from household to road  接户管
    for source in source_nodes:  # TODO
        sewer.append([source[0], near_road[source[0]]])

    checked, sink_list, sewer_nodes = [root[0]], [root], {root[0]}
    while len(checked) < len(near_nodes):
        candidates, to_check = [], []
        for i in near_nodes:  # 待检测的节点road
            if i not in to_check and i[0] not in checked:
                to_check.append(i)

        # get path
        for pnt in sink_list:
            candidates.append(get_nearest_node(pnt, to_check))
        _, source, sink = sorted(candidates)[0]  # road nodes

        if source[0] in sewer_nodes:  # 已有管网经过该节点
            sink_list.append(source)
            checked.append(source[0])
        else:
            sewer_direct = [[source[0], sink[0]]]  # household
            _, _, path_nodes = get_pipe_path(road_pair, source[0], sink[0], sewer_nodes)  # todo
            sewer_road_o = [[path_nodes[i], path_nodes[i + 1]] for i in range(len(path_nodes) - 1)]
            num = check_nodes(sewer_road_o, near_nodes)
            sewer_road, _ = del_copy_pipe(sewer_road_o, sewer)  # 去除重复管段
            # get path cost
            diameter = get_diameter(source_nodes[0][4], pipe_type)
            cost_d, cost_r = 0, 0
            for d in sewer_direct:  # cost of sewer direct
                work_d = get_earthwork(nodes_dict[d[0]], nodes_dict[d[1]], diameter, depth_min, wide, depth_max, slope)
                length_d, _ = get_distance(nodes_dict[d[0]], nodes_dict[d[1]])
                result_pl_d = pipeline_cost(work_d[0], length_d, diameter, work_d[1], pipe_type, work_d[2], price, s_ls)
                cost_d += sum(result_pl_d)

            if sewer_road:  # not empty
                for r in sewer_road:  # cost of sewer-road
                    work_r = get_earthwork(nodes_dict[r[0]], nodes_dict[r[1]], diameter, depth_min, wide, depth_max,
                                           slope)
                    length_r, _ = get_distance(nodes_dict[r[0]], nodes_dict[r[1]])
                    result_pl_r = pipeline_cost(work_r[0], length_r, diameter, work_r[1], pipe_type, work_r[2], price,
                                                s_ls)
                    cost_r += sum(result_pl_r)
            else:  # empty
                cost_r = 0

            if (cost_r / num) / (cost_d / 2) > coeff:  # direct is select
                sewer.extend(sewer_direct)
                # for next iteration
                sink_list.append(source)
                checked.append(source[0])
                for s in sewer_direct:
                    sewer_nodes.update(s)
            else:
                sewer.extend(sewer_road)
                # for next iteration
                sink_list.append(source)
                checked.append(source[0])
                for s in sewer_road:
                    sewer_nodes.update(s)
    return sewer


def rebuilt_facility(sewer, node_dict, standard, technology, Q, POP, f_ls):
    """
    根据重建的污水管网铺设路劲，求解WTP的建设位点。为了尽可能的通过重力输送污水，算法中认为高程最低点为理想的WTP建设位点。

    :param sewer: nested list。 [id1, id2], [id2, id3],  ...]
    :param node_dict: dictionary. 目标区域路网节点\household\dem...，
    :param standard:
    :param technology:
    :param Q: 人均污水排放量 0.085
    :param POP: 户均人口， 3
    :param f_ls: 设计寿命，默认30年

    :return:
    facility:
    sewer_nodes: nested list. 数据结构同household
    """
    sewer_id = unfold_list(sewer)
    source = [i for i in sewer_id if i[0] == "H"]  # 污水源，household
    sewer_nodes = [node_dict[i] for i in sewer_id]
    wtp = deepcopy(sorted(sewer_nodes, key=lambda x: x[3])[0])  # 最终完善管网数据时会改动流量、埋深等参数
    wtp[4] = len(source) * Q * POP
    facility = get_technology(wtp, standard, technology, f_ls)
    return facility, sewer_nodes


def get_pipeline_parameters(flow, pipeline, node_dict, pipe_type, price, deep,
                            depth_min=0.5, wide=0.5, depth_max=4.0, slope=0.005, s_ls=30):
    """
    用于求解给定管段的管径、流量、埋深、pump、造价等

    :param flow: 当前管段污水流量
    :param pipeline: 管段ID列表[id_s, id_e]
    :param node_dict: 节点字典，包含目标区域内所有类型节点(sewer)。
    :param pipe_type:
    :param price:
    :param deep: 埋深
    :param depth_min:
    :param wide:
    :param depth_max:
    :param slope:
    :param s_ls:

    :return:
    source: 详细信息
    pipe：
    """
    source, sink = node_dict[pipeline[0]], node_dict[pipeline[1]]
    source[9][2] = deep
    diameter = get_diameter(flow, pipe_type)
    work = get_earthwork(source, sink, diameter, depth_min, wide, depth_max, slope)
    earthwork, drop, pump, deep = work
    length, _ = get_distance(source, sink)
    capex, opex = pipeline_cost(earthwork, length, diameter, drop, pipe_type, pump, price, s_ls)
    pipe = list(pipeline) + [deep, flow, drop, pump, capex, opex]
    return source, pipe


def get_sewer_network_cost(sewer, node_dict, root, price, pipe_type, depth_min=0.5, depth_max=4.0,
                           wide=0.5, slope=0.004, s_ls=30):  # todo
    """
    用于一个WTP的配套管网的计算，根据新确定的WTP位置，重构污水收集管网，明确各管网节点的埋深、流量；各管段的规格。

    :param sewer: nested list。[[id1, id2], [id2, id3], ...]
    :param node_dict: 节点字典，包含目标区域内所有类型节点(sewer对应节点）。
    :param root: wtp id
    :param price:
    :param pipe_type:
    :param depth_min:
    :param depth_max:
    :param wide:
    :param slope:
    :param s_ls:

    :return:
    sewer_network: nested list. [[id1, id2, type], [id2, id3, type], ...]
    sewer_nodes: nested list. 数据结构同household nodes list, 且各节点埋深已确定.
    capex: 建设投资造价
    opex: 运维造价
    """
    sewer = deepcopy(sewer)
    sewer_network, sewer_nodes = [], []
    copy_pipe_nodes = [i for j in sewer for i in j]
    pipe_nodes = unfold_list(sewer)

    # 设置初始值  获取叶节点(源节点)， id
    todo_nodes = []
    for i in pipe_nodes:  # id
        if copy_pipe_nodes.count(i) == 1 and i != root:
            todo_nodes.append(i)
        else:
            continue

    checked = []  # to store checked sewer pipeline
    pipe_dict = {}  # 用于存储计算后的管段信息 todo
    source_dict = dict(zip(todo_nodes, [node_dict[i] for i in todo_nodes]))  # flow需处理好 详细信息
    while sewer:  # 每轮检测后的管段均从sewer中删除，当sewer为空时结束迭代
        todo_line = []
        # 获取待处理管段，与叶节点直接相连的管段
        for pnt in todo_nodes:
            for pl in sewer:
                if pnt in pl:
                    index_num = get_index(pl.index(pnt))
                    todo_line.append((pnt, pl[index_num]))  # [source, sink]
                    checked.append(pl)
                else:
                    continue

        # 计算各新管段
        for line in todo_line:
            node = source_dict[line[0]]  # source node
            source, pipe = get_pipeline_parameters(node[4], line, node_dict, pipe_type, price, node[9][2], depth_min,
                                                   wide, depth_max, slope, s_ls)
            pipe_dict[tuple(pipe[:2])] = pipe
            sewer_nodes.append(source)
            sewer_network.append(pipe)

        # abort algorithm
        if root in [i[0] for i in sewer_nodes]:
            break

        # for next iteration
        # 删除已经处理后的管段
        for checked_item in checked:
            if checked_item in sewer:
                sewer.remove(checked_item)
            else:
                continue
        # 更新todo_node
        copy_pipe_nodes = [i for j in sewer for i in j]
        pipe_nodes = unfold_list(sewer)
        todo_nodes = []
        for i in pipe_nodes:  # id
            if copy_pipe_nodes.count(i) == 1 and i != root:
                todo_nodes.append(i)
            else:
                continue
        # 更新source_dict, 包括各点的流量、drop等
        source_dict = dict(zip(todo_nodes, [node_dict[i] for i in todo_nodes]))  # flow需处理好 详细信息
        for pnt in todo_nodes:
            to_node = []  # line
            for k in pipe_dict:
                if pnt == k[1]:
                    to_node.append(pipe_dict[k])
            # 更新flow、depth、drop、pump
            flow = sum([i[3] for i in to_node])
            deep = np.max([i[2] for i in to_node])
            drop = math.ceil(np.mean([i[4] for i in to_node]))
            pump = math.ceil(np.mean([i[5] for i in to_node]))
            source_dict[pnt][4] = flow
            source_dict[pnt][9][2] = deep
            source_dict[pnt][9][3] = drop
            source_dict[pnt][9][4] = pump

    # 输出
    capex = sum([i[6] for i in sewer_network])
    opex = sum([i[7] for i in sewer_network])
    for item in sewer_nodes:  # 更新管材规格
        item[9][1] = get_diameter(item[4], pipe_type)
    return sewer_network, sewer_nodes, capex, opex


def check_overlap(wtp_dict, sewer_dict, source_dict):
    """
    This function is used to check pipeline intersect/overlap. If the sewer network of two WTP is intersect/overlap
    these WTP would be merged to a bigger WTP. 将规模较大的一个WTP选为新WTP的建设位点。

    :param wtp_dict:
    :param sewer_dict:
    :param source_dict:

    :return:
    """
    facility, source = {}, {}
    sewer_nodes_dict = {}  # make a copy, to avoid data corruption
    for K, V in sewer_dict.items():
        sewer = [i[:2] for i in V]
        nodes = unfold_list(sewer)
        sewer_nodes_dict[K] = nodes

    neighbor = []  # 直接相连
    for k, n in sewer_nodes_dict.items():
        todo_list = [k]
        for K, N in sewer_nodes_dict.items():
            if K != k:  # self
                if len(set(n + N)) != len(n) + len(N):  # intersect/overlap
                    todo_list.append(K)
                else:  # 不相交、重叠
                    continue
        neighbor.append(todo_list)

    # merge
    for i in range(len(neighbor)):
        for j in range(len(neighbor)):
            x = list(set(neighbor[i] + neighbor[j]))
            y = len(neighbor[i]) + len(neighbor[i])
            if i != j and len(x) < y:
                neighbor[i] = x
                neighbor[j] = []

    for i in neighbor:
        if len(i) == 1:
            facility[i[0]] = wtp_dict[i[0]]
            source[i[0]] = source_dict[i[0]]
        elif len(i) == 0:
            continue
        elif len(i) > 1:
            wtp, nodes = [], []
            for j in i:
                nodes.extend(source_dict[j])
                wtp.append(wtp_dict[j])
            WTP = sorted(wtp, key=lambda X: X[4])[-1]
            facility[WTP[0]] = WTP
            source[WTP[0]] = nodes
    return facility, source


def tidy_results(facility_dict, source_dict, nodes_dict, road_dict, road_pair, near_road, standard, technology, price,
                 pipe_type, Q, POP, coeff=3.0, depth_min=0.5, depth_max=4.0, wide=0.5, slope=0.004, f_ls=30, s_ls=30):
    """
    根据Expansion_Module_cluster函数输出确定每个污水处理设施的管道铺设方案，管道沿道路铺设。本函数中，凡是处理规模大于等于4户的污水处理设施，其
    污水收集管网沿道路铺设。并且，将WTP的建设位置设置在nearest road nodes中的高程最低点附近。

    :param facility_dict：dictionary。key为处理设施ID号；value为其详细信息（list），结构同household节点。数据需基于原始数据copy
    :param source_dict: dictionary。污水处理设施的污水源节点。key为facility之ID号，value为nested list, 每一个子列表数据结构同household
                        节点，如 {id：[[id,x,...], [], ...], ...}
    :param nodes_dict: nested list，目标区域内的所有road、dem、household节点
    :param road_dict: dictionary。路网节点列表。数据结构同household节点
    :param near_road:
    :param road_pair:
    :param price:
    :param pipe_type:
    :param Q:
    :param POP:
    :param standard:
    :param technology:
    :param coeff: float，选择系数，用于确定两节点间管道是否沿道路铺设。默认值为3.0
    :param depth_min:
    :param depth_max:
    :param wide:
    :param slope:
    :param f_ls:
    :param s_ls:

    :return:
    WTP_dict: dictionary, 优化后的facility_dict
    pipeline_dict: dictionary, 管段id（起始点ID），管材规格。{WTP_id: [[ID1, ID2, deep, flow, drop, pump, capex, opex], [], ...], ...], ...}
    pipe_dict: dictionary，仅节点，数据结构同household，需包括埋深、管材规格
    """
    nodes_dict = deepcopy(nodes_dict)
    WTP_dict, pipeline_dict, pipe_dict = {}, {}, {}
    for wtp_id, wtp in facility_dict.items():
        if len(source_dict[wtp_id]) == 1:  # only one source node
            # write out
            WTP_dict[wtp_id] = wtp
            pipeline_dict[wtp_id] = []
            pipe_dict[wtp_id] = []

        elif len(source_dict[wtp_id]) == 2:  # two source nodes
            if wtp_id == source_dict[wtp_id][0][0]:
                sewer = [[source_dict[wtp_id][1][0], wtp_id, 0, 0, 0, 0, 0, 0]]
            else:
                sewer = [[source_dict[wtp_id][0][0], wtp_id, 0, 0, 0, 0, 0, 0]]
            so_nodes, si_node = nodes_dict[sewer[0][0]], nodes_dict[sewer[0][1]]
            length, _ = get_distance(so_nodes, si_node)
            diameter = get_diameter(so_nodes[4], pipe_type)
            work = get_earthwork(so_nodes, si_node, diameter, depth_min, wide, depth_max, slope)
            earthwork, drop, pump, deep = work
            capex, opex = pipeline_cost(earthwork, length, diameter, drop, pipe_type, pump, price, s_ls)
            sewer[0][2:] = deep, so_nodes[4], drop, pump, capex, opex
            wtp[12][2:] = capex, opex
            si_node[9][2] = deep
            WTP_dict[wtp_id] = wtp
            pipeline_dict[wtp_id] = sewer
            pipe_dict[wtp_id] = [so_nodes, si_node]

        elif 3 >= len(source_dict[wtp_id]) >= 3:  # 3-4 nodes
            sewer = built_sewer_network_direct(source_dict[wtp_id], wtp_id)
            WTP, sewer_node = rebuilt_facility(sewer, nodes_dict, standard, technology, Q, POP, f_ls)
            sewer_node_dict = list_to_dict(sewer_node, 0)
            sewer_results = get_sewer_network_cost(sewer, sewer_node_dict, WTP[0], price, pipe_type, depth_min,
                                                   depth_max, wide, slope)
            WTP[12][2:] = sewer_results[2:]
            WTP_dict[wtp_id] = WTP
            pipeline_dict[wtp_id] = sewer_results[0]
            pipe_dict[wtp_id] = sewer_results[1]

        elif len(source_dict[wtp_id]) >= 4:
            sewer = built_sewer_network_road(source_dict[wtp_id], road_dict, road_pair, near_road, price, pipe_type,
                                             coeff, depth_min, depth_max, wide, slope, s_ls)
            WTP, sewer_node = rebuilt_facility(sewer, nodes_dict, standard, technology, Q, POP, f_ls)
            sewer_node_dict = list_to_dict(sewer_node, 0)
            sewer_results = get_sewer_network_cost(sewer, sewer_node_dict, WTP[0], price, pipe_type, depth_min,
                                                   depth_max, wide, slope)
            WTP[12][2:] = sewer_results[2:]
            WTP_dict[wtp_id] = WTP
            pipeline_dict[wtp_id] = sewer_results[0]
            pipe_dict[wtp_id] = sewer_results[1]

    return WTP_dict, pipeline_dict, pipe_dict


# ----------------------------------------------------------------------------------------------------------------------
# 合并模块 （Merging Module） functions for merging_module
# ----------------------------------------------------------------------------------------------------------------------


def get_control_node(source_dict):
    """
    在合并测试时需求解当前WTP的最邻近wtp，为减少计算量，以每个wtp的污水源节点的坐标均值作为距离核算的标准点。

    :param source_dict: dictionary，以wtp的ID号为key，接入该wtp的household节点为构成的嵌套列表为value

    :return:
    control_node_dict: dictionary, {wtp_id: [x,y], ...}
    """
    control_node_dict = {}
    for k, v in source_dict.items():
        x = np.mean([i[1] for i in v])
        y = np.mean([i[2] for i in v])
        control_node_dict[k] = [k, x, y, 0]
    return control_node_dict


def get_candidate_WTPs(control_node_dict, WTP, checked, near_dis=200.0):
    """
    根据控制点，求解目标wtp的邻近WTP(不超过限制距离），并按距离远近升序排列。

    :param control_node_dict: dictionary，{wtp_id: [x,y], ...}
    :param WTP: 当前待检测WTP, [ID, X, Y, 0]
    :param checked: 已合并，或检测后无合并可能的wtp列表，仅ID号
    :param near_dis: 限制距离，默认200米。超过该值则表明不适合合并

    :return:
    candidate_WTP: list， [wtp1，wtp2, wtp3, ... wtp5],
    """
    candidates, candidate_WTP = [], []
    for k, f in control_node_dict.items():
        if k not in checked and WTP[0] != k:
            _, distance = get_distance(WTP, f)
            candidates.append([distance, k])
        else:
            continue

    # 输出
    candidates = sorted(candidates)
    for item in candidates:
        if item[0] <= near_dis:
            candidate_WTP.append(item[1])
        else:
            continue
    return candidate_WTP


def get_nearest_nodes_pair(source_nodes, candidate_WTP, source_dict):
    """
    用于计算当前WTP与潜在合并对象（wtp）的最邻近节点对，并按距离输出。  todo

    :param source_nodes:
    :param candidate_WTP:
    :param source_dict:

    :return:
    out_list: nested list. e.g. [[distance, wtp_id, pnt, PNT], ...] 按distance大小升序
    """
    candidates_pair = []
    for wtp_id in candidate_WTP:
        pair_list = []
        for pnt in source_dict[wtp_id]:
            for PNT in source_nodes:
                distance, _ = get_distance(pnt, PNT)
                pair_list.append([distance, wtp_id, pnt, PNT])
        candidates_pair.append(sorted(pair_list)[0])
    return sorted(candidates_pair)


def get_path_cost(sewer, flow, all_nodes, price_dict, pipe_type, depth_min=0.5,
                  depth_max=4, wide=0.5, slope=0.005, s_ls=30):
    """
    用于估算管网路径的造价，用于大致估算某一段管网的造价。

    :param sewer: nested list, 管网的管段ID
    :param flow: 流量
    :param all_nodes: dictionary
    :param price_dict: dictionary
    :param pipe_type: dictionary
    :param depth_min:
    :param depth_max
    :param wide:
    :param slope:
    :param s_ls:

    :return:
    cost: float，造价
    """
    cost = 0
    diameter = get_diameter(flow, pipe_type)
    for pipe in sewer:
        source, sink = all_nodes[pipe[1]], all_nodes[pipe[0]]
        length, _ = get_distance(source, sink)
        earthwork, drop, pump, deep = get_earthwork(source, sink, diameter, depth_min, wide, depth_max, slope)
        capex, opex = pipeline_cost(earthwork, length, diameter, drop, pipe_type, pump, price_dict, s_ls)
        cost += capex + opex
    return cost


def path_simplification(path, sewer_nodes):
    """
    污水接入已有管网目标节点时，新生成的管网可能与已有管网重叠，或形成环路。为避免此类情况发生，需对新生成路径进行检测，当路径中节点为已有管网节点时，
    该点之后的路径将被删除。

    :param path: list, [start, id1, id2, ..., end]
    :param sewer_nodes: id list. id of sewer nodes.

    :return:
    path_new: list, [start, id1, id2, ..., end_new]
    """
    path_new = []
    for node in path:
        if node not in sewer_nodes:
            path_new.append(node)
        else:
            path_new.append(node)
            break
    return path_new


def merging_test(facility_i, facility2_i, sewer1_i, sewer2_i, source_node1_i, source_node2_i, candidate, all_nodes,
                 dem_nodes, road_dict, near_road, road_pair, standard, technology, price_dict, pipe_type, Q, POP,
                 coeff=3.0, expected=20000,  s_coeff=1.5, depth_min=0.5, depth_max=4, wide=0.5, slope=0.005, cellsize=30,
                 f_ls=30, s_ls=30):
    """
    用于测试在预期的户均造价内，两个WTP是否合并。首先明确两个WTP按何种方式进行合并（①直接、②沿道路），再核算新的管网造价，如造价小于预期值则合并，
    反之则不合并。

    :param facility_i: 当前WTP， 详细信息
    :param facility2_i: 待测试WTP
    :param sewer1_i: nested list, wtp1的配套污水收集管网
    :param sewer2_i: nested list, wtp2的配套污水收集管网
    :param source_node1_i: nested list, wtp1的污水源节点
    :param source_node2_i: nested list, wtp2的污水源节点
    :param candidate
    :param all_nodes: dictionary, 所有节点
    :param dem_nodes: nested list，dem节点
    :param road_dict: dictionary，数据结构同household。 目标区域路网节点
    :param near_road: dictionary，household节点的最邻近road节点{H01：R01，...}
    :param road_pair:
    :param standard
    :param technology
    :param price_dict:
    :param pipe_type:
    :param Q
    :param POP
    :param coeff:
    :param expected: 用户预期的户均造价
    :param s_coeff: A*搜索系数，默认值为1.5
    :param depth_min:
    :param depth_max:
    :param wide:
    :param slope:
    :param cellsize:
    :param f_ls:
    :param s_ls:

    :return:
    bool: True/False，一旦为False，则表示不合并，余下参数全设置为None
    WTP: list, WTP详细信息
    sewer: nested list,
    source: nested list, 节点详细信息
    """
    # 不合并时的输出  # make a copy
    facility, sewer1, sewer2 = deepcopy(facility_i), deepcopy(sewer1_i), deepcopy(sewer2_i)
    source_node1, source_node2 = deepcopy(source_node1_i), deepcopy(source_node2_i)

    cost_on = sum(facility_i[12]) + sum(facility2_i[12])  # 不合并时的造价（总）

    all_nodes = deepcopy(all_nodes)
    sewer_nodes1 = unfold_list([[i[0], i[1]] for i in sewer1_i])  # 节点ID
    sewer_nodes2 = unfold_list([[i[0], i[1]] for i in sewer2_i])  # 节点ID

    # pretreatment
    sewer_wtp = [i[:2] for i in sewer1] + [i[:2] for i in sewer2]  # 管段起始节点ID
    flow1 = len(source_node2) * Q * POP
    flow2 = len(source_node1) * Q * POP
    household_num = len(source_node1) + len(source_node2)

    # get nearest nodes pair
    distance, wtp1_node, wtp2_node = candidate

    # sewer direct
    if distance > 5 * cellsize:  # do A*
        path1, _, _ = a_star(wtp2_node, wtp1_node, dem_nodes, price_dict, pipe_type, flow1, s_coeff, depth_min, wide,
                             depth_max, slope, cellsize, f_ls)
        path_1 = path_simplification(path1[2], sewer_nodes2)
        path_2 = path_simplification(path_1[::-1], sewer_nodes1)
        sewer_d = [list(i) for i in zip(path_2[:-1], path_2[1:])]

    else:  # straight line
        sewer_d = [[wtp2_node[0], wtp1_node[0]]]

    sewer_dir = sewer_wtp + sewer_d
    sewer_nodes_dir = unfold_list(sewer_dir)  # id
    sewer_node_dir = [all_nodes[i] for i in sewer_nodes_dir]
    facility_dir = deepcopy(sorted(sewer_node_dir, key=lambda x: x[3])[0])
    facility_dir[4] = flow1 + flow2
    WTP_dir = get_technology(facility_dir, standard, technology, s_ls)
    wtp_cost_dir = sum(WTP_dir[12])
    sewer_net_dir, _, capex, opex = get_sewer_network_cost(sewer_dir, all_nodes, WTP_dir[0], price_dict, pipe_type,
                                                           depth_min, depth_max, wide, slope, s_ls)
    cost_dir = (wtp_cost_dir + capex + opex) * coeff
    facility_dir[12][2:] = capex, opex

    # sewer road
    source_new = source_node1 + source_node2
    sewer_road = built_sewer_network_road(source_new, road_dict, road_pair, near_road, price_dict, pipe_type, coeff,
                                          depth_min, depth_max, wide, slope, s_ls)
    sewer_nodes_road = unfold_list(sewer_road)
    sewer_node_road = [all_nodes[i] for i in sewer_nodes_road]
    facility_road = deepcopy(sorted(sewer_node_road, key=lambda x: x[3])[0])
    facility_road[4] = flow1 + flow2
    WTP_road = get_technology(facility_road, standard, technology, s_ls)
    wtp_cost_road = sum(WTP_dir[12])
    sewer_net_road, _, capex, opex = get_sewer_network_cost(sewer_road, all_nodes, WTP_road[0], price_dict, pipe_type,
                                                            depth_min, depth_max, wide, slope, s_ls)
    cost_road = wtp_cost_road + capex + opex
    facility_road[12][2:] = capex, opex

    # 明确是否合并，并输出
    if expected * household_num <= min(cost_road, cost_dir, cost_on):  # 预期值过小
        # print("用户预期造价过低，按模型计算的最小值输出...")
        if min(cost_road, cost_dir, cost_on) == cost_on:
            return False, facility_i, sewer1_i, source_node1_i
        elif min(cost_road, cost_dir, cost_on) == cost_road:
            return True, WTP_road, sewer_net_road, source_new
        elif min(cost_road, cost_dir, cost_on) == cost_dir and cost_dir != cost_road:
            return True, WTP_dir, sewer_net_dir, source_new
        elif cost_on == cost_dir == cost_road:
            return True, WTP_road, sewer_net_road, source_new
        else:
            raise ValueError(".....")

    elif expected * household_num > min(cost_road, cost_dir, cost_on):
        if expected >= (cost_road / household_num):
            return True, WTP_road, sewer_net_road, source_new
        elif cost_dir <= expected * household_num < cost_road:
            return True, WTP_dir, sewer_net_dir, source_new
        elif cost_on <= expected * household_num < min(cost_road, cost_dir):
            return False, facility_i, sewer1_i, source_node1_i
        else:
            raise ValueError("*****")


def get_merging_test_result(facility, sewer, source, candidates, wtp_dict, sewer_dict, source_dict, all_nodes,
                            dem_nodes, road_dict, near_road, road_pair, standard, technology, price_dict, pipe_type,  Q,
                            POP, coeff, expected, s_coeff, depth_min, depth_max, wide, slope, cellsize, f_ls, s_ls):
    """
    配合merging module使用，用于检测潜在合并对象，经济上是否具备合并的可能。

    :param facility: 当前WTP
    :param sewer: 当前WTP 配套管网
    :param source: 当前WTP污水源
    :param candidates: 待合并测试的潜在对象
    :param wtp_dict:
    :param sewer_dict:
    :param source_dict:
    :param all_nodes:
    :param dem_nodes:
    :param road_dict:
    :param near_road:
    :param road_pair:
    :param standard:
    :param technology:
    :param price_dict:
    :param pipe_type:
    :param Q:
    :param POP:
    :param coeff:
    :param expected:
    :param s_coeff:
    :param depth_min:
    :param depth_max:
    :param wide:
    :param slope:
    :param cellsize:
    :param f_ls:
    :param s_ls:

    :return:
    wtp_id:
    test_results:
    """
    test_results_list = []
    test_results, wtp_id = None, None
    for item in candidates:
        wtp_id, candidate = item[1], [item[0], item[2], item[3]]
        facility_t, source_t, sewer_t = wtp_dict[wtp_id], source_dict[wtp_id], sewer_dict[wtp_id]
        test_results = merging_test(facility, facility_t, sewer, sewer_t, source, source_t, candidate, all_nodes,
                                    dem_nodes, road_dict, near_road, road_pair, standard, technology, price_dict,
                                    pipe_type, Q, POP, coeff, expected, s_coeff, depth_min, depth_max, wide, slope,
                                    cellsize, f_ls, s_ls)
        test_results_list.append([wtp_id, test_results])

    bool_list = list(set([i[1][0] for i in test_results_list]))
    if len(bool_list) == 1 and bool_list[0] is False:  # 测试结果均为 False。 不合并
        wtp_id = test_results_list[0][0]
        test_results = test_results_list[0][1]
    elif len(bool_list) == 1 and bool_list[0] is True:  # 测试结果均为 True。 合并
        wtp_id = test_results_list[0][0]
        test_results = test_results_list[0][1]
    elif len(bool_list) == 2:  # MERGING
        candidates = []
        for i in test_results_list:
            if i[1][0] is True:
                candidates.append(i)
        wtp_id = candidates[0][0]
        test_results = candidates[0][1]
    return wtp_id, test_results


def merging_module(wtp_dict, sewer_dict, source_dict, all_nodes, dem_nodes, near_road, road_dict, road_pair, standard,
                   technology, price_dict, pipe_type, Q, POP, coeff=3.0, expected=20000, s_coeff=1.5, depth_min=0.5,
                   depth_max=4, wide=0.5, slope=0.005, cellsize=30, f_ls=30, s_ls=30, near_coeff=10.0):
    """
    外壳函数，接收EM输出数据，进行合并测试，合并后户均造价.

    :param wtp_dict: dictionary
    :param sewer_dict: dictionary
    :param source_dict: dictionary
    :param all_nodes:
    :param dem_nodes:
    :param near_road:
    :param road_dict: dictionary，数据结构同household。 目标区域路网节点
    :param road_pair:
    :param standard:
    :param technology:
    :param price_dict:
    :param pipe_type:
    :param Q:
    :param POP:
    :param coeff:
    :param expected:
    :param s_coeff:
    :param depth_min:
    :param depth_max:
    :param wide:
    :param slope:
    :param cellsize:
    :param f_ls:
    :param s_ls:
    :param near_coeff: 用于控制待合并测试WTP与当前wtp的距离，默认值为1。取值越大距离越远、待测试的WTP越多

    :return:
    facility_dict: dictionary
    sewer_network: dictionary
    source_node: dictionary
    """
    facility_dict, sewer_network, source_node = {}, {}, {}  # outputs
    checked = []  # for iteration
    # source节点概化 -- pretreatment
    control = get_control_node(source_dict)  # 每个wtp一个控制点，取source节点坐标均值
    # get limit distance  # 取各节点最邻近距离的最大值
    near_distance = math.ceil(get_nearest_distance(control.values()))

    # 设置初始条件
    facility = sorted(wtp_dict.values(), key=lambda X: X[4], reverse=True)[0]
    checked.append(facility[0])
    source, sewer = source_dict[facility[0]], sewer_dict[facility[0]]  # current wtp
    # while循环
    while len(checked) < len(wtp_dict):
        print(". ", end="")
        # 获取在测试wtp列表
        x, y = np.mean([i[1] for i in source]), np.mean([i[2] for i in source])
        wtp_c, sewer_node = [facility[0], x, y, 0], []

        # 获取潜在待合并的WTP
        candidate_WTPs = get_candidate_WTPs(control, wtp_c, checked, near_distance * near_coeff)
        candidates = get_nearest_nodes_pair(source, candidate_WTPs, source_dict)
        if candidates:  # not empty
            # for 循环测试wtp  # todo
            wtp_id, test_results = get_merging_test_result(facility, sewer, source, candidates, wtp_dict, sewer_dict,
                                                           source_dict, all_nodes, dem_nodes, road_dict, near_road,
                                                           road_pair, standard, technology, price_dict, pipe_type,  Q,
                                                           POP, coeff, expected, s_coeff, depth_min, depth_max, wide,
                                                           slope, cellsize, f_ls, s_ls)
            if test_results[0]:  # None False 合并
                # for next iteration
                if wtp_id is None:
                    raise Exception("wtp_id == None")
                else:
                    checked.append(wtp_id)

                # for next iteration
                _, facility, sewer, source = test_results

                # for write out
                if len(set(checked)) == len(wtp_dict):  # 完成合并测试
                    facility_dict[facility[0]] = facility
                    sewer_network[facility[0]] = sewer
                    source_node[facility[0]] = source
            elif test_results[0] is False:  # False  不合并
                # write out
                _, facility, sewer, source = test_results
                facility_dict[facility[0]] = facility
                sewer_network[facility[0]] = sewer
                source_node[facility[0]] = source

                # for next iteration
                wtp_id, facility = sorted([[k, v] for k, v in wtp_dict.items() if k not in checked],
                                          key=lambda X: X[1][4], reverse=True)[0]
                checked.append(wtp_id)
                source, sewer = source_dict[wtp_id], sewer_dict[wtp_id]  # current wtp

                if len(checked) == len(wtp_dict):  # 完成测试，直接输出
                    facility_dict[facility[0]] = wtp_dict[facility[0]]
                    sewer_network[facility[0]] = sewer_dict[facility[0]]
                    source_node[facility[0]] = source_dict[facility[0]]
                else:
                    continue

            else:
                raise MyException("some wrong with function merging_test!")

        else:  # 当前WTP无满足条件的合并对象
            if facility[0] not in facility_dict:  # new facility
                # write out
                facility_dict[facility[0]] = facility
                sewer_network[facility[0]] = sewer
                source_node[facility[0]] = source

                # for next iteration
                wtp_id, facility = sorted([[k, v] for k, v in wtp_dict.items() if k not in checked],
                                          key=lambda X: X[1][4], reverse=True)[0]
                checked.append(wtp_id)
                source, sewer = source_dict[wtp_id], sewer_dict[wtp_id]  # current wtp

                if len(checked) == len(wtp_dict):  # 完成测试，直接输出
                    facility_dict[facility[0]] = wtp_dict[facility[0]]
                    sewer_network[facility[0]] = sewer_dict[facility[0]]
                    source_node[facility[0]] = source_dict[facility[0]]
                else:
                    continue

            else:
                # write out
                facility_dict[facility[0]] = wtp_dict[facility[0]]
                sewer_network[facility[0]] = sewer_dict[facility[0]]
                source_node[facility[0]] = source_dict[facility[0]]

                # for next iteration
                wtp_id, facility = sorted([[k, v] for k, v in wtp_dict.items() if k not in checked],
                                          key=lambda X: X[1][4], reverse=True)[0]
                checked.append(wtp_id)
                source, sewer = source_dict[wtp_id], sewer_dict[wtp_id]  # current wtp

                if len(checked) == len(wtp_dict):  # 完成测试，直接输出
                    facility_dict[facility[0]] = wtp_dict[facility[0]]
                    sewer_network[facility[0]] = sewer_dict[facility[0]]
                    source_node[facility[0]] = source_dict[facility[0]]
                else:
                    continue
    return facility_dict, sewer_network, source_node


def get_pipe_nodes_results(nodes_dict, sewer_network, pipe_type):
    """
    根据各WTP的管网数据获取关键管网节点信息。

    :param nodes_dict:
    :param sewer_network: EM或MM输出的管网数据，dictionary， wtp ID as key， pipelines as value
    :param pipe_type:

    :return:
    pipe_node_dict: dictionary,
    """
    all_nodes = deepcopy(nodes_dict)
    pipe_node_dict = {}
    for k, v in sewer_network.items():
        if v:  # NOT EMPTY
            nodes_list = []  # pipe nodes of WTPi
            for item in v:
                node = all_nodes[item[1]]
                node[4] = item[3]  # Q
                diameter = get_diameter(item[3], pipe_type)
                node[9][1:] = [diameter, item[2], item[4], item[5]]  # pipeline type, deep, drop, pump
                nodes_list.append(node)
            pipe_node_dict[k] = nodes_list
        else:  # empty
            pipe_node_dict[k] = v
    return pipe_node_dict


# ----------------------------------------------------------------------------------------------------------------------
# modelling outputs statistics
# ----------------------------------------------------------------------------------------------------------------------


def degree_of_dec(facility_dict, source_dict):
    """
    用于计算目标区域内农村污水治理的分散程度，该计算方法同时考虑了污水处理设施的数量和规模，以农村居民户数。

    :param facility_dict: 模型计算得出的污水处理设施数据，可以EM或MM模块的计算结果， dictionary
    :param source_dict:  模型计算得出的污水处理设施配套管网数据{id: [[id1, id2], [id2, id3], ...], ...}

    :return:
    dd: degree of decentralization, float
    """
    W_t = sum([i[4] for i in facility_dict.values()])
    H_t = sum([len(i) for i in source_dict.values()])
    hi_wi = 0
    for k, v in source_dict.items():
        hi_wi += len(v) * facility_dict[k][4]
    return hi_wi / (W_t * H_t)


# 管道管径及各节点的埋深？

# ----------------------------------------------------------------------------------------------------------------------
# write out
# ----------------------------------------------------------------------------------------------------------------------


def create_point_feature(point_list, spatial_ref, shapefile):
    """
    根据坐标点创建点属性数矢量数据。

    :param point_list: nested list. [[ID, X, Y, ...], [...], ...]
    :param spatial_ref: 空间坐标系
    :param shapefile: 存储路径及文件名称

    :return:
    shapefile: point feature 存储路径
    """
    arcpy.env.overwriteOutput = True
    point = arcpy.Point()
    pointGeometryList = []
    for pnt in point_list:
        # point.ID = pnt[0]
        point.X = pnt[1]
        point.Y = pnt[2]

        pointGeometry = arcpy.PointGeometry(point, spatial_ref)
        pointGeometryList.append(pointGeometry)

    arcpy.CopyFeatures_management(pointGeometryList, shapefile)
    return


def create_ployline_feature(line_list, spatial_ref, shapefile):
    """

    :param line_list: nested list [[start, end], [], ...]
    :param spatial_ref: spatial reference
    :param shapefile: file path

    :return:
    shapefile: file and file path
    """
    point, array, feature = arcpy.Point(), arcpy.Array(), []  # A list that will hold each of the Polyline objects

    for line in line_list:
        # For each point, set the x,y properties and add to the array object.
        coord_pair1 = line[0]
        point.X, point.Y = coord_pair1[1], coord_pair1[2]
        array.add(point)

        coord_pair2 = line[1]
        point.X, point.Y = coord_pair2[1], coord_pair2[2]
        array.add(point)

        polyline = arcpy.Polyline(array, spatial_ref)  # Create a Polyline object based on the array of points
        array.removeAll()  # Clear the array for future use
        feature.append(polyline)  # Append to the list of Polyline objects

    arcpy.CopyFeatures_management(feature, shapefile)  # Create a copy of the Polyline objects.
    return


def add_fields_value(feature, fields_name, point_list):
    """
    给feature添加属性字段, 其中需注意的是fields_name和point_list需要相互对应。并添加记录
    测试过，能正常运行。但是，偶尔会报错(ArcGIS系统错误) Error: 99999

    :param feature: shapefile
    :param fields_name: fields name list [[id, t], [x, f], [y, f], ...]
    :param point_list: 属性列表(nested list)  [[id,x,y,z,q, facility_id], [], ...]

    :return:
    feature: fields added
    """
    # 添加字段
    for n, t in fields_name:
        arcpy.AddField_management(feature, n, t)

    arcpy.DeleteField_management(feature, "Id")

    # 更新数值
    cnt, rows = 0, arcpy.UpdateCursor(feature)
    for row in rows:
        for item in fields_name:
            row.setValue(item[0], point_list[cnt][fields_name.index(item)])
            rows.updateRow(row)
        cnt += 1
    return


def write_out(results, describe, csv_file):
    """
    将核心结果输出至指定的CSV文件中。EM和MM模块的结果分别输出

    :param results: dictionary. {Total_household: 11, Treated_household: 10, Coverage: 0.11, AVG_cost: 1111,
        Facility_capex: 111, Facility_o&m: 011, Sewer_capex: 11, Degree: 0.011}
    :param describe: 描述文字
    :param csv_file: 输出文件名及存储位置，

    :return:
    csv_file
    """
    with open(csv_file, 'w') as csv:
        csv.write(describe)
        csv.write('\nName,Value\n')
        for k, v in results.items():
            csv.write("{},{}\n".format(k, v))
    return


if __name__ == "__main__":
    sf = r"E:\Subject\test\test_line.shp"
    ss = r'E:\Subject\ResearchPurpose\\SMUBIT_PRE\source_code\RuST_simulation V0.1.1\example_data\cluster.prj'
    sr = arcpy.SpatialReference(ss)
    ln = [[[1, 311371.3407, 3293892.966], [2, 311355.2758, 3293882.1577]],
          [[2, 311355.2758, 3293882.1577], [3, 311327.4832, 3293831.1853]],
          [[4, 311350.6736, 3293814.1732], [5, 311370.6799, 3293813.0435]],
          [[4, 311350.6736, 3293814.1732], [6, 311382.9774, 3293846.0753]]]

    a = r'E:\Subject\ResearchPurpose\\SMUBIT_PRE\source_code\RuST_simulation V 1.0.0\example_data\cluster.shp'
    bs = r'E:\Subject\test\test_cluster.shp'
    xx = 'markerC'
    o = ['C0', 'C1']
