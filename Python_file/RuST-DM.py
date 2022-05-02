# coding: utf-8 -*-
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

# General import
import time
from functions import *

print("RuST-DM model!")
print(time.asctime(time.localtime(time.time())))

folder_s = os.getcwd()  # 根文件夹
folder = folder_s.strip("\Python_file")

# set input data/folder/parameter
input_folder = folder + "\\" + "Input_data"
output_folder = folder + "\\" + "Results"  # USER INOUT
household_file = input_folder + r"\household.shp"
road_file = input_folder + r"\road.shp"
DEM_ENVI_file = input_folder + r"\DEM.shp"

EM_folder = output_folder + "\\" + 'EM_results'
MM_folder = output_folder + "\\" + 'MM_results'
check_folder(EM_folder)
check_folder(MM_folder)

connector_name = r""  # optional
if connector_name:
    connector_file = input_folder + "\\" + connector_name
else:
    connector_file = r''

# **********************************************************************************************************************
# User input parameters
# **********************************************************************************************************************

C_RuST = 1.0         # RuST coverage
err = 0.1            # 覆盖率计算允许误差
coeff = 5            # 管网连接系数
search_coeff = 1.5   # A*搜索半径系数
expect = 20000       # 户均预期造价
near_coeff = 3.0     # 邻近WTP搜索系数
field = 'markerC'    # 类簇标识
marker = 'L0'        # 离散点标识码

# ----------------------------------------------------------------------------------------------------------------------
# set built-in datasets
standard_file = folder + r"\Database" + r"\standard.csv"
technology_file = folder + r"\Database" + r"\technology.csv"
parameter_file = folder + r"\Database" + r"\pipe_para.csv"
pipe_type_file = folder + r"\Database" + r"\pipe_type.csv"
price_file = folder + r"\Database" + r"\price.csv"

# ----------------------------------------------------------------------------------------------------------------------
# read built-in data
standard = read_standard(standard_file)
technology = read_technology_dataset(technology_file)
parameter = read_parameters(parameter_file)  # sewer design parameters
pipe_type = read_pipe_type(pipe_type_file)
price = read_parameters(price_file)

Q = parameter['seg']  # sewage, m3/day  per capita
POP = parameter['pop']    # average population, per household
depth_min, depth_max, slope = parameter['dep_min'], parameter['dep_max'], parameter['slope']
wide, s_ls, f_ls = parameter['wide'], parameter['s_ls'], parameter['f_ls']


# ======================================================================================================================
# read input data
household_nodes, spatial_ref = read_household(household_file, field, Q, POP)
road_nodes, road_pair = read_road(road_file)
dem, envi = read_raster(DEM_ENVI_file)
if connector_file:
    connector = read_connector(connector_file)
else:
    connector = None

print("Read input data!")
print(time.asctime(time.localtime(time.time())))

# pretreatment
household_nodes = assign_value(household_nodes, envi, 3, 6)
dem = assign_value(dem, envi, 3, 6)
road_nodes = assign_value(road_nodes, dem, 3, 3)   # UPDATE HEIGHT
road_nodes = assign_value(road_nodes, envi, 3, 6)
near_road = get_nearest_road(road_nodes, household_nodes)

house_node_dict = list_to_dict(household_nodes)
road_dict = list_to_dict(road_nodes)
dem_dict = list_to_dict(dem)

nodes_dict = deepcopy(house_node_dict)  # household,
nodes_dict.update(road_dict)
nodes_dict.update(dem_dict)

exp_cost = expect * 0.8 - 7 * 30 - price['grease'] - price['septic']  # todo

# household filter
house_cluster = get_household_cluster(household_nodes)
house_node_dict, household_num, rate, error = household_filter(house_cluster, marker, C_RuST, err)

house_node_dict = dict([(nodes_dict[i[0]][5], i) for i in house_node_dict])

household_cluster = get_dictionary(house_node_dict, marker)
household_id = []  # id list
for _, v in house_node_dict.items():
    household_id.extend(v)
household = [nodes_dict[i] for i in household_id]
household_dict = list_to_dict(deepcopy(household), 0)

print("\nData is successfully pretreated!!!!")
print(time.asctime(time.localtime(time.time())))

print("====================================================================================")
print("Number of household: %d (%d)" % (household_num, len(household_nodes)))
print("RuST coverage: %0.3f (%0.3f)" % (rate, C_RuST))
print("\n====================================================================================")
print("Ready to performed Expansion Module......")
print(time.asctime(time.localtime(time.time())))
print("************************************************************************************")

wtp_dict, source_dict = Expansion_Module_cluster(household_dict, household_cluster, standard, technology, price,
                                                 pipe_type, POP, Q, depth_min, depth_max, wide, slope, f_ls, s_ls)

DD_EM = round(degree_of_dec(wtp_dict, source_dict), 5)

print("DD_em: %0.5f" % DD_EM)

results = tidy_results(wtp_dict, source_dict, nodes_dict, road_dict, road_pair, near_road, standard, technology, price,
                       pipe_type, Q, POP, coeff, depth_min, depth_max, wide, slope, f_ls, s_ls)
wtp_dict, sewer_dict, _ = results

# check sewer overlap
wtp_dict, source_dict = check_overlap(wtp_dict, sewer_dict, source_dict)
results_new = tidy_results(wtp_dict, source_dict, nodes_dict, road_dict, road_pair, near_road, standard, technology,
                           price, pipe_type, Q, POP, coeff, depth_min, depth_max, wide, slope, f_ls, s_ls)
wtp_dict, sewer_dict, pipe_dict = results_new

# renew dictionary KEY
wtp_dict, sewer_dict, pipe_dict, source_dict = revise_wtp_id(wtp_dict, sewer_dict, pipe_dict, source_dict)

capex_w = round(sum([i[12][0] for i in wtp_dict.values()]) / household_num, 2)
opex_w = round(sum([i[12][1] for i in wtp_dict.values()]) / household_num, 2)
capex_s = round(sum([i[12][2] for i in wtp_dict.values()]) / household_num, 2)
opex_s = round(sum([i[12][3] for i in wtp_dict.values()]) / household_num, 2)
total = round(capex_w + capex_s + opex_w + opex_s, 2)
EM_dict = {"Total Household": len(household_nodes), "Treated Household": household_num, "Coverage": rate, "Degree": DD_EM,
           "Total Cost": total, "WTP Capex": capex_w, "WTP Opex": opex_w, "Sewer Capex": capex_s, "Sewer Opex": opex_s}
print("\n**************************************************************************************")

print("Expansion Module is successfully performed!!!!")
print(time.asctime(time.localtime(time.time())))
print("--------------------------------------------------------------------------------------")
print("Number of facility: %d" % len(wtp_dict))
print("degree of decentralization: %0.5f" % DD_EM)
print("Investment breakdown (CNY/household)")
print("sum: %0.2f, WTP capex: %0.2f, WTP opex: %0.2f, sewer capex: %0.2f, sewer opex: %0.2f" %
      (total, capex_w, opex_w, capex_s, opex_s))
print("--------------------------------------------------------------------------------------")
print("\n**************************************************************************************")


# ====================================================================================
# EM Write out！！
# ====================================================================================

print("Ready to write EM results......")
print("**************************************************************************************")
print(EM_folder)
# make a copy
EM_wtp = deepcopy(wtp_dict)
EM_sewer = deepcopy(sewer_dict)
EM_pipe = deepcopy(pipe_dict)  # pipe node
EM_source = deepcopy(source_dict)

# output file
EM_wtp_file = EM_folder + "\\" + "wtp_em.shp"
EM_sewer_file = EM_folder + "\\" + "sewer_em.shp"
EM_pipe_file = EM_folder + "\\" + "pipe_em.shp"
EM_source_file = EM_folder + "\\" + "source_em.shp"
EM_drop_file = EM_folder + "\\" + "drop_em.shp"
EM_pump_file = EM_folder + "\\" + "pump_em.shp"
EM_results = output_folder + "\\" + "EM_results.csv"

# WTP
print("Writing out WTP...")
WTP_list = []
for v in EM_wtp.values():
    WTP_list.append([v[0], v[1], v[2], v[3], v[4], v[10][0], v[10][1], v[10][2], sum(v[12])] + v[12])

try:
    create_point_feature(WTP_list, spatial_ref, EM_wtp_file)
    fields_name = [["wtp_ID", "TEXT"], ["X", "DOUBLE"], ["Y", "DOUBLE"], ["Elev", "SHORT"], ["Flow", "DOUBLE"],
                   ["ENVI_req", "TEXT"], ["Sta_lev", "TEXT"], ["Tech_ID", "TEXT"], ["Total_C", "DOUBLE"],
                   ["WTP_CAP", "DOUBLE"], ["WTP_OP", "DOUBLE"], ["SEWER_CAP", "DOUBLE"], ["SEWER_OP", "DOUBLE"]]
    add_fields_value(EM_wtp_file, fields_name, WTP_list)
except:
    raise MyException("EMPTY LIST!!!")

# SEWER
print("Writing out sewer network...")
sewer_pipe, sewer_list = [], []
for k, v in EM_sewer.items():
    if v:
        for pl in v:
            if pl[0][0] == "H" and pl[1][0] != "H":
                continue
            elif pl[1][0] == "H" and pl[1] not in wtp_dict.keys():
                continue
            elif pl[1][0] == "H" and pl[1] in wtp_dict.keys():
                sewer_pipe.append([nodes_dict[pl[0]], nodes_dict[pl[1]]])
                diameter = get_diameter(pl[3], pipe_type)
                length, _ = get_distance(nodes_dict[pl[0]], nodes_dict[pl[1]])
                sewer_list.append([k, pl[3], diameter, length])  # wtp_id, flow, diameter, length
            else:
                sewer_pipe.append([nodes_dict[pl[0]], nodes_dict[pl[1]]])
                diameter = get_diameter(pl[3], pipe_type)
                length, _ = get_distance(nodes_dict[pl[0]], nodes_dict[pl[1]])
                sewer_list.append([k, pl[3], diameter, length])  # wtp_id, flow, diameter, length
    else:
        continue
if sewer_pipe:  # not empty
    create_ployline_feature(sewer_pipe, spatial_ref, EM_sewer_file)
    fields_name = [["wtp_ID", "TEXT"], ["Flow", "DOUBLE"], ["Diameter", "SHORT"], ["Length", "DOUBLE"]]
    add_fields_value(EM_sewer_file, fields_name, sewer_list)
else:
    print("NO SEWER NETWORK!!!")

# PIPE NODE
print("Writing out pipe nodes...")
pipe_nodes = []
for k, v in EM_pipe.items():
    if v:  # not empty
        for p in v:
            if p[0][0] != "H":  # not including source node
                pipe_nodes.append([k, p[1], p[2], p[3], p[4], p[9][2], p[9][3], p[9][4]])
if pipe_nodes:  # not empty
    create_point_feature(pipe_nodes, spatial_ref, EM_pipe_file)
    fields_name = [["wtp_ID", "TEXT"], ["X", "DOUBLE"], ["Y", "DOUBLE"], ["Elev", "SHORT"],["Flow", "DOUBLE"],
                   ["Depth", "DOUBLE"]]
    add_fields_value(EM_pipe_file, fields_name, pipe_nodes)
else:
    print("NO SEWER NETWORK!!!")

# PUMP
print("Writing out pump station...")
pump_list = [i[:5] for i in pipe_nodes if i[7]]
if pump_list:  # not empty
    create_point_feature(pump_list, spatial_ref, EM_pump_file)
    fields_name = [["wtp_ID", "TEXT"], ["X", "DOUBLE"], ["Y", "DOUBLE"], ["Elev", "SHORT"], ["Flow", "DOUBLE"]]
    add_fields_value(EM_pump_file, fields_name, pump_list)
else:
    print("NO PUMP STATION!!!")

# DROP
print("Writing out drop well...")
drop_list = [i[:5] for i in pipe_nodes if i[6]]
if drop_list:  # not empty
    create_point_feature(drop_list, spatial_ref, EM_drop_file)
    fields_name = [["wtp_ID", "TEXT"], ["X", "DOUBLE"], ["Y", "DOUBLE"], ["Elev", "SHORT"], ["Flow", "DOUBLE"]]
    add_fields_value(EM_drop_file, fields_name, drop_list)
else:
    print("NO DROP WELL!!!")

# Source Node
print("Writing out source nodes...")
source_list = []
for k, v in EM_source.items():
    for i in v:
        source_list.append([k, i[1], i[2], i[3], i[4]])
try:
    create_point_feature(source_list, spatial_ref, EM_source_file)
    fields_name = [["wtp_ID", "TEXT"], ["X", "DOUBLE"], ["Y", "DOUBLE"], ["Elev", "SHORT"], ["Flow", "DOUBLE"]]
    add_fields_value(EM_source_file, fields_name, source_list)
except:
    raise MyException("EMPTY LIST!!!")

# write to csv file
describe = time.asctime(time.localtime(time.time())) + "\n" + "Fallowing the modelling results of Expansion Module."
write_out(EM_dict, describe, EM_results)

print(time.asctime(time.localtime(time.time())))
print("Expansion Module results are successfully wrote!!!!")

print("Ready to performed Merging Module......")
print(time.asctime(time.localtime(time.time())))

if len(wtp_dict.keys()) > 1:
    MM_r = merging_module(wtp_dict, sewer_dict, source_dict, nodes_dict, dem, near_road, road_dict, road_pair, standard,
                          technology, price, pipe_type, Q, POP, coeff, exp_cost, search_coeff, depth_min, depth_max,
                          wide, slope, f_ls, s_ls, near_coeff)
    facility_dict, sewer_network, source_node = MM_r

    # check sewer overlap
    WTPs_dict, sources_dict = check_overlap(facility_dict, sewer_network, source_node)
    new_MM_r = tidy_results(WTPs_dict, sources_dict, nodes_dict, road_dict, road_pair, near_road, standard, technology,
                            price, pipe_type, Q, POP, coeff, depth_min, depth_max, wide, slope, f_ls, s_ls)
    facility_dict, sewer_network, source_node = new_MM_r
    pipe_nodes = get_pipe_nodes_results(nodes_dict, sewer_network, pipe_type)
else:
    facility_dict, sewer_network, source_node, pipe_nodes = wtp_dict, sewer_dict, source_dict, pipe_dict

DD_MM = round(degree_of_dec(facility_dict, source_node), 5)

print("\n**************************************************************************************")

# Final results
capex_w = round(sum([i[12][0] for i in facility_dict.values()]) / household_num, 2)
opex_w = round(sum([i[12][1] for i in facility_dict.values()]) / household_num, 2)
capex_s = round(sum([i[12][2] for i in facility_dict.values()]) / household_num, 2)
opex_s = round(sum([i[12][3] for i in facility_dict.values()]) / household_num, 2)
total = round(capex_w + capex_s + opex_w + opex_s, 2)
MM_dict = {"Total Household": len(household_nodes), "Treated Household": household_num, "Coverage": rate, "Degree": DD_EM,
           "Total Cost": total, "WTP Capex": capex_w, "WTP Opex": opex_w, "Sewer Capex": capex_s, "Sewer Opex": opex_s}
print("\n**************************************************************************************")

print("Merging Module is successfully performed!!!!")
print(time.asctime(time.localtime(time.time())))
print("--------------------------------------------------------------------------------------")
print("Number of facility: %d" % len(facility_dict))
print("degree of decentralization: %0.5f" % DD_MM)
print("Investment breakdown (CNY/household)")
print("sum: %0.2f, WTP capex: %0.2f, WTP opex: %0.2f, sewer capex: %0.2f, sewer opex: %0.2f" %
      (total, capex_w, opex_w, capex_s, opex_s))
print("--------------------------------------------------------------------------------------")
print("\n**************************************************************************************")

# ====================================================================================
# EM Write out！！
# ====================================================================================

print("Ready to write MM results......")
print("**************************************************************************************")
print(MM_folder)

# SET OUTPUT FILE
MM_wtp_file = MM_folder + "\\" + "wtp_mm.shp"
MM_sewer_file = MM_folder + "\\" + "sewer_mm.shp"
MM_pipe_file = MM_folder + "\\" + "pipe_mm.shp"
MM_source_file = MM_folder + "\\" + "source_mm.shp"
MM_drop_file = MM_folder + "\\" + "drop_mm.shp"
MM_pump_file = MM_folder + "\\" + "pump_mm.shp"
MM_results = output_folder + "\\" + "MM_results.csv"

# WTP
print("Writing out WTP...")
WTP_list = []
for v in facility_dict.values():
    WTP_list.append([v[0], v[1], v[2], v[3], v[4], v[10][0], v[10][1], v[10][2], sum(v[12])] + v[12])
try:
    create_point_feature(WTP_list, spatial_ref, MM_wtp_file)
    fields_name = [["wtp_ID", "TEXT"], ["X", "DOUBLE"], ["Y", "DOUBLE"], ["Elev", "SHORT"], ["Flow", "DOUBLE"],
                   ["ENVI_req", "TEXT"], ["Sta_lev", "TEXT"], ["Tech_ID", "TEXT"], ["Total_C", "DOUBLE"],
                   ["WTP_CAP", "DOUBLE"], ["WTP_OP", "DOUBLE"], ["SEWER_CAP", "DOUBLE"], ["SEWER_OP", "DOUBLE"]]
    add_fields_value(MM_wtp_file, fields_name, WTP_list)
except:
    raise MyException("EMPTY LIST!!!")

# SEWER
print("Writing out sewer network...")
sewer_pipe, sewer_list = [], []
for k, v in sewer_network.items():
    if v:
        for pl in v:
            if pl[0][0] == "H" and pl[1][0] != "H":
                continue
            elif pl[1][0] == "H" and pl[1] not in wtp_dict.keys():
                continue
            elif pl[1][0] == "H" and pl[1] in wtp_dict.keys():
                sewer_pipe.append([nodes_dict[pl[0]], nodes_dict[pl[1]]])
                diameter = get_diameter(pl[3], pipe_type)
                length, _ = get_distance(nodes_dict[pl[0]], nodes_dict[pl[1]])
                sewer_list.append([k, pl[3], diameter, length])  # wtp_id, flow, diameter, length
            else:
                sewer_pipe.append([nodes_dict[pl[0]], nodes_dict[pl[1]]])
                diameter = get_diameter(pl[3], pipe_type)
                length, _ = get_distance(nodes_dict[pl[0]], nodes_dict[pl[1]])
                sewer_list.append([k, pl[3], diameter, length])  # wtp_id, flow, diameter, length
    else:
        continue
if sewer_pipe:  # not empty
    create_ployline_feature(sewer_pipe, spatial_ref, MM_sewer_file)
    fields_name = [["wtp_ID", "TEXT"], ["Flow", "DOUBLE"], ["Diameter", "SHORT"], ["Length", "DOUBLE"]]
    add_fields_value(MM_sewer_file, fields_name, sewer_list)
else:
    print("NO SEWER NETWORK!!!")

# PIPE NODE
print("Writing out pipe nodes...")
pipe_node = []
for k, v in pipe_nodes.items():
    if v:  # not empty
        for p in v:
            pipe_node.append([k, p[1], p[2], p[3], p[4], p[9][2], p[9][3], p[9][4]])
if pipe_node:  # not empty
        create_point_feature(pipe_node, spatial_ref, MM_pipe_file)
        fields_name = [["wtp_ID", "TEXT"], ["X", "DOUBLE"], ["Y", "DOUBLE"], ["Elev", "SHORT"], ["Flow", "DOUBLE"],
                       ["Depth", "DOUBLE"]]
        add_fields_value(MM_pipe_file, fields_name, pipe_node)
else:
    print("NO PIPE NODES!!!")

# PUMP
print("Writing out pump station...")
pump_list = [i[:5] for i in pipe_node if i[7]]
if pump_list:  # not empty
    create_point_feature(pump_list, spatial_ref, MM_pump_file)
    fields_name = [["wtp_ID", "TEXT"], ["X", "DOUBLE"], ["Y", "DOUBLE"], ["Elev", "SHORT"], ["Flow", "DOUBLE"]]
    add_fields_value(MM_pump_file, fields_name, pump_list)
else:
    print("NO PUMP STATION!!!")

# DROP
print("Writing out drop well...")
drop_list = [i[:5] for i in pipe_node if i[6]]
if drop_list:  # not empty
    create_point_feature(drop_list, spatial_ref, MM_drop_file)
    fields_name = [["wtp_ID", "TEXT"], ["X", "DOUBLE"], ["Y", "DOUBLE"], ["Elev", "SHORT"], ["Flow", "DOUBLE"]]
    add_fields_value(MM_drop_file, fields_name, drop_list)
else:
    print("NO DROP WELL!!!")

# Source Node
print("Writing out source nodes...")
source_list = []
for k, v in source_dict.items():
    for i in v:
        source_list.append([k, i[1], i[2], i[3], i[4]])
try:
    create_point_feature(source_list, spatial_ref, MM_source_file)
    fields_name = [["wtp_ID", "TEXT"], ["X", "DOUBLE"], ["Y", "DOUBLE"], ["Elev", "SHORT"], ["Flow", "DOUBLE"]]
    add_fields_value(MM_source_file, fields_name, source_list)
except:
    raise MyException("EMPTY LIST!!!")

# write to csv file
describe = time.asctime(time.localtime(time.time())) + "\n" + "Fallowing the modelling results of Expansion Module."
write_out(MM_dict, describe, MM_results)

print("RuST-OM is successfully performed!!!!")
print(time.asctime(time.localtime(time.time())))