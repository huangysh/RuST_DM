------------------------------------------------- FOR CSV FILES -----------------------------------------------
本文件对模型的核心内置参数及参数存储文件进行了详细的介绍。方便用户在使用SS时，根据研究区域的实际情况
对参数进行调整，进而得出更加可靠的输出。

-----standard.csv-----农村生活污水处理设施水污染物排放标准
column name
ID		ID number
scale_min		scal of facility(min)
scale_max	scale of facility(max)
water_level	water function zoneing level (from 1 to 5)
tec_level		discharge grade of effluent from sewage treatment facilities (from 1to 4)

=================================================================

-----technology.csv-----污水处理工艺信息表
column name
ID		ID number
tech_name	wastewater treatment technology name
level		discharge grade of effluent from sewage treatment facilities (from 1to 4)
scale_min		scal of facility (min)
scale_max	scale of facility (max)
cost		construction cost (cny/m3)
operation		cost for operation and maintenance (cny/m3)

=================================================================

-----pipe_type.csv-----排水管道规格及单价
ID	diameter（mm）	price（CNY/household）
100	--		--
250	--		--
300	--		--
400	--		--

-----
column name
diameter	管材公称直径，mm
price	管材价格，元/米

=================================================================

-----price.csv-----管网系统相关工程、部件单价
column name
earthwork	price for earthwork, including filling and excavation, cny
inspection well	price of inspection well, cny
drop well		price of drop well， cny
grease tank	price of grease tank，cny
septic tank	price of septic tank, cny
pump station	price of pump station, cny

=================================================================

-----pipeline parameter.csv-----管网系统相关工程、部件单价
column name
population	average population of households
sewage discharge	wastewater discharge of each resident , L/(d. cap)
slope		pipeline slope, ‰
depth_min	minimum  buried depth of pipeline, meter
depth_max	maxmum  buried depth of pipeline, meter
trench wide	trench wide for pipe-laying, meter
sewer lifespan	lifespan of sewer system, year
facility lifespan	lifespan of wastewater treatment facility, year

