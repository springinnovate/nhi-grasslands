Unreleased Changes
------------------
* Added explorer for ERA data. Set growing season temp definition to >= 5C. Only looks at the current year, not the growing season per se. (Southern hemisphere will be split between two seasons). (https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR)
* Added SPEI average index for 12/24/48 months. Chose 12 since we are aggregating per year, since SPEI is monthly, we take the mean value for the year for the 12/24/48 band. (https://developers.google.com/earth-engine/datasets/catalog/CSIC_SPEI_2_10#bands)
* Added annual mean of FLDAS 10-40cm soil moisture for the given year. (We originally proposed GLADS but that is a 3 hour cadence, FLDAS is 1 month, need to consider GEE compute limits). (https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001)
* Added soil carbon at 10/30/60cm from the OpenLandMap Soil Organic Carbon Content dataset. (https://developers.google.com/earth-engine/datasets/catalog/OpenLandMap_SOL_SOL_ORGANIC-CARBON_USDA-6A1C_M_v02#description)
* Added SRTM Multi-Scale Topographic Position Index, negative values show pixel is lower than its surrounds and positive is higher than surroundings. (https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_SRTM_mTPI)
* Added SRTM Topographic Diversity ranging from 0-1 where low values show similar moisture/exposure/hillslope environments and high is a high diversity of them. (https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_SRTM_topoDiversity#description)
