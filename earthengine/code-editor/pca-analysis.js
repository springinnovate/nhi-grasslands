var DEFAULTYEAR = 2019;
var ERA5_DATASET = "ECMWF/ERA5/MONTHLY";
var SAMPLE_SCALE_METERS = 30;
var ERA5_START_YEAR = 1979;
var ANNUAL_TEMPERATURE_STAT = "mean";
var GROWING_SEASON_TEMP_C = 5;
var UPSTREAM_AREA_FOR_STREAMS_KM2 = 25;
var INTERANNUAL_RAINFALL_WINDOW_YEARS = 10;
var CLEAR_LABEL = "(*clear*)";
var MODIS_BURNED_AREA_START_YEAR = 2000;
var MODIS_BURNED_AREA_END_YEAR = 2026;

var MODIS_BURNED_AREA_IC = ee
    .ImageCollection("MODIS/061/MCD64A1")
    .select("BurnDate");
var MODIS_BURN_PROJ = ee.Image(MODIS_BURNED_AREA_IC.first()).projection();

var MODIS_BURN_YEAR_MONTHLY = MODIS_BURNED_AREA_IC.map(function (img) {
    var burnYear = ee.Number(ee.Date(img.get("system:time_start")).get("year"));
    var burned = img.gt(0).selfMask();

    return burned
        .multiply(0)
        .add(burnYear)
        .toInt16()
        .rename("burn_year")
        .setDefaultProjection(MODIS_BURN_PROJ)
        .copyProperties(img, ["system:time_start"]);
});

var GRASSLAND_PROB_IC = ee.ImageCollection(
    "projects/global-pasture-watch/assets/ggc-30m/v1/nat-semi-grassland_p"
);
var HMI_IMG = ee.Image(
    "projects/hm-30x30/assets/output/v20240801/HMv20240801_2022s_AA_300"
);
var HII_IC = ee
    .ImageCollection("projects/HII/v1/hii")
    .filterDate("2001-01-01", "2021-01-01");

var PROBABILITY_INTEGRITY_START_YEAR = 2001;
var PROBABILITY_INTEGRITY_END_YEAR = 2020;
var GRASSLAND_PROB_THRESHOLD = 60;
var HMI_THRESHOLD = 0.1;
var HII_THRESHOLD = 0.08;

function noTwoConsecutiveZerosFromAnnualBinary(buildAnnualBinary) {
    var years = ee.List.sequence(
        PROBABILITY_INTEGRITY_START_YEAR,
        PROBABILITY_INTEGRITY_END_YEAR
    );
    var annualBinaryIC = ee.ImageCollection.fromImages(
        years.map(function (year) {
            year = ee.Number(year);
            return ee
                .Image(buildAnnualBinary(year))
                .rename("g")
                .set("year", year);
        })
    );
    var list = annualBinaryIC.toList(annualBinaryIC.size());
    return ee.ImageCollection.fromImages(
        ee.List.sequence(0, ee.Number(list.size()).subtract(2)).map(
            function (i) {
                i = ee.Number(i);
                return ee.Image(list.get(i)).or(ee.Image(list.get(i.add(1))));
            }
        )
    )
        .reduce(ee.Reducer.min())
        .eq(1);
}

var PROBABILITY_INTEGRITY_INDEX = noTwoConsecutiveZerosFromAnnualBinary(
    function (year) {
        return GRASSLAND_PROB_IC.filterDate(
            ee.Date.fromYMD(year, 1, 1),
            ee.Date.fromYMD(year.add(1), 1, 1)
        )
            .first()
            .select(0)
            .gte(GRASSLAND_PROB_THRESHOLD);
    }
)
    .and(
        noTwoConsecutiveZerosFromAnnualBinary(function (year) {
            return HII_IC.filterDate(
                ee.Date.fromYMD(year, 1, 1),
                ee.Date.fromYMD(year.add(1), 1, 1)
            )
                .mean()
                .divide(7000)
                .lt(HII_THRESHOLD);
        })
    )
    .and(HMI_IMG.lte(HMI_THRESHOLD))
    .selfMask()
    .toByte();

function probabilityIntegrityIndex() {
    return PROBABILITY_INTEGRITY_INDEX;
}

function modisYearsSinceBurn(year) {
    var targetYear = ee
        .Number(year)
        .toInt()
        .max(MODIS_BURNED_AREA_START_YEAR)
        .min(MODIS_BURNED_AREA_END_YEAR);

    var lastBurnYear = MODIS_BURN_YEAR_MONTHLY.filterDate(
        ee.Date.fromYMD(MODIS_BURNED_AREA_START_YEAR, 1, 1),
        ee.Date.fromYMD(targetYear.add(1), 1, 1)
    ).max();

    return lastBurnYear
        .multiply(0)
        .add(targetYear)
        .subtract(lastBurnYear)
        .rename("years_since_burn")
        .toInt8()
        .setDefaultProjection(MODIS_BURN_PROJ);
}

function toCelsius(image) {
    return image.subtract(273.15);
}

function toMillimeters(image) {
    return image.multiply(1000);
}

function yearStart(year) {
    return ee.Date.fromYMD(ee.Number(year).toInt(), 1, 1);
}

function era5MonthlyForYear(year) {
    var start = yearStart(year);
    return ee
        .ImageCollection(ERA5_DATASET)
        .filterDate(start, start.advance(1, "year"));
}

function growingSeasonMonthlyForYear(year) {
    return era5MonthlyForYear(year).map(function (image) {
        var isGrowingSeason = image
            .select("mean_2m_air_temperature")
            .gte(ee.Number(GROWING_SEASON_TEMP_C).add(273.15));
        return image.updateMask(isGrowingSeason);
    });
}

function annualPrecipForYear(year) {
    return toMillimeters(
        era5MonthlyForYear(year).select("total_precipitation").sum()
    );
}

function annualTemperatureForYear(year) {
    var monthly = era5MonthlyForYear(year).select("mean_2m_air_temperature");
    var image = monthly.mean();
    return toCelsius(image);
}

function annualMinTemperatureForYear(year) {
    return toCelsius(
        era5MonthlyForYear(year).select("minimum_2m_air_temperature").min()
    );
}

function annualMaxTemperatureForYear(year) {
    return toCelsius(
        era5MonthlyForYear(year).select("maximum_2m_air_temperature").max()
    );
}

function growingSeasonAverageTemperatureForYear(year) {
    return toCelsius(
        growingSeasonMonthlyForYear(year)
            .select("mean_2m_air_temperature")
            .mean()
    );
}

function growingSeasonAveragePrecipitationForYear(year) {
    return toMillimeters(
        growingSeasonMonthlyForYear(year).select("total_precipitation").mean()
    );
}

function interannualRainfallVariability(endYear) {
    var startYear = ee
        .Number(endYear)
        .subtract(INTERANNUAL_RAINFALL_WINDOW_YEARS - 1)
        .max(ERA5_START_YEAR) //keeps it in range
        .toInt();
    var years = ee.List.sequence(startYear, endYear);
    var annualTotals = ee.ImageCollection.fromImages(
        years.map(function (year) {
            return annualPrecipForYear(year);
        })
    );
    var mean = annualTotals.mean();
    var stdDev = annualTotals.reduce(ee.Reducer.stdDev());
    return stdDev.divide(mean).multiply(100).updateMask(mean.neq(0));
}

function SPEIbase(monthWindow) {
    return function (year) {
        var bandName = "SPEI_" + monthWindow + "_month";
        return ee
            .ImageCollection("CSIC/SPEI/2_10")
            .filterDate(year + "-01-01", year + 1 + "-01-01")
            .select(bandName)
            .mean();
    };
}

function fldasAnnualSoilMoisture(year) {
    return ee
        .ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001")
        .filterDate(year + "-01-01", year + 1 + "-01-01")
        .select("SoilMoi10_40cm_tavg")
        .mean();
}

function soilOrganicCarbon(depth) {
    function _op(year) {
        return ee
            .Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02")
            .select("b" + depth);
    }
    return _op;
}

function srtmTopographicDiversity(year) {
    return ee
        .Image("CSP/ERGo/1_0/Global/SRTM_topoDiversity")
        .select("constant");
}

function srtmMTPI(year) {
    return ee.Image("CSP/ERGo/1_0/Global/SRTM_mTPI").select("elevation");
}

function distanceToStreams(year) {
    var streams = ee
        .Image("MERIT/Hydro/v1_0_1")
        .select("upa")
        .gte(25)
        .unmask(0);

    return streams
        .fastDistanceTransform()
        .sqrt()
        .multiply(ee.Image.pixelArea().sqrt());
}

function makeLayerDefinition(name, build, defaultRange) {
    return {
        name: name,
        build: function (year) {
            return ee.Image(build(year)).rename("B0");
        },
        defaultRange: defaultRange
    };
}

function debugPredictorCoverage(year, region) {
  var panel = ui.Panel([], ui.Panel.Layout.flow('vertical'), {
    position: 'bottom-right',
    width: '420px',
    padding: '8px'
  });
  Map.add(panel);
  panel.add(ui.Label('Predictor coverage debug: ' + year));

  PCA_LAYER_DEFS.forEach(function(def, i) {
    var bandName = PCA_BAND_NAMES[i];
    var img = ee.Image(def.build(year)).rename(bandName).float();

    if (def.name === 'Years since last burn from current year') {
      img = img.unmask(parseInt(DEFAULTYEAR, 10) - MODIS_BURNED_AREA_START_YEAR + 1);
    }

    var valid = img.mask().reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: region,
      scale: PCA_ANALYSIS_SCALE_METERS,
      maxPixels: PCA_MAX_PIXELS,
      tileScale: 4
    }).get(bandName);

    var minmax = img.reduceRegion({
      reducer: ee.Reducer.minMax(),
      geometry: region,
      scale: PCA_ANALYSIS_SCALE_METERS,
      maxPixels: PCA_MAX_PIXELS,
      tileScale: 4
    });

    valid.evaluate(function(v) {
      minmax.evaluate(function(mm) {
        panel.add(ui.Label(
          def.name + ' | valid=' + v + ' | stats=' + JSON.stringify(mm)
        ));
      });
    });
  });
}

var LAYER_DEFINITIONS = [
    makeLayerDefinition(
        "Grassland Reference Sites",
        probabilityIntegrityIndex,
        { min: 0, max: 1 }
    ),
    makeLayerDefinition(
        "Mean annual temperature (C) (ESA/Monthly)",
        annualTemperatureForYear,
        { min: -20, max: 30 }
    ),
    makeLayerDefinition(
        "Minimum annual temperature (C) (ESA/Monthly)",
        annualMinTemperatureForYear,
        { min: -40, max: 20 }
    ),
    makeLayerDefinition(
        "Maximum annual temperature (C) (ESA/Monthly)",
        annualMaxTemperatureForYear,
        { min: 0, max: 45 }
    ),
    makeLayerDefinition(
        "Annual precipitation (mm) (ESA/Monthly)",
        annualPrecipForYear,
        {
            min: 0,
            max: 3000
        }
    ),
    makeLayerDefinition(
        "Growing season avg temp (C)",
        growingSeasonAverageTemperatureForYear,
        { min: 0, max: 25 }
    ),
    makeLayerDefinition(
        "Growing season avg precipitation (mm)",
        growingSeasonAveragePrecipitationForYear,
        { min: 0, max: 250 }
    ),
    makeLayerDefinition(
        "Interannual rainfall variability (CV%, " +
            INTERANNUAL_RAINFALL_WINDOW_YEARS +
            "-year)",
        interannualRainfallVariability,
        { min: 0, max: 50 }
    ),
    makeLayerDefinition("Annual mean of SPEI 12 month index", SPEIbase(12), {
        min: -2,
        max: 2
    }),
    makeLayerDefinition("Annual mean of SPEI 24 month index", SPEIbase(24), {
        min: -2,
        max: 2
    }),
    makeLayerDefinition("Annual mean of SPEI 48 month index", SPEIbase(48), {
        min: -2,
        max: 2
    }),
    makeLayerDefinition(
        "Annual soil moisture (GLDAS 10-40 cm)",
        fldasAnnualSoilMoisture,
        {
            min: 0,
            max: 40
        }
    ),
    makeLayerDefinition(
        "Soil organic carbon (10 cm, OpenLandMap, no time domain)",
        soilOrganicCarbon(10),
        {
            min: 0,
            max: 25
        }
    ),
    makeLayerDefinition(
        "Soil organic carbon (30 cm, OpenLandMap, no time domain)",
        soilOrganicCarbon(30),
        {
            min: 0,
            max: 25
        }
    ),
    makeLayerDefinition(
        "Soil organic carbon (60 cm, OpenLandMap, no time domain)",
        soilOrganicCarbon(60),
        {
            min: 0,
            max: 25
        }
    ),
    makeLayerDefinition(
        "SRTM topographic diversity (0-1 low-high diversity)",
        srtmTopographicDiversity,
        {
            min: 0,
            max: 1
        }
    ),

    makeLayerDefinition(
        "SRTM mTPI <0 to >0 lower or higher neighbors",
        srtmMTPI,
        {
            min: -30,
            max: 30
        }
    ),
    makeLayerDefinition(
        "Distance to streams (m) thresholded from " +
            UPSTREAM_AREA_FOR_STREAMS_KM2 +
            " km2 drainage",
        distanceToStreams,
        {
            min: 1,
            max: 5000
        }
    ),
    makeLayerDefinition(
        "Years since last burn from current year",
        modisYearsSinceBurn,
        {
            min: 0,
            max: 26
        }
    )
];



var PCA_ANALYSIS_SCALE_METERS = 500;
var PCA_MAX_PIXELS = 1e7;
var PCA_REFERENCE_POINTS_PER_CLASS = 400;

function safeBandName(name) {
  return name.replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

var PCA_LAYER_DEFS = LAYER_DEFINITIONS.filter(function(def) {
  return def.name !== 'Grassland Reference Sites';
});
var PCA_LABELS = PCA_LAYER_DEFS.map(function(def) { return def.name; });
var PCA_BAND_NAMES = PCA_LABELS.map(safeBandName);
var PCA_PC_NAMES = PCA_BAND_NAMES.map(function(_, i) { return 'PC' + (i + 1); });

function predictorBandAt(i, year) {
  var def = PCA_LAYER_DEFS[i];
  var band = ee.Image(def.build(year)).rename(PCA_BAND_NAMES[i]).float();
  if (def.name === 'Years since last burn from current year') {
    band = band.unmask(parseInt(year, 10) - MODIS_BURNED_AREA_START_YEAR + 1);
  }
  return band;
}

function buildPredictorImage(year) {
  var image = predictorBandAt(0, year);
  for (var i = 1; i < PCA_LAYER_DEFS.length; i++) {
    image = image.addBands(predictorBandAt(i, year));
  }
  return image;
}

function zScoreImage(image, region, scale) {
  var meanDict = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: scale,
    maxPixels: PCA_MAX_PIXELS,
    tileScale: 4
  });

  var sdDict = image.reduceRegion({
    reducer: ee.Reducer.stdDev(),
    geometry: region,
    scale: scale,
    maxPixels: PCA_MAX_PIXELS,
    tileScale: 4
  });

  var meanImage = ee.Image.constant(PCA_BAND_NAMES.map(function(name) {
    return ee.Number(meanDict.get(name));
  })).rename(PCA_BAND_NAMES);

  var sdImage = ee.Image.constant(PCA_BAND_NAMES.map(function(name) {
    return ee.Number(sdDict.get(name)).max(1e-6);
  })).rename(PCA_BAND_NAMES);

  return image.subtract(meanImage).divide(sdImage);
}

function runRegionPca(year, region) {
  var predictors = buildPredictorImage(year).select(PCA_BAND_NAMES);
  var z = zScoreImage(predictors, region, PCA_ANALYSIS_SCALE_METERS);
  var arrays = z.toArray();

  var covar = arrays.reduceRegion({
    reducer: ee.Reducer.centeredCovariance(),
    geometry: region,
    scale: PCA_ANALYSIS_SCALE_METERS,
    maxPixels: PCA_MAX_PIXELS,
    tileScale: 4
  });

  var covarArray = ee.Array(covar.get('array'));
  var eigens = covarArray.eigen();
  var eigenValues = eigens.slice(1, 0, 1).project([0]);
  var eigenVectors = eigens.slice(1, 1);
  var totalVar = ee.Number(eigenValues.toList().reduce(ee.Reducer.sum()));
  var variancePct = eigenValues.divide(totalVar).multiply(100);

  var pcs = ee.Image(eigenVectors)
    .matrixMultiply(arrays.toArray(1))
    .arrayProject([0])
    .arrayFlatten([PCA_PC_NAMES]);

  return {
    pcs: pcs,
    variancePct: variancePct,
    eigenVectors: eigenVectors
  };
}

function makeLoadingFeatures(eigenVectors) {
  return ee.FeatureCollection(PCA_LABELS.map(function(label, i) {
    return ee.Feature(null, {
      variable: label,
      pc1: ee.Number(eigenVectors.get([0, i])),
      pc2: ee.Number(eigenVectors.get([1, i])),
      pc3: ee.Number(eigenVectors.get([2, i]))
    });
  }));
}

var pcaPanel = ui.Panel([], ui.Panel.Layout.flow('vertical'), {
  position: 'top-right',
  width: '420px',
  padding: '8px'
});
Map.add(pcaPanel);

Map.addLayer(
  PROBABILITY_INTEGRITY_INDEX,
  {min: 0, max: 1, palette: ['ff0000', '00aa00']},
  'Grassland Reference Sites',
  true
);

var emptyPc1 = ee.Image.constant(0).rename('PC1').selfMask();
var emptyPc2 = ee.Image.constant(0).rename('PC2').selfMask();
//var emptyRgb = ee.Image.constant([0, 0, 0]).rename(['PC1', 'PC2', 'PC3']).selfMask();

var pcaPc2Layer = Map.addLayer(
  emptyPc2,
  {min: -3, max: 3},
  'PCA PC2',
  false
);

var pcaPc1Layer = Map.addLayer(
  emptyPc1,
  {min: -3, max: 3},
  'PCA PC1',
  true
);


function renderPca(region) {
  pcaPanel.clear();
  pcaPanel.add(ui.Label('PCA for ' + DEFAULTYEAR));

  var predictors = buildPredictorImage(DEFAULTYEAR);

  var samples = predictors.sample({
    region: region,
    scale: PCA_ANALYSIS_SCALE_METERS,
    numPixels: 5000,
    geometries: false,
    dropNulls: true,
    tileScale: 4
  });

  samples.size().evaluate(function(n) {
    if (!n || n < 3) {
      pcaPanel.add(ui.Label(
        'Not enough valid pixels for PCA at this polygon/scale. Draw a larger polygon or reduce PCA_ANALYSIS_SCALE_METERS.'
      ));
      return;
    }

    var result = runRegionPca(DEFAULTYEAR, region);
    var loadings = makeLoadingFeatures(result.eigenVectors);

    var referenceSamples = result.pcs
      .addBands(PROBABILITY_INTEGRITY_INDEX.unmask(0).rename('reference'))
      .stratifiedSample({
        numPoints: PCA_REFERENCE_POINTS_PER_CLASS,
        classBand: 'reference',
        region: region,
        scale: PCA_ANALYSIS_SCALE_METERS,
        seed: 0,
        dropNulls: true,
        geometries: false,
        tileScale: 4
      });

    pcaPc1Layer.setEeObject(result.pcs.select('PC1').clip(region));
    pcaPc1Layer.setVisParams({min: -3, max: 3});

    pcaPc2Layer
.setEeObject(result.pcs.select(['PC1', 'PC2', 'PC3']).clip(region));
    pcaPc2Layer
.setVisParams({bands: ['PC1', 'PC2', 'PC3'], min: -2, max: 2});

    pcaPanel.add(
      ui.Chart.array.values(result.variancePct, 0, PCA_PC_NAMES)
        .setChartType('ColumnChart')
        .setOptions({
          title: 'Explained variance (%)',
          legend: {position: 'none'},
          hAxis: {slantedText: true, slantedTextAngle: 60}
        })
    );

    pcaPanel.add(
      ui.Chart.feature.byFeature(loadings, 'variable', ['pc1', 'pc2'])
        .setChartType('ColumnChart')
        .setOptions({
          title: 'PC1 / PC2 weights',
          hAxis: {slantedText: true, slantedTextAngle: 60}
        })
    );

    pcaPanel.add(
      ui.Chart.feature.groups(referenceSamples, 'PC1', 'PC2', 'reference')
        .setChartType('ScatterChart')
        .setOptions({
          title: 'PC1 vs PC2 by reference flag',
          hAxis: {title: 'PC1'},
          vAxis: {title: 'PC2'}
        })
    );
  });
}

function debugStackIntersection(year, region) {
  var predictors = buildPredictorImage(year).select(PCA_BAND_NAMES);
  var validMask = predictors.mask().reduce(ee.Reducer.min()).rename('valid');

  var count = validMask.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: region,
    scale: PCA_ANALYSIS_SCALE_METERS,
    maxPixels: PCA_MAX_PIXELS,
    tileScale: 4
  });

  print('Full predictor intersection count', count);
  Map.addLayer(validMask.clip(region), {min: 0, max: 1}, 'PCA valid intersection', true);
}


var drawingTools = Map.drawingTools();
drawingTools.setShown(true);
drawingTools.setDrawModes(['polygon', 'rectangle']);
/*drawingTools.onDraw(function(geom, layer) {
  var region = layer.toGeometry();
  //debugPredictorCoverage(DEFAULTYEAR, region);
  renderPca(region);
  //debugStackIntersection(DEFAULTYEAR, region);
});
drawingTools.onEdit(function(geom, layer) {
  renderPca(layer.toGeometry());
});
*/
pcaPanel.add(ui.Label('Draw a polygon or rectangle to run PCA'));

///////////

var drawnOutlineLayer = null;

function showDrawnOutline(region) {
  var outlineImage = ee.FeatureCollection([ee.Feature(region)]).style({
    color: 'magenta',
    width: 2,
    fillColor: '00000000'
  });

  if (drawnOutlineLayer) {
    Map.remove(drawnOutlineLayer);
  }

  drawnOutlineLayer = Map.addLayer(outlineImage, null, 'Drawn region', true);
}

drawingTools.onDraw(function(geom, layer) {
  var region = layer.toGeometry();
  layer.setShown(false);
  showDrawnOutline(region);
  renderPca(region);
  });

drawingTools.onEdit(function(geom, layer) {
  var region = layer.toGeometry();
  layer.setShown(false);
  showDrawnOutline(region);
  renderPca(region);
});