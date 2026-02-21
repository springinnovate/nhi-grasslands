var GRASSLAND_PROB_IC = ee.ImageCollection(
  'projects/global-pasture-watch/assets/ggc-30m/v1/nat-semi-grassland_p'
);

var GRASSLAND_CLASS_IC = ee.ImageCollection(
  'projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c'
);

var HMI_IMG = ee.Image(
  'projects/hm-30x30/assets/output/v20240801/HMv20240801_2022s_AA_300'
);

var HII_IC = ee.ImageCollection('projects/HII/v1/hii');


var START_YEAR = 2018;
var END_YEAR = 2020;
var GRASSLAND_CLASS_ID = 2;
var GRASSLAND_PROB_THRESHOLD = 0.6;
var HMI_THRESHOLD = 0.01;
var HII_THRESHOLD = 0.01;


var years = ee.List.sequence(START_YEAR, END_YEAR);

function annualBinaryFromCollection(ic, perYearToBinary) {
  return ee.ImageCollection.fromImages(
    years.map(function(y) {
      y = ee.Number(y);
      var img = ic
        .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y.add(1), 1, 1))
        .first();
      return perYearToBinary(img).rename('g').set('year', y);
    })
  ).sort('year');
}

function noTwoConsecutiveZeros(annualBinaryIC) {
  var list = annualBinaryIC.toList(annualBinaryIC.size());
  var pairOk = ee.ImageCollection.fromImages(
    ee.List.sequence(0, ee.Number(list.size()).subtract(2)).map(function(i) {
      i = ee.Number(i);
      var a = ee.Image(list.get(i)).select('g');
      var b = ee.Image(list.get(i.add(1))).select('g');
      return a.or(b).rename('pair_ok');
    })
  ).reduce(ee.Reducer.min());
  return pairOk.eq(1);
}


var annualGrassClassBinary = annualBinaryFromCollection(
  GRASSLAND_CLASS_IC,
  function(img) {
    return img.select(0).eq(GRASSLAND_CLASS_ID);
  }
);

var categoricalStabilityMask = noTwoConsecutiveZeros(annualGrassClassBinary)
  .rename('categorical_grass_stability_' + START_YEAR + '_' + END_YEAR);

Map.addLayer(
  categoricalStabilityMask.selfMask(),
  {min: 0, max: 1, palette: ['00ff00']},
  'Categorical grass stability ' + START_YEAR + '-' + END_YEAR,
  true
);

// Probability (probability >= GRASSLAND_PROB_THRESHOLD)
var annualGrassProbBinary = annualBinaryFromCollection(
  GRASSLAND_PROB_IC,
  function(img) {
    return img.select(0).gte(GRASSLAND_PROB_THRESHOLD);
  }
);

var probabilityStabilityMask = noTwoConsecutiveZeros(annualGrassProbBinary)
  .rename('probability_grass_stability_' + START_YEAR + '_' + END_YEAR);

Map.addLayer(
  probabilityStabilityMask.selfMask(),
  {min: 0, max: 1, palette: ['00aa00']},
  'Probability grass stability ' + START_YEAR + '-' + END_YEAR,
  true
);


// Human impact masks
var hiiMean = HII_IC.mean();

var lowHiiMask = hiiMean.divide(7000).lt(HII_THRESHOLD);

Map.addLayer(
  lowHiiMask.selfMask(),
  {min: 0, max: 1, palette: ['ff0000']},
  'HII < ' + HII_THRESHOLD,
  true
);

var lowHmiMask = HMI_IMG.lte(HMI_THRESHOLD);

Map.addLayer(
  lowHmiMask.selfMask(),
  {min: 0, max: 1, palette: ['aa00aa']},
  'HMI < ' + HMI_THRESHOLD,
  true
);


// Final indices
var probabilityIntegrityIndex = probabilityStabilityMask
  .and(lowHiiMask)
  .and(lowHmiMask)
  .rename('probability_integrity_index');

Map.addLayer(
  probabilityIntegrityIndex.selfMask(),
  {min: 0, max: 1, palette: ['0000aa']},
  'Probability Integrity Index',
  true
);

var categoricalIntegrityIndex = categoricalStabilityMask
  .and(lowHiiMask)
  .and(lowHmiMask)
  .rename('categorical_integrity_index');

Map.addLayer(
  categoricalIntegrityIndex.selfMask(),
  {min: 0, max: 1, palette: ['0000ff']},
  'Categorical Integrity Index',
  true
);
