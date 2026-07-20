var GRASSLAND_PROB_IC = ee.ImageCollection(
  "projects/global-pasture-watch/assets/ggc-30m/v1/nat-semi-grassland_p"
);

var GRASSLAND_CLASS_IC = ee.ImageCollection(
  "projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c"
);

var HMI_IMG = ee.Image(
  "projects/hm-30x30/assets/output/v20240801/HMv20240801_2022s_AA_300"
);

var START_YEAR = 2001;
var END_YEAR = 2020;

// Hard-coding 2001-2020 because that's the only dates these are defined
var HII_IC = ee
  .ImageCollection("projects/HII/v1/hii")
  .filterDate(
    ee.Date.fromYMD(START_YEAR, 1, 1),
    ee.Date.fromYMD(END_YEAR + 1, 1, 1)
  );

var GRASSLAND_CLASS_ID = 2;
var GRASSLAND_PROB_THRESHOLD = 60;
var HMI_THRESHOLD = 0.1;
var HII_THRESHOLD = 0.08;

var panel = ui.Panel({ style: { position: "top-left", padding: "8px" } });
panel.add(ui.Label("Thresholds"));

function makeNumberRow(label, value, onChange) {
  var lab = ui.Label(label, { width: "180px" });

  var box = ui.Textbox({
    value: value.toString(),
    onChange: function (v) {
      v = parseFloat(v);
      onChange(v);
    },
    style: { width: "80px" }
  });

  var row = ui.Panel([lab, box], ui.Panel.Layout.Flow("horizontal"));
  panel.add(row);

  return box;
}

ui.root.insert(0, panel);

var years = ee.List.sequence(START_YEAR, END_YEAR);

function annualBinaryFromCollection(ic, perYearToBinaryOp) {
  return ee.ImageCollection.fromImages(
    years.map(function (y) {
      y = ee.Number(y);
      var img = ic
        .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y.add(1), 1, 1))
        .first();
      return perYearToBinaryOp(img).rename("g").set("year", y);
    })
  ).sort("year");
}

function annualBinaryFromCollectionReduced(ic, perYearToBinaryOp) {
  return ee.ImageCollection.fromImages(
    years.map(function (y) {
      y = ee.Number(y);
      var annualIc = ic.filterDate(
        ee.Date.fromYMD(y, 1, 1),
        ee.Date.fromYMD(y.add(1), 1, 1)
      );
      return perYearToBinaryOp(annualIc).rename("g").set("year", y);
    })
  ).sort("year");
}

function noTwoConsecutiveZeros(annualBinaryIC) {
  var list = annualBinaryIC.toList(annualBinaryIC.size());
  var pairOk = ee.ImageCollection.fromImages(
    ee.List.sequence(0, ee.Number(list.size()).subtract(2)).map(function (i) {
      i = ee.Number(i);
      var a = ee.Image(list.get(i)).select("g");
      var b = ee.Image(list.get(i.add(1))).select("g");
      return a.or(b).rename("pair_ok");
    })
  ).reduce(ee.Reducer.min());
  return pairOk.eq(1);
}

var rerunDebounce;
var rerun = ui.util.debounce(function () {
  buildLayers();
}, 200);

function buildLayers() {
  Map.layers().reset();
  var annualGrassClassBinary = annualBinaryFromCollection(
    GRASSLAND_CLASS_IC,
    function (img) {
      return img.select(0).eq(GRASSLAND_CLASS_ID);
    }
  );

  var categoricalStabilityMask = noTwoConsecutiveZeros(
    annualGrassClassBinary
  ).rename("categorical_grass_stability_" + START_YEAR + "_" + END_YEAR);

  Map.addLayer(
    categoricalStabilityMask.selfMask(),
    { min: 0, max: 1, palette: ["00ff00"] },
    "Categorical grass stability " + START_YEAR + "-" + END_YEAR,
    true
  );

  var annualGrassProbBinary = annualBinaryFromCollection(
    GRASSLAND_PROB_IC,
    function (img) {
      return img.select(0).gte(GRASSLAND_PROB_THRESHOLD);
    }
  );

  var probabilityStabilityMask = noTwoConsecutiveZeros(
    annualGrassProbBinary
  ).rename("probability_grass_stability_" + START_YEAR + "_" + END_YEAR);

  Map.addLayer(
    probabilityStabilityMask.selfMask(),
    { min: 0, max: 1, palette: ["00aa00"] },
    "Probability (" +
      GRASSLAND_PROB_THRESHOLD +
      "%) grass stability " +
      START_YEAR +
      "-" +
      END_YEAR,
    true
  );

  var annualLowHiiBinary = annualBinaryFromCollectionReduced(
    HII_IC,
    function (annualIc) {
      // mean just makes it go to an image, the input is an image coll
      return annualIc.mean().divide(7000).lt(HII_THRESHOLD);
    }
  );

  var lowHiiStabilityMask = noTwoConsecutiveZeros(annualLowHiiBinary).rename(
    "low_hii_stability_" + START_YEAR + "_" + END_YEAR
  );

  Map.addLayer(
    lowHiiStabilityMask.selfMask(),
    { min: 0, max: 1, palette: ["ff0000"] },
    "HII < " + HII_THRESHOLD + " stability " + START_YEAR + "-" + END_YEAR,
    true
  );

  var lowHmiMask = HMI_IMG.lte(HMI_THRESHOLD);

  Map.addLayer(
    lowHmiMask.selfMask(),
    { min: 0, max: 1, palette: ["aa00aa"] },
    "HMI < " + HMI_THRESHOLD,
    true
  );

  var probabilityIntegrityIndex = probabilityStabilityMask
    .and(lowHiiStabilityMask)
    .and(lowHmiMask)
    .rename("probability_integrity_index");

  Map.addLayer(
    probabilityIntegrityIndex.selfMask(),
    { min: 0, max: 1, palette: ["0000aa"] },
    "Probability Integrity Reference",
    true
  );

  var categoricalIntegrityIndex = categoricalStabilityMask
    .and(lowHiiStabilityMask)
    .and(lowHmiMask)
    .rename("categorical_integrity_index");

  Map.addLayer(
    categoricalIntegrityIndex.selfMask(),
    { min: 0, max: 1, palette: ["0000ff"] },
    "Categorical Integrity Reference",
    true
  );
  Map.setControlVisibility({
    layerList: true
  });
}

makeNumberRow(
  "Grassland prob (0-100%) >=",
  GRASSLAND_PROB_THRESHOLD,
  function (v) {
    GRASSLAND_PROB_THRESHOLD = v;
    rerun();
  }
);

makeNumberRow("HMI (0-1) <=", HMI_THRESHOLD, function (v) {
  HMI_THRESHOLD = v;
  rerun();
});

makeNumberRow("HII (0-1) <", HII_THRESHOLD, function (v) {
  HII_THRESHOLD = v;
  rerun();
});

buildLayers();
