import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";


// =========================================================
// 1. Create Cesium terrain provider
// =========================================================

const terrainProvider =
    await Cesium.createWorldTerrainAsync();


// =========================================================
// 2. Create the Cesium viewer
// =========================================================

const viewer = new Cesium.Viewer(
    "cesiumContainer",
    {
        terrainProvider:
            terrainProvider,

        baseLayerPicker: true,
        timeline: true,
        animation: true,

        geocoder: true,
        homeButton: true,
        navigationHelpButton: true,
        sceneModePicker: true,
        fullscreenButton: true,
    }
);


// =========================================================
// 3. Historical imagery - E.4.3
// =========================================================

const years = [
    2015,
    2018,
    2021,
    2024,
];


// =========================================================
// 4. Sicily / Vulcano LiDAR study-area bounds
// =========================================================

const studyAreaRectangle =
    Cesium.Rectangle.fromDegrees(
        14.957943,   // west
        38.413989,   // south
        14.961042,   // east
        38.416862    // north
    );


// =========================================================
// 5. Store historical imagery layers
// =========================================================

const historicalLayers = {};


// =========================================================
// 6. Create imagery layer for every historical year
// =========================================================

for (const year of years) {

    const provider =
        new Cesium.UrlTemplateImageryProvider(
            {
                url:
                    `/imagery/${year}/{z}/{x}/{y}.png`,

                minimumLevel: 10,

                maximumLevel: 18,

                rectangle:
                    studyAreaRectangle,

                tilingScheme:
                    new Cesium.WebMercatorTilingScheme(),
            }
        );


    const layer =
        viewer.imageryLayers.addImageryProvider(
            provider
        );


    historicalLayers[year] =
        layer;


    // Show 2015 initially.
    layer.show =
        year === 2015;
}


// =========================================================
// 7. Historical imagery year selector
// =========================================================

const yearContainer =
    document.createElement("div");


yearContainer.style.position =
    "absolute";

yearContainer.style.top =
    "10px";

yearContainer.style.left =
    "10px";

yearContainer.style.zIndex =
    "1000";

yearContainer.style.background =
    "rgba(30, 30, 30, 0.85)";

yearContainer.style.padding =
    "10px";

yearContainer.style.borderRadius =
    "6px";

yearContainer.style.color =
    "white";

yearContainer.style.fontFamily =
    "Arial, sans-serif";


// =========================================================
// 8. Selector label
// =========================================================

const yearLabel =
    document.createElement("label");


yearLabel.textContent =
    "Historical imagery: ";


yearLabel.style.marginRight =
    "8px";


yearContainer.appendChild(
    yearLabel
);


// =========================================================
// 9. Year dropdown
// =========================================================

const yearSelect =
    document.createElement("select");


yearSelect.style.padding =
    "4px";


for (const year of years) {

    const option =
        document.createElement("option");


    option.value =
        year;


    option.textContent =
        year;


    yearSelect.appendChild(
        option
    );
}


yearSelect.value =
    "2015";


yearContainer.appendChild(
    yearSelect
);


// Add selector to Cesium.
const cesiumContainer =
    document.getElementById(
        "cesiumContainer"
    );


cesiumContainer.appendChild(
    yearContainer
);


// =========================================================
// 10. Show selected historical year
// =========================================================

function showYear(year) {

    for (const currentYear of years) {

        historicalLayers[currentYear].show =
            Number(currentYear) ===
            Number(year);
    }


    console.log(
        `Historical imagery year: ${year}`
    );
}


// =========================================================
// 11. Handle year selection
// =========================================================

yearSelect.addEventListener(
    "change",
    () => {

        showYear(
            yearSelect.value
        );
    }
);


// =========================================================
// 12. Load LiDAR-derived building footprints
// =========================================================

const buildingDataSource =
    await Cesium.GeoJsonDataSource.load(
        "/lidar_buildings.geojson",
        {
            clampToGround: false,
        }
    );


// =========================================================
// 13. Get building entities
// =========================================================

const buildingEntities =
    buildingDataSource.entities.values;


console.log(
    `Loaded ${buildingEntities.length} LiDAR building polygons.`
);


// =========================================================
// 14. Convert each polygon into a 3D building
// =========================================================

for (const entity of buildingEntities) {

    if (!entity.polygon) {
        continue;
    }


    // -----------------------------------------------------
    // Read LiDAR-derived building height
    // -----------------------------------------------------

    let buildingHeight =
        5.0;


    if (entity.properties) {

        const heightProperty =
            entity.properties.height_m;


        if (heightProperty) {

            const heightValue =
                heightProperty.getValue(
                    Cesium.JulianDate.now()
                );


            if (
                heightValue !== undefined &&
                heightValue !== null &&
                Number.isFinite(
                    Number(heightValue)
                )
            ) {

                buildingHeight =
                    Number(heightValue);
            }
        }
    }


    // -----------------------------------------------------
    // Safety minimum
    // -----------------------------------------------------

    buildingHeight =
        Math.max(
            buildingHeight,
            2.0
        );


    // -----------------------------------------------------
    // Building appearance
    // -----------------------------------------------------

    entity.polygon.material =
        Cesium.Color.SLATEGRAY.withAlpha(
            0.90
        );


    entity.polygon.outline =
        false;


    // -----------------------------------------------------
    // Building base
    // -----------------------------------------------------

    entity.polygon.height =
        0.0;


    entity.polygon.heightReference =
        Cesium.HeightReference.CLAMP_TO_GROUND;


    // -----------------------------------------------------
    // Extrude building
    // -----------------------------------------------------

    entity.polygon.extrudedHeight =
        buildingHeight;


    entity.polygon.extrudedHeightReference =
        Cesium.HeightReference.RELATIVE_TO_GROUND;


    // -----------------------------------------------------
    // Building footprint area
    // -----------------------------------------------------

    let areaText =
        "Not available";


    if (entity.properties) {

        const areaProperty =
            entity.properties.area_m2;


        if (areaProperty) {

            const areaValue =
                areaProperty.getValue(
                    Cesium.JulianDate.now()
                );


            if (
                areaValue !== undefined &&
                areaValue !== null
            ) {

                areaText =
                    `${Number(areaValue).toFixed(2)} m²`;
            }
        }
    }


    // -----------------------------------------------------
    // Building information popup
    // -----------------------------------------------------

    entity.description = `
        <div>
            <h3>LiDAR-derived Building</h3>

            <p>
                <b>Estimated height:</b>
                ${buildingHeight.toFixed(2)} m
            </p>

            <p>
                <b>Estimated footprint:</b>
                ${areaText}
            </p>

            <p>
                <b>Source:</b>
                Phase 3 LiDAR classification
            </p>

            <p>
                <b>Geometry:</b>
                Approximate LiDAR-derived building footprint
            </p>
        </div>
    `;
}


// =========================================================
// 15. Add LiDAR building layer
// =========================================================

viewer.dataSources.add(
    buildingDataSource
);


// =========================================================
// 16. Fly camera to LiDAR buildings initially
// =========================================================

if (buildingEntities.length > 0) {

    await viewer.flyTo(
        buildingDataSource,
        {
            duration: 2.0,

            offset:
                new Cesium.HeadingPitchRange(
                    0.0,

                    Cesium.Math.toRadians(
                        -45.0
                    ),

                    0.0
                ),
        }
    );
}


// =========================================================
// 17. E.5.1 - Poll GeoAI Agent Result
// =========================================================
//
// The FastAPI server exposes:
//
// http://localhost:8000/agent/latest_result
//
// The frontend checks this endpoint every 3 seconds.
// When a new latitude/longitude is available,
// Cesium flies the camera to that location.
// =========================================================

let lastAgentLocation =
    null;


async function pollAgentResult() {

    try {

        const response =
            await fetch(
                "http://localhost:8000/agent/latest_result"
            );


        // -------------------------------------------------
        // Check HTTP response
        // -------------------------------------------------

        if (!response.ok) {

            console.warn(
                "Agent result endpoint returned HTTP status:",
                response.status
            );

            return;
        }


        // -------------------------------------------------
        // Read JSON result
        // -------------------------------------------------

        const data =
            await response.json();


        console.log(
            "Latest agent result:",
            data
        );


        // -------------------------------------------------
        // Ignore empty result
        // -------------------------------------------------

        if (
            data.latitude === undefined ||
            data.longitude === undefined
        ) {

            return;
        }


        // -------------------------------------------------
        // Convert coordinates to numbers
        // -------------------------------------------------

        const latitude =
            Number(data.latitude);


        const longitude =
            Number(data.longitude);


        // -------------------------------------------------
        // Validate coordinates
        // -------------------------------------------------

        if (
            !Number.isFinite(latitude) ||
            !Number.isFinite(longitude)
        ) {

            console.warn(
                "Invalid agent coordinates:",
                data
            );

            return;
        }


        // -------------------------------------------------
        // Prevent repeated camera flights
        // -------------------------------------------------

        const locationKey =
            `${latitude},${longitude}`;


        if (
            lastAgentLocation ===
            locationKey
        ) {

            return;
        }


        lastAgentLocation =
            locationKey;


        // -------------------------------------------------
        // Log new agent result
        // -------------------------------------------------

        console.log(
            "New GeoAI agent location:",
            latitude,
            longitude
        );


        // -------------------------------------------------
        // Fly camera to agent-selected location
        // -------------------------------------------------

        viewer.camera.flyTo(
            {
                destination:
                    Cesium.Cartesian3.fromDegrees(
                        longitude,
                        latitude,
                        800
                    ),

                duration:
                    2.0,
            }
        );


    } catch (error) {

        // The frontend should continue running even if
        // the FastAPI server is temporarily unavailable.

        console.warn(
            "Could not retrieve latest agent result:",
            error
        );
    }
}


// =========================================================
// 18. Start E.5.1 polling
// =========================================================

// Check immediately.
pollAgentResult();


// Then check every 3 seconds.
setInterval(
    pollAgentResult,
    3000
);


// =========================================================
// 19. Export viewer
// =========================================================

export {
    viewer
};