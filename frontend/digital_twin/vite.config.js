import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

const cesiumSource = "node_modules/cesium/Build/Cesium";
const cesiumBaseUrl = "cesiumStatic";

export default defineConfig({
    define: {
        CESIUM_BASE_URL: JSON.stringify(`/${cesiumBaseUrl}`),
    },

    plugins: [
        viteStaticCopy({
            targets: [
                {
                    src: `${cesiumSource}/ThirdParty`,
                    dest: `${cesiumBaseUrl}/ThirdParty`,
                },
                {
                    src: `${cesiumSource}/Workers`,
                    dest: `${cesiumBaseUrl}/Workers`,
                },
                {
                    src: `${cesiumSource}/Assets`,
                    dest: `${cesiumBaseUrl}/Assets`,
                },
                {
                    src: `${cesiumSource}/Widgets`,
                    dest: `${cesiumBaseUrl}/Widgets`,
                },
            ],
        }),
    ],
});