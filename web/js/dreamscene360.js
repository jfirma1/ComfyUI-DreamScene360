import { app } from "../../scripts/app.js";

// DreamScene360 - minimal UI extensions
app.registerExtension({
    name: "DreamScene360",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DreamScene360_PanoToPointcloud") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                this.color = "#1a3a2a";
                this.bgcolor = "#0d1f15";
            };
        }
    }
});
