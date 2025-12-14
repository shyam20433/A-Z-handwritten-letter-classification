// -----------------------------
// NEURAL NETWORK VISUALIZER
// -----------------------------
const nnCanvas = document.getElementById("nnCanvas");
const nnCtx = nnCanvas.getContext("2d");

const LAYERS = [
    { count: 20, x: 100 },   // sampled input
    { count: 16, x: 300 },
    { count: 12, x: 500 },
    { count: 26, x: 750 },   // output A–Z
];

let activations = {
    L0: new Array(20).fill(0),
    L1: new Array(16).fill(0),
    L2: new Array(12).fill(0),
    L3: new Array(26).fill(0),
};

function drawNN() {
    nnCtx.clearRect(0, 0, nnCanvas.width, nnCanvas.height);

    for (let l = 0; l < LAYERS.length; l++) {
        const layer = LAYERS[l];
        const ySpacing = nnCanvas.height / (layer.count + 1);

        for (let n = 0; n < layer.count; n++) {
            let y = (n + 1) * ySpacing;
            let act = activations["L" + l][n];
            let intensity = Math.min(255, Math.floor(act * 255));

            nnCtx.beginPath();
            nnCtx.fillStyle = `rgb(${intensity}, ${50}, ${200 - intensity})`;
            nnCtx.strokeStyle = "#fff";
            nnCtx.lineWidth = 1;

            nnCtx.arc(layer.x, y, 10, 0, Math.PI * 2);
            nnCtx.fill();
            nnCtx.stroke();

            // draw edges to next layer
            if (l < LAYERS.length - 1) {
                const next = LAYERS[l + 1];
                for (let k = 0; k < next.count; k++) {
                    let ny = (k + 1) * (nnCanvas.height / (next.count + 1));
                    nnCtx.strokeStyle = `rgba(255,255,255,${act})`;
                    nnCtx.beginPath();
                    nnCtx.moveTo(layer.x + 10, y);
                    nnCtx.lineTo(next.x - 10, ny);
                    nnCtx.stroke();
                }
            }
        }
    }

    requestAnimationFrame(drawNN);
}

// start drawing
drawNN();

// -----------------------------
// ACTIVATE NETWORK (ANIMATION)
// -----------------------------
function nn_activate(probabilities, predictedLetter) {
    // Random small activations for input layer
    activations.L0 = activations.L0.map(() => Math.random() * 0.6);

    // Hidden layers random simulated forward pass
    activations.L1 = activations.L1.map(() => Math.random());
    activations.L2 = activations.L2.map(() => Math.random());

    // Output layer = EXACT model softmax probabilities
    activations.L3 = probabilities;
}
