//-----------------------------------------
// 1. CANVAS INITIAL SETUP
//-----------------------------------------
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

//-----------------------------------------
// 2. DRAWING LOGIC
//-----------------------------------------
canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!drawing) return;

    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

//-----------------------------------------
// 3. CLEAR BUTTON
//-----------------------------------------
document.getElementById("clearBtn").onclick = () => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
};

//-----------------------------------------
// 4. PREDICT BUTTON
//-----------------------------------------
document.getElementById("predictBtn").onclick = async () => {

    let img = get28x28();   // Preprocessed 28×28 array

    let response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: img })
    });

    let result = await response.json();

    // Show predicted letter
    document.getElementById("resultLetter").innerText = result.letter;
    document.getElementById("confidence").innerText =
        `Confidence: ${result.confidence.toFixed(2)}%`;

    // Fill probabilities box
    let box = document.getElementById("probabilities");
    box.innerHTML = "";

    let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

    result.probabilities.forEach((p, i) => {
        let div = document.createElement("div");
        div.innerText = `${letters[i]} — ${(p * 100).toFixed(2)}%`;
        box.appendChild(div);
    });

    // 🔥 Trigger NN Visualizer Animation
    nn_activate(result.probabilities, result.letter);
};

//-----------------------------------------
// 5. Convert 300×300 → 28×28 (Kaggle matching)
//-----------------------------------------
function get28x28() {

    // Step 1: Get 300×300 pixels
    let big = ctx.getImageData(0, 0, 300, 300);

    let gray = [];
    for (let i = 0; i < big.data.length; i += 4) {
        gray.push(big.data[i] / 255); // normalize
    }

    // Step 2: Bounding box detection
    let xs = [], ys = [];
    for (let y = 0; y < 300; y++) {
        for (let x = 0; x < 300; x++) {
            if (gray[y * 300 + x] > 0.1) {
                xs.push(x);
                ys.push(y);
            }
        }
    }

    if (xs.length === 0) return new Array(28 * 28).fill(0);

    let minX = Math.min(...xs), maxX = Math.max(...xs);
    let minY = Math.min(...ys), maxY = Math.max(...ys);

    let w = maxX - minX;
    let h = maxY - minY;

    // Step 3: Crop bounding region
    let crop = document.createElement("canvas");
    crop.width = w;
    crop.height = h;
    crop.getContext("2d").putImageData(
        ctx.getImageData(minX, minY, w, h),
        0, 0
    );

    // Step 4: Resize & center → 28×28
    let tmp = document.createElement("canvas");
    tmp.width = 28;
    tmp.height = 28;
    let tctx = tmp.getContext("2d");

    tctx.fillStyle = "black";
    tctx.fillRect(0, 0, 28, 28);

    let size = Math.max(w, h);
    let scale = 20 / size;

    tctx.drawImage(
        crop,
        0, 0, w, h,
        (28 - w * scale) / 2,
        (28 - h * scale) / 2,
        w * scale, h * scale
    );

    // Step 5: Extract final pixels
    let data = tctx.getImageData(0, 0, 28, 28).data;

    let out = [];
    for (let i = 0; i < data.length; i += 4) {
        out.push(data[i] / 255); // grayscale normalized
    }

    return out;
}
