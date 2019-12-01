const z0 = document.getElementById('z0');
const z1 = document.getElementById('z1');

const slide0 = document.getElementById('slide0');
const slide1 = document.getElementById('slide1');

slide0.onchange = function() { z0.value = this.value; run(); }
slide1.onchange = function() { z1.value = this.value; run(); }

z0.onchange = function() { slide0.value = this.value; run(); }
z1.onchange = function() { slide1.value = this.value; run(); }

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const session = new onnx.InferenceSession();
loaded = session.loadModel("./vae-dec.onnx");

async function run() {
    const z = new Float32Array(1 * 2);
    z[0] = document.getElementById('z0').value;
    z[1] = document.getElementById('z1').value;
    const z_tensor = new onnx.Tensor(z, 'float32', [1, 2]);

    const output_map = await session.run([z_tensor]);
    const y = output_map.get('12');

    y_uint8 = Uint8ClampedArray.from(y.data);
    const img_arr = new Uint8ClampedArray(4 * 784);
    for (let i = 0; i < y_uint8.length; i += 1) {
        img_arr[i * 4 + 0] = y_uint8[i];
        img_arr[i * 4 + 1] = y_uint8[i];
        img_arr[i * 4 + 2] = y_uint8[i];
        img_arr[i * 4 + 3] = 255;
    }

    img_data = new ImageData(img_arr, 28);
    ctx.putImageData(img_data, 0, 0);
}

loaded.then(run);
