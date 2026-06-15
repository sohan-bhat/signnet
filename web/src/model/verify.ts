import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { predict } from "./forward.ts";
import type { ModelWeights } from "./weights";
import type { WeightManifest } from "./weights";

const here = dirname(fileURLToPath(import.meta.url));
const WEB = join(here, "..", "..");
const PUBLIC = join(WEB, "public");
const FIXTURES = join(WEB, "verify", "fixtures.json");

interface Fixture {
  file: string;
  classId: number;
  input32: number[];
  logits: number[];
  probs: number[];
  pred: number;
}

const TOL = 2e-3;

function loadWeightsFromDisk(): ModelWeights {
  const manifest = JSON.parse(
    readFileSync(join(PUBLIC, "weights.json"), "utf-8")
  ) as WeightManifest;
  const buf = readFileSync(join(PUBLIC, "weights.bin"));
  const blob = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const get = (key: keyof ModelWeights) => {
    const m = manifest.params[key];
    return blob.subarray(m.offset, m.offset + m.count);
  };
  return {
    w0: get("w0"), b0: get("b0"),
    w1: get("w1"), b1: get("b1"),
    w2: get("w2"), b2: get("b2"),
    w3: get("w3"), b3: get("b3"),
  };
}

function main() {
  const weights = loadWeightsFromDisk();
  const fixtures = JSON.parse(readFileSync(FIXTURES, "utf-8")) as Fixture[];

  console.log(`Verifying ${fixtures.length} fixtures (tolerance ${TOL} on probs)\n`);
  let allPass = true;

  for (const fx of fixtures) {
    const input = new Float32Array(fx.input32);
    const { probs, pred } = predict(input, weights);

    let maxProbDiff = 0;
    for (let i = 0; i < probs.length; i++) {
      maxProbDiff = Math.max(maxProbDiff, Math.abs(probs[i] - fx.probs[i]));
    }
    const predMatch = pred === fx.pred;
    const probMatch = maxProbDiff <= TOL;
    const pass = predMatch && probMatch;
    allPass = allPass && pass;

    console.log(
      `${pass ? "PASS" : "FAIL"}  ${fx.file.padEnd(8)} ` +
        `py_pred=${fx.pred} js_pred=${pred} ` +
        `js_p=${probs[pred].toFixed(4)} py_p=${fx.probs[fx.pred].toFixed(4)} ` +
        `maxProbDiff=${maxProbDiff.toExponential(2)}`
    );
  }

  console.log(`\n${allPass ? "✓ ALL FIXTURES PASSED" : "✗ SOME FIXTURES FAILED"}`);
  process.exit(allPass ? 0 : 1);
}

main();
