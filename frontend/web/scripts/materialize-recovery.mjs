import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../../..");
const chunkDir = path.join(repoRoot, "recovery", "source-b64");
const outputDir = path.join(repoRoot, "frontend", "web", "src", "recovery");

const files = [
  {
    output: "GoldMIND-v11.2-engine.jsx",
    parts: Array.from({ length: 7 }, (_, i) => `GoldMIND_v11_2_engine.${String(i).padStart(2, "0")}.b64part`),
    sha256: "9dcd54b67d0aab1eece47a2943b9ac2523bfb1577437dc8f50d583b18fc3f17e",
  },
  {
    output: "GoldMIND-v11.2.jsx",
    parts: ["GoldMIND_v11_2.00.b64part"],
    sha256: "d1426c9cd4738266397c5514872161780f54b03b03d1e433696065cd41c7f54f",
  },
  {
    output: "GoldMIND-v11_2-preview.jsx",
    parts: ["GoldMIND_v11_2_preview.00.b64part"],
    sha256: "daade7896c73f63a37d37158a6922402d479ef2403374221827130ff224de116",
  },
  {
    output: "GoldMIND-Complete.jsx",
    parts: ["GoldMIND_Complete.00.b64part", "GoldMIND_Complete.01.b64part"],
    sha256: "a9e98eb12fdccb6d0b4dc4510f85f078bef18abdc8fa3291784fdafa860128f1",
  },
];

fs.mkdirSync(outputDir, { recursive: true });

for (const spec of files) {
  const encoded = spec.parts
    .map((part) => fs.readFileSync(path.join(chunkDir, part), "utf8").trim())
    .join("");

  const compressed = Buffer.from(encoded, "base64");
  const source = zlib.gunzipSync(compressed);
  const actualSha = crypto.createHash("sha256").update(source).digest("hex");

  if (actualSha !== spec.sha256) {
    throw new Error(`${spec.output}: SHA256 mismatch. Expected ${spec.sha256}, got ${actualSha}`);
  }

  const outputPath = path.join(outputDir, spec.output);
  fs.writeFileSync(outputPath, source);
  console.log(`materialized ${path.relative(repoRoot, outputPath)} (${actualSha})`);
}
