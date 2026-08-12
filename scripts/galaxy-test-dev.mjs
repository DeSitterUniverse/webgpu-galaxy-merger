import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const astroCli = fileURLToPath(
  new URL("../node_modules/astro/bin/astro.mjs", import.meta.url),
);
const child = spawn(
  process.execPath,
  [astroCli, "dev", ...process.argv.slice(2)],
  {
    stdio: "inherit",
    env: {
      ...process.env,
      PUBLIC_GALAXY_TEST_MODE: "true",
    },
  },
);

child.once("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exitCode = code ?? 1;
});
