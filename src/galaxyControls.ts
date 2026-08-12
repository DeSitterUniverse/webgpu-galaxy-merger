import { MIN_LIVE_HALO_TEXTURE_WIDTH } from "./galaxyPhysics";
import {
  GALAXY_SOLVERS,
  type GalaxySolverKind,
} from "./galaxySolver";

// 1. Tell TypeScript that the custom variable exists on the Window object
declare global {
  interface Window {
    galaxyIsPaused?: boolean;
  }
}

// 2. Declare state on the global window object so it survives any page navigation
if (typeof window !== "undefined" && typeof window.galaxyIsPaused === "undefined") {
  window.galaxyIsPaused = false;
}

const renderPauseState = () => {
  const iconPause = document.getElementById("icon-pause");
  const iconPlay = document.getElementById("icon-play");
  iconPause?.classList.toggle("hidden", Boolean(window.galaxyIsPaused));
  iconPlay?.classList.toggle("hidden", !window.galaxyIsPaused);
};

const setOutput = (id: string, value: string) => {
  const output = document.getElementById(id);
  if (output) output.textContent = value;
};

const clampNumber = (
  value: number,
  minimum: number,
  maximum: number,
  fallback: number,
) => Math.min(maximum, Math.max(minimum, Number.isFinite(value) ? value : fallback));

const onGalaxyStatus = (event: Event) => {
  const detail = (
    event as CustomEvent<{
      state: "initializing" | "ready" | "unsupported" | "error";
      message: string;
    }>
  ).detail;
  setOutput("galaxy-backend-status", detail.message);
  const output = document.getElementById("galaxy-backend-status");
  if (output) output.hidden = detail.state === "ready";
};

const onGalaxyInitialized = (event: Event) => {
  const detail = (
    event as CustomEvent<{
      initialCoreSeparation: number;
    }>
  ).detail;
  setOutput("galaxy-time-value", "0.00");
  setOutput(
    "galaxy-core-separation-value",
    detail.initialCoreSeparation.toFixed(2),
  );
};

const onGalaxyLiveStats = (event: Event) => {
  const detail = (
    event as CustomEvent<{
      simulationTime?: number;
      coreSeparation?: number;
      binary?: boolean;
      merged?: boolean;
    }>
  ).detail;
  if (detail.simulationTime !== undefined) {
    setOutput("galaxy-time-value", detail.simulationTime.toFixed(2));
  }
  if (detail.merged !== undefined || detail.coreSeparation !== undefined) {
    setOutput(
      "galaxy-core-separation-value",
      detail.merged
        ? "Merged"
        : detail.binary
          ? "Bound Binary"
        : (detail.coreSeparation ?? 0).toFixed(2),
    );
  }
};

export const setGalaxyPaused = (paused: boolean) => {
  window.galaxyIsPaused = paused;
  renderPauseState();
  window.dispatchEvent(
    new CustomEvent("galaxy:pause", { detail: { paused } }),
  );
};

// 2. Define the function that wires up the buttons and slider
export const initGalaxyControls = () => {
  if (typeof document === "undefined") return;

  window.removeEventListener("galaxy:status", onGalaxyStatus);
  window.removeEventListener("galaxy:initialized", onGalaxyInitialized);
  window.removeEventListener("galaxy:live-stats", onGalaxyLiveStats);
  window.addEventListener("galaxy:status", onGalaxyStatus);
  window.addEventListener("galaxy:initialized", onGalaxyInitialized);
  window.addEventListener("galaxy:live-stats", onGalaxyLiveStats);

  const btn = document.getElementById("galaxy-pause-btn");
  const slider = document.getElementById(
    "galaxy-stars-slider"
  ) as HTMLInputElement;
  const sliderVal = document.getElementById("galaxy-stars-value");
  const solverButtons = Array.from(
    document.querySelectorAll<HTMLButtonElement>("[data-galaxy-solver]"),
  );

  // PAUSE BUTTON LOGIC
  // Check for 'data-loaded' to prevent attaching the click event twice
  if (btn && !btn.hasAttribute("data-loaded")) {
    btn.setAttribute("data-loaded", "true");

    renderPauseState();

    btn.addEventListener("click", () => {
      setGalaxyPaused(!window.galaxyIsPaused);
    });
  }

  // SLIDER LOGIC
  if (slider && !slider.hasAttribute("data-loaded")) {
    slider.setAttribute("data-loaded", "true");

    // MOBILE SYNC: Update the UI to match the engine's mobile default
    const isMobile = window.innerWidth < 768;
    const initialWidth = isMobile ? MIN_LIVE_HALO_TEXTURE_WIDTH : 72;

    slider.value = initialWidth.toString();
    if (sliderVal) sliderVal.textContent = `${initialWidth}x${initialWidth}`;

    // 'input' fires instantly while dragging (updates the text)
    slider.addEventListener("input", (e) => {
      const val = (e.target as HTMLInputElement).value;
      if (sliderVal) sliderVal.textContent = `${val}x${val}`;
    });

    // 'change' fires ONLY when the mouse is released (sends data to the GPU)
    slider.addEventListener("change", (e) => {
      const val = clampNumber(
        parseInt((e.target as HTMLInputElement).value, 10),
        MIN_LIVE_HALO_TEXTURE_WIDTH,
        Number(slider.max),
        initialWidth,
      );
      window.dispatchEvent(
        new CustomEvent("galaxy:resize", { detail: { textureWidth: val } })
      );
    });
  }

  for (const solverButton of solverButtons) {
    if (solverButton.dataset.loaded) continue;
    solverButton.dataset.loaded = "true";
    solverButton.addEventListener("click", () => {
      const solver = solverButton.dataset.galaxySolver as GalaxySolverKind;
      const definition = GALAXY_SOLVERS[solver];
      if (!definition) return;
      for (const button of solverButtons) {
        button.setAttribute(
          "aria-pressed",
          String(button === solverButton),
        );
      }
      slider.max = String(definition.maxTextureWidth);
      const width = Math.min(Number(slider.value), definition.maxTextureWidth);
      slider.value = String(width);
      if (sliderVal) sliderVal.textContent = `${width}x${width}`;
      window.dispatchEvent(
        new CustomEvent("galaxy:solver", { detail: { solver } }),
      );
    });
  }

  // RADIUS SLIDER LOGIC
  const radiusSlider = document.getElementById(
    "galaxy-radius-slider"
  ) as HTMLInputElement;
  const radiusVal = document.getElementById("galaxy-radius-value");

  if (radiusSlider && !radiusSlider.hasAttribute("data-loaded")) {
    radiusSlider.setAttribute("data-loaded", "true");
    radiusSlider.addEventListener("input", (e) => {
      if (radiusVal)
        radiusVal.textContent = (e.target as HTMLInputElement).value;
    });
    radiusSlider.addEventListener("change", (e) => {
      const val = clampNumber(
        parseInt((e.target as HTMLInputElement).value, 10),
        15,
        80,
        35,
      );
      window.dispatchEvent(
        new CustomEvent("galaxy:radius", { detail: { radius: val } })
      );
    });
  }

  // OFFSET SLIDER LOGIC
  const offsetSlider = document.getElementById(
    "galaxy-offset-slider"
  ) as HTMLInputElement;
  const offsetVal = document.getElementById("galaxy-offset-value");

  if (offsetSlider && !offsetSlider.hasAttribute("data-loaded")) {
    offsetSlider.setAttribute("data-loaded", "true");
    offsetSlider.addEventListener("input", (e) => {
      if (offsetVal)
        offsetVal.textContent = (e.target as HTMLInputElement).value;
    });
    offsetSlider.addEventListener("change", (e) => {
      const val = clampNumber(
        parseInt((e.target as HTMLInputElement).value, 10),
        10,
        60,
        25,
      );
      window.dispatchEvent(
        new CustomEvent("galaxy:offset", { detail: { offset: val } })
      );
    });
  }
};
