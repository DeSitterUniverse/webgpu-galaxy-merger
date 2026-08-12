import { describe, expect, it } from "vitest";
import {
  choosePowerOfTwoInterval,
  MAX_TIME_INTERVAL,
} from "./galaxyAdaptiveTimesteps";

describe("adaptive block timesteps", () => {
  it("quantizes local acceleration criteria into power-of-two intervals", () => {
    expect(choosePowerOfTwoInterval(100, 1, 0.08)).toBe(1);
    expect(choosePowerOfTwoInterval(1, 2, 0.08)).toBe(2);
    expect(choosePowerOfTwoInterval(0.01, 2, 0.08)).toBe(MAX_TIME_INTERVAL);
  });

  it("never returns an interval outside the synchronization hierarchy", () => {
    for (const acceleration of [1e-8, 0.01, 0.1, 1, 10, 1000]) {
      const interval = choosePowerOfTwoInterval(acceleration, 1.4, 0.08);
      expect(interval).toBeGreaterThanOrEqual(1);
      expect(interval).toBeLessThanOrEqual(MAX_TIME_INTERVAL);
      expect(interval & (interval - 1)).toBe(0);
    }
  });
});
