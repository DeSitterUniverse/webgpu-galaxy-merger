import { describe, expect, it } from "vitest";
import { createGalaxyInitialState } from "../../src/galaxyPhysics";
import {
  BARNES_HUT_TEST_CONSTANTS,
  calculateTreeHalfExtent,
  calculateBarnesHutMemoryLayout,
  chooseTreeDepth,
  maximumTreeNodeCount,
} from "../../src/galaxyBarnesHutSolver";

describe("approximate solver mathematics", () => {
  it("sizes the Barnes-Hut root around the initialized system", () => {
    const initial = createGalaxyInitialState({
      textureWidth: 56,
      radius: 35,
      offset: 25,
    });
    let maximumAbsolutePosition = 0;
    for (let index = 0; index < initial.parameters.particleCount; index++) {
      const offset = index * 8;
      maximumAbsolutePosition = Math.max(
        maximumAbsolutePosition,
        Math.abs(initial.state[offset]!),
        Math.abs(initial.state[offset + 1]!),
        Math.abs(initial.state[offset + 2]!),
      );
    }
    const halfExtent = calculateTreeHalfExtent(maximumAbsolutePosition);
    expect(halfExtent).toBeGreaterThan(maximumAbsolutePosition + 7.9);
    expect(halfExtent).toBe(128);
    const leafDimension = 1 << chooseTreeDepth(initial.parameters.particleCount);
    expect(
      2 * halfExtent / leafDimension,
    ).toBe(2);
    expect(BARNES_HUT_TEST_CONSTANTS.nodeStride).toBe(32);
    expect(BARNES_HUT_TEST_CONSTANTS.childStride).toBe(32);
    expect(maximumTreeNodeCount(56 ** 2, 7)).toBe(13_129);
  });

  it("keeps the compact node capacity linear in particle count", () => {
    const particleCount = 256 ** 2;
    const depth = chooseTreeDepth(particleCount);
    const denseDepthEightNodes = (8 ** 9 - 1) / 7;
    expect(depth).toBe(7);
    expect(maximumTreeNodeCount(particleCount, depth)).toBe(168_521);
    expect(maximumTreeNodeCount(particleCount, depth)).toBeLessThan(
      denseDepthEightNodes,
    );
  });

  it("keeps every million-body tree buffer below 256 MiB", () => {
    const memory = calculateBarnesHutMemoryLayout(1024 ** 2);
    expect(memory.depth).toBe(10);
    expect(memory.maximumNodes).toBeLessThan(5 * 1024 ** 2);
    expect(memory.largestBufferBytes).toBeLessThanOrEqual(256 * 2 ** 20);
    expect(memory.totalBytes).toBeLessThan(512 * 2 ** 20);
  });
});
