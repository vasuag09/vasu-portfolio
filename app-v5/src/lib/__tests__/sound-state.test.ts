import { describe, expect, it } from "vitest";

import {
  resolveSoundPreference,
  shouldShowSoundGate,
  soundEventsForChange,
} from "../sound-state";

describe("resolveSoundPreference", () => {
  it("maps stored values to on/off", () => {
    expect(resolveSoundPreference("1")).toBe("on");
    expect(resolveSoundPreference("0")).toBe("off");
  });

  it("is unset for missing or corrupted storage", () => {
    expect(resolveSoundPreference(null)).toBe("unset");
    expect(resolveSoundPreference("yes")).toBe("unset");
    expect(resolveSoundPreference("")).toBe("unset");
  });
});

describe("shouldShowSoundGate", () => {
  it("shows on a first visit with motion allowed and no stored choice", () => {
    expect(
      shouldShowSoundGate({ bootPlaying: true, preference: "unset" }),
    ).toBe(true);
  });

  it("never shows when the boot overlay is not playing (return visit / reduced motion)", () => {
    expect(
      shouldShowSoundGate({ bootPlaying: false, preference: "unset" }),
    ).toBe(false);
  });

  it("never shows once a choice exists", () => {
    expect(shouldShowSoundGate({ bootPlaying: true, preference: "on" })).toBe(false);
    expect(shouldShowSoundGate({ bootPlaying: true, preference: "off" })).toBe(false);
  });
});

describe("soundEventsForChange", () => {
  it("emits open when a case study opens", () => {
    expect(
      soundEventsForChange(
        { selectedProjectId: null, synapseOpen: false },
        { selectedProjectId: "nm-gpt", synapseOpen: false },
      ),
    ).toEqual(["open"]);
  });

  it("emits close when the terminal closes", () => {
    expect(
      soundEventsForChange(
        { selectedProjectId: null, synapseOpen: true },
        { selectedProjectId: null, synapseOpen: false },
      ),
    ).toEqual(["close"]);
  });

  it("emits open for a project switch (panel content replaced)", () => {
    expect(
      soundEventsForChange(
        { selectedProjectId: "nm-gpt", synapseOpen: false },
        { selectedProjectId: "fundlymart", synapseOpen: false },
      ),
    ).toEqual(["open"]);
  });

  it("emits nothing when overlay state is unchanged", () => {
    expect(
      soundEventsForChange(
        { selectedProjectId: "nm-gpt", synapseOpen: false },
        { selectedProjectId: "nm-gpt", synapseOpen: false },
      ),
    ).toEqual([]);
  });
});
