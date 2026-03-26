import {
  Layout,
  Code2,
  Cpu,
  FlaskConical,
  User,
  BookOpen,
} from "lucide-react";

/**
 * Single source of truth for all navigation data.
 * Used by: LayerNav, MobileNav, Header, useKeyboardShortcuts, layers util.
 */
export const NAVIGATION_ITEMS = [
  {
    path: "/",
    label: "Overview",
    systemLabel: "System Overview",
    shortLabel: "IN",
    layerTag: "INPUT LAYER",
    layerTitle: "System Overview",
    icon: Layout,
    shortcut: "1",
  },
  {
    path: "/projects",
    label: "Projects",
    systemLabel: "Deployments",
    shortLabel: "H1",
    layerTag: "HIDDEN LAYER 1",
    layerTitle: "Trained Models",
    icon: Code2,
    shortcut: "2",
  },
  {
    path: "/skills",
    label: "Skills",
    systemLabel: "Tech Stack",
    shortLabel: "H2",
    layerTag: "HIDDEN LAYER 2",
    layerTitle: "Weights & Biases",
    icon: Cpu,
    shortcut: "3",
  },
  {
    path: "/research",
    label: "Research",
    systemLabel: "Research Lab",
    shortLabel: "H3",
    layerTag: "HIDDEN LAYER 3",
    layerTitle: "Research Lab",
    icon: FlaskConical,
    shortcut: "4",
  },
  {
    path: "/about",
    label: "About",
    systemLabel: "About Me",
    shortLabel: "OUT",
    layerTag: "OUTPUT LAYER",
    layerTitle: "About",
    icon: User,
    shortcut: "5",
  },
  {
    path: "/blog",
    label: "Blog",
    systemLabel: "Engineering Logs",
    shortLabel: "SIG",
    layerTag: "SIGNAL LAYER",
    layerTitle: "Signal Propagation",
    icon: BookOpen,
    shortcut: "6",
  },
];

/**
 * Maps a pathname to the corresponding navigation item index.
 */
export function getActiveLayer(pathname) {
  const idx = NAVIGATION_ITEMS.findIndex((item) => {
    if (item.path === "/") return pathname === "/";
    return pathname.startsWith(item.path);
  });
  return idx >= 0 ? idx : -1;
}

/**
 * Returns the navigation item matching the current pathname.
 */
export const getActiveItem = (pathname) => {
  return (
    NAVIGATION_ITEMS.find((item) => {
      if (item.path === "/") return pathname === "/";
      return pathname.startsWith(item.path);
    }) || NAVIGATION_ITEMS[0]
  );
};

/**
 * Build a routes map from shortcuts to paths (for keyboard navigation).
 */
export const SHORTCUT_ROUTES = Object.fromEntries(
  NAVIGATION_ITEMS.map((item) => [item.shortcut, item.path]),
);
