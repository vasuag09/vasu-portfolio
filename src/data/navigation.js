import {
  Layout,
  Code2,
  Cpu,
  BookOpen,
  User,
} from "lucide-react";

export const NAVIGATION_ITEMS = [
  {
    path: "/",
    label: "Overview",
    systemLabel: "System Overview",
    icon: Layout,
    shortcut: "1",
  },
  {
    path: "/projects",
    label: "Projects",
    systemLabel: "Deployments",
    icon: Code2,
    shortcut: "2",
  },
  {
    path: "/skills",
    label: "Skills",
    systemLabel: "Tech Stack",
    icon: Cpu,
    shortcut: "3",
  },
  {
    path: "/blog",
    label: "Blog",
    systemLabel: "Engineering Log",
    icon: BookOpen,
    shortcut: "4",
  },
  {
    path: "/about",
    label: "About",
    systemLabel: "About Me",
    icon: User,
    shortcut: "5",
  },
];

export const getActiveItem = (pathname) => {
  return (
    NAVIGATION_ITEMS.find((item) => {
      if (item.path === "/") return pathname === "/";
      return pathname.startsWith(item.path);
    }) || NAVIGATION_ITEMS[0]
  );
};
