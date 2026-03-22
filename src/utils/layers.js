/**
 * Maps a pathname to the corresponding neural network layer index (0-4).
 */
export function getActiveLayer(pathname) {
  if (pathname === "/") return 0;
  if (pathname.startsWith("/projects")) return 1;
  if (pathname.startsWith("/skills")) return 2;
  if (pathname.startsWith("/research")) return 3;
  if (pathname.startsWith("/about")) return 4;
  return -1;
}
