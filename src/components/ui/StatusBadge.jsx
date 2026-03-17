import React from "react";
import { Loader } from "lucide-react";

const statusConfig = {
  LIVE: "bg-emerald-500/10 border-emerald-500/50 text-emerald-400",
  RESEARCH: "bg-blue-500/10 border-blue-500/50 text-blue-400",
  CODE: "bg-purple-500/10 border-purple-500/50 text-purple-400",
  BUILDING: "bg-yellow-500/10 border-yellow-500/50 text-yellow-400",
};

export default function StatusBadge({ status }) {
  return (
    <span
      className={`text-[10px] font-mono px-2 py-1 rounded border flex items-center gap-2 shrink-0 ${statusConfig[status] || statusConfig.CODE}`}
    >
      {status === "BUILDING" && <Loader size={10} className="animate-spin" />}
      {status}
    </span>
  );
}
