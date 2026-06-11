import {
  Database,
  Filter,
  Layers,
  Box,
  Brain,
  Activity,
  Save,
  Globe,
} from "lucide-react";

export const pipelineStages = [
  { label: "Raw", icon: Database },
  { label: "Clean", icon: Filter },
  { label: "Transform", icon: Layers },
  { label: "Batch", icon: Box },
  { label: "Train", icon: Brain },
  { label: "Eval", icon: Activity },
  { label: "Save", icon: Save },
  { label: "Deploy", icon: Globe },
];
