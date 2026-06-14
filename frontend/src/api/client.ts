import axios from "axios";
import type { RecommendationRequest, RecommendationResponse } from "../types/recommendation";

const api = axios.create({ baseURL: "/api" });

export async function getRecommendation(
  req: RecommendationRequest
): Promise<RecommendationResponse> {
  const { data } = await api.post<RecommendationResponse>("/recommend", req);
  return data;
}

export interface TrackMeta {
  id: string; label: string; available: boolean; waveform_required: boolean;
}
export async function getTracks(): Promise<TrackMeta[]> {
  const { data } = await api.get<{ tracks: TrackMeta[] }>("/tracks");
  return data.tracks;
}
