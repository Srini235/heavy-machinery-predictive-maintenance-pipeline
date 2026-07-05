import { useState, useEffect } from "react"
import { AlertTriangle, CheckCircle2, Loader2, Wrench, Sparkles, Activity } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"

const API_URL = "http://localhost:8000/predict"

type Component = { component: string; predicted_class: number; status: string }
type Prediction = {
  components: Component[]
  stability: string
  flagged_component: string | null
  repair_procedure: string | null
  repair_guidance: string | null
  llm_recommendation: string | null
  latency_ms: number
}

const SENSORS = [
  { key: "operating_hours", label: "Operating hours", step: 1 },
  { key: "pressure_mean_bar", label: "Pressure mean (bar)", step: 0.1 },
  { key: "pressure_std_bar", label: "Pressure std (bar)", step: 0.1 },
  { key: "flow_mean_lpm", label: "Flow mean (L/min)", step: 0.1 },
  { key: "oil_temp_mean_c", label: "Oil temp (°C)", step: 0.1 },
  { key: "vibration_rms_mms", label: "Vibration RMS (mm/s)", step: 0.1 },
  { key: "motor_power_kw", label: "Motor power (kW)", step: 0.1 },
  { key: "pump_speed_mean_rpm", label: "Pump speed (rpm)", step: 1 },
  { key: "cooling_efficiency_pct", label: "Cooling efficiency (%)", step: 0.1 },
] as const

const MACHINE_TYPES = ["Excavator", "Telehandler", "Backhoe Loader"]

const PRESETS: Record<string, Record<string, number | string>> = {
  healthy: {
    operating_hours: 300, pressure_mean_bar: 200, pressure_std_bar: 4, flow_mean_lpm: 8.3,
    oil_temp_mean_c: 50, vibration_rms_mms: 1.5, motor_power_kw: 16, pump_speed_mean_rpm: 1445,
    cooling_efficiency_pct: 95, machine_type: "Excavator",
  },
  degraded: {
    operating_hours: 1600, pressure_mean_bar: 150, pressure_std_bar: 12, flow_mean_lpm: 6.2,
    oil_temp_mean_c: 82, vibration_rms_mms: 6.5, motor_power_kw: 22, pump_speed_mean_rpm: 1350,
    cooling_efficiency_pct: 45, machine_type: "Excavator",
  },
}

const COMPONENT_LABELS: Record<string, string> = {
  cooler_condition: "Cooler", valve_condition: "Valve",
  pump_leakage: "Pump leakage", accumulator_pressure: "Accumulator",
}

export default function App() {
  const [values, setValues] = useState<Record<string, number | string>>(PRESETS.degraded)
  const [result, setResult] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const update = (key: string, v: string) =>
    setValues((p) => ({ ...p, [key]: key === "machine_type" ? v : (v === "" ? 0 : Number(v)) }))

  const assess = async () => {
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await fetch(API_URL, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || "Request failed")
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not reach the API. Is the backend running on :8000?")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("demo") === "1") assess()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const unstable = result?.stability === "unstable"

  return (
    <div className="min-h-screen w-full px-4 py-10 flex flex-col items-center">
      <div className="w-full max-w-4xl">
        <div className="mb-8 text-center">
          <div className="inline-flex items-center gap-2 mb-2 text-primary">
            <Wrench className="h-6 w-6" />
            <span className="text-sm font-semibold uppercase tracking-wider">Group 105 · SE4ML</span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Predictive Maintenance — Mobile Hydraulics</h1>
          <p className="text-muted-foreground mt-2">
            Enter per-cycle hydraulic telemetry to assess component health and circuit stability.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {/* Inputs */}
          <Card>
            <CardHeader>
              <CardTitle>Sensor Readings</CardTitle>
              <CardDescription>Per work-cycle machine telemetry</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="machine_type">Machine type</Label>
                <select
                  id="machine_type"
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={values.machine_type as string}
                  onChange={(e) => update("machine_type", e.target.value)}
                >
                  {MACHINE_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {SENSORS.map((s) => (
                  <div key={s.key} className="space-y-1.5">
                    <Label htmlFor={s.key} className="text-xs">{s.label}</Label>
                    <Input id={s.key} type="number" step={s.step}
                      value={values[s.key]} onChange={(e) => update(s.key, e.target.value)} />
                  </div>
                ))}
              </div>
              <div className="flex gap-2 pt-1">
                <Button variant="outline" size="sm" onClick={() => setValues(PRESETS.healthy)}>Healthy preset</Button>
                <Button variant="outline" size="sm" onClick={() => setValues(PRESETS.degraded)}>Degraded preset</Button>
              </div>
              <Button className="w-full mt-1" onClick={assess} disabled={loading}>
                {loading ? <><Loader2 className="h-4 w-4 animate-spin" /> Assessing…</> : "Assess Machine"}
              </Button>
            </CardContent>
          </Card>

          {/* Results */}
          <Card>
            <CardHeader>
              <CardTitle>Assessment</CardTitle>
              <CardDescription>Component conditions · stability · repair guidance</CardDescription>
            </CardHeader>
            <CardContent>
              {error && (
                <div className="rounded-md bg-red-50 text-red-700 text-sm p-3 border border-red-200">{error}</div>
              )}
              {!result && !error && (
                <p className="text-muted-foreground text-sm py-10 text-center">
                  Enter readings and click <strong>Assess Machine</strong>.
                </p>
              )}
              {result && (
                <div className="space-y-5">
                  {/* stability */}
                  <div className="flex items-center gap-3">
                    {unstable ? <AlertTriangle className="h-7 w-7 text-red-600" />
                              : <CheckCircle2 className="h-7 w-7 text-emerald-600" />}
                    <div>
                      <Badge variant={unstable ? "warning" : "healthy"}>
                        Circuit {result.stability.toUpperCase()}
                      </Badge>
                      <p className="text-xs text-muted-foreground mt-1">Real-time stability model</p>
                    </div>
                  </div>

                  {/* component grid */}
                  <div>
                    <p className="text-sm font-medium mb-2 flex items-center gap-2">
                      <Activity className="h-4 w-4" /> Component conditions
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      {result.components.map((c) => (
                        <div key={c.component}
                          className={`rounded-md border p-2.5 text-sm flex items-center justify-between ${
                            c.status === "healthy" ? "bg-emerald-50 border-emerald-200" : "bg-red-50 border-red-200"}`}>
                          <span>{COMPONENT_LABELS[c.component] ?? c.component}</span>
                          <Badge variant={c.status === "healthy" ? "healthy" : "warning"}>
                            {c.status === "healthy" ? "OK" : "Attention"}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* repair guidance */}
                  {result.flagged_component && (
                    <div className="rounded-md bg-secondary p-4 space-y-2">
                      <p className="text-sm font-semibold flex items-center gap-2">
                        <Wrench className="h-4 w-4" /> {result.repair_procedure}
                      </p>
                      {result.llm_recommendation ? (
                        <div>
                          <p className="text-xs font-medium flex items-center gap-1 text-primary mb-1">
                            <Sparkles className="h-3 w-3" /> AI recommendation (RAG + LLM)
                          </p>
                          <p className="text-sm text-muted-foreground leading-relaxed">
                            {result.llm_recommendation}
                          </p>
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {result.repair_guidance}
                        </p>
                      )}
                    </div>
                  )}

                  <p className="text-xs text-muted-foreground text-right">
                    Inference latency: {result.latency_ms} ms
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <p className="text-center text-xs text-muted-foreground mt-8">
          Multi-output Random Forest · real-time stability model · RAG + OpenRouter LLM · secured API · Group 105
        </p>
      </div>
    </div>
  )
}
