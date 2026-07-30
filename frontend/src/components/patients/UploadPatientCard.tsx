import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertCircle, Download, FilePlus2, Loader2 } from "lucide-react";
import { PATIENT_FILE_TEMPLATE, parsePatientFile } from "../../data/patientFile";
import { useRoster } from "../../store/useRoster";

/**
 * Drop-zone card in the roster grid: a clinician uploads a patient file, it is
 * parsed and validated against the API's field bounds, and the patient joins the
 * roster ready for a recommendation.
 */
export function UploadPatientCard() {
  const addPatient = useRoster((s) => s.addPatient);
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);

  async function handleFiles(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setErrors([]);
    setWarnings([]);
    const file = files[0];
    try {
      const { patient, errors, warnings } = parsePatientFile(await file.text(), file.name);
      setWarnings(warnings);
      if (!patient) {
        setErrors(errors);
        return;
      }
      addPatient(patient);
      navigate(`/patients/${patient.id}`);
    } catch (e: any) {
      setErrors([`Could not read the file: ${e?.message ?? e}`]);
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  function downloadTemplate(e: React.MouseEvent) {
    e.stopPropagation();
    const url = URL.createObjectURL(
      new Blob([PATIENT_FILE_TEMPLATE], { type: "text/plain" })
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "ventassist-patient-template.txt";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
      className={`group flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-5 text-center transition ${
        dragging
          ? "border-cyan-500 bg-cyan-50/70"
          : "border-slate-300 bg-white/40 hover:border-cyan-400 hover:bg-white/70"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".txt,.json,.csv,text/plain,application/json"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />

      {busy ? (
        <Loader2 className="animate-spin text-cyan-600" size={22} />
      ) : (
        <div className="grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-cyan-100 to-indigo-100 text-cyan-700 ring-1 ring-inset ring-slate-900/5 transition group-hover:scale-105">
          <FilePlus2 size={20} />
        </div>
      )}

      <div className="mt-3 font-display text-base font-semibold text-slate-900">
        Upload patient file
      </div>
      <p className="mt-1 max-w-[15rem] text-[12px] leading-relaxed text-slate-500">
        Drop a <span className="num">.txt</span> or <span className="num">.json</span> chart export
        here, or click to browse. The 12 clinical values and weight are required.
      </p>

      <button onClick={downloadTemplate} className="btn-ghost mt-3">
        <Download size={13} /> Template
      </button>

      {(errors.length > 0 || warnings.length > 0) && (
        <div className="mt-3 w-full space-y-1 text-left">
          {errors.map((e, i) => (
            <p key={`e${i}`} className="flex gap-1.5 rounded-lg border border-rose-200 bg-rose-50 px-2 py-1 text-[11px] leading-relaxed text-rose-700">
              <AlertCircle size={12} className="mt-px shrink-0" /> {e}
            </p>
          ))}
          {warnings.map((w, i) => (
            <p key={`w${i}`} className="rounded-lg border border-amber-200 bg-amber-50 px-2 py-1 text-[11px] leading-relaxed text-amber-700">
              {w}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
