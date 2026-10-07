import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertCircle, Download, FilePlus2, Loader2 } from "lucide-react";
import {
  apiErrorMessage, extractWaveform, extractWaveformRecord, isWfdbFile,
  looksLikeWaveform,
} from "../../api/client";
import { PATIENT_FILE_TEMPLATE, parsePatientFile } from "../../data/patientFile";
import { useRoster } from "../../store/useRoster";

/**
 * Drop-zone card in the roster grid: a clinician uploads a patient file, it is
 * parsed and validated against the API's field bounds, saved to the database, and the
 * patient joins the roster ready for a recommendation. The id comes back from the
 * server, so the page it navigates to is the stored patient.
 *
 * TRACK B: a raw waveform recording can be dropped here too, either on its own or
 * together with the patient file. Track B needs 6 waveform features, and until now
 * the only way to supply them was to compute HRV, perfusion index and the rest by
 * hand and type them in — which is not something a clinician with a recording can
 * do. Dropping the recording now sends it to `/api/waveform/extract`, which runs
 * the same extractors the training features came from, and attaches the result to
 * the patient so Track B becomes selectable.
 *
 * Files are classified by CONTENT, not extension: both tiers use `.txt`, so a
 * waveform is recognised by its `fs:`/`channels:`/`signal:` header or by being
 * rows of samples.
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
    try {
      // Sort the dropped files by what they are, so the clinician can drop a
      // patient file and a recording together in either order. A WFDB record is
      // recognised by extension because its .dat is binary — reading it as text
      // would produce mojibake, not a classification.
      const picked = Array.from(files);
      const recordFiles = picked.filter((f) => isWfdbFile(f.name));
      const textFiles = picked.filter((f) => !isWfdbFile(f.name));
      const read = await Promise.all(
        textFiles.map(async (f) => ({ file: f, text: await f.text() }))
      );
      const waveFiles = read.filter((r) => looksLikeWaveform(r.text));
      const chartFiles = read.filter((r) => !looksLikeWaveform(r.text));

      if (chartFiles.length === 0) {
        setErrors([
          (recordFiles.length > 0
            ? "This looks like a WFDB record folder, "
            : "This looks like a waveform recording, ") +
            "but a patient needs the 12 clinical values too. Drop the patient " +
            "file alongside it (you can select both at once).",
        ]);
        return;
      }
      if (chartFiles.length > 1) {
        setErrors(["Drop one patient file at a time (plus an optional waveform)."]);
        return;
      }

      const { file, text } = chartFiles[0];
      const { patient, errors, warnings } = parsePatientFile(text, file.name);
      const notes = [...warnings];
      if (!patient) {
        setErrors(errors);
        setWarnings(notes);
        return;
      }

      // Extract the waveform server-side and attach it. A recording that yields
      // only some of the 6 is still worth attaching — the router imputes the rest
      // and reports reduced coverage — so partial extraction is a warning, not an
      // error, and the patient is still created.
      if (recordFiles.length > 0 || waveFiles.length > 0) {
        try {
          // A record folder is preferred when both are present: it is the
          // dataset's own format, with no transcription step to lose precision.
          const res = recordFiles.length > 0
            ? await extractWaveformRecord(recordFiles)
            : await extractWaveform(waveFiles[0].file);
          patient.waveform = res.features;
          patient.track = "track_b";
          const got = Object.values(res.features).filter((v) => v != null).length;
          notes.push(
            `Waveform: ${res.duration_s.toFixed(0)}s at ${res.fs.toFixed(1)} Hz → ` +
              `${got}/6 features extracted (${Math.round(res.coverage * 100)}% coverage).`
          );
          notes.push(...res.warnings);
        } catch (e: any) {
          notes.push(
            `Could not extract the waveform (${apiErrorMessage(e)}) — the patient was ` +
              "created from the clinical values alone, on Track A."
          );
        }
      }
      setWarnings(notes);

      try {
        const saved = await addPatient(patient);
        navigate(`/patients/${saved.id}`);
      } catch (e: any) {
        // The file was fine — the database was not. Say which, so the clinician
        // doesn't go looking for a formatting problem that isn't there.
        setErrors([`Could not save the patient: ${apiErrorMessage(e)}`]);
      }
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
        accept=".txt,.json,.csv,.hea,.dat,text/plain,application/json"
        multiple
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
        Add a waveform — a <span className="num">.txt</span> export or a WFDB
        record folder (<span className="num">.hea</span> +{" "}
        <span className="num">.dat</span>) — in the same drop for{" "}
        <span className="font-semibold">Track B</span>.
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
