import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { PatientList } from "./pages/PatientList";
import { PatientDetail } from "./pages/PatientDetail";
import { ModelValidation } from "./pages/ModelValidation";
import { SignIn } from "./pages/SignIn";
import { RequireAuth } from "./components/auth/RequireAuth";
import "./styles/index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        {/* The only page reachable without a session. */}
        <Route path="/signin" element={<SignIn />} />
        {/* Everything below shows patient data, so all of it is behind the gate. */}
        <Route path="/" element={<RequireAuth><PatientList /></RequireAuth>} />
        <Route path="/patients/:patientId" element={<RequireAuth><PatientDetail /></RequireAuth>} />
        <Route path="/validation" element={<RequireAuth><ModelValidation /></RequireAuth>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
