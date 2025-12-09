import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import datetime
import os
import io
import tempfile

# ---------------------------
# Configuración general
# ---------------------------
DATA_FILE = "registro_telefonos.csv"

if not os.path.exists(DATA_FILE):
    df_init = pd.DataFrame(columns=["Marca", "Modelo", "Fallo", "Solucion", "Fecha"])
    df_init.to_csv(DATA_FILE, index=False)

# ---------------------------
# Funciones de datos (cacheadas)
# ---------------------------
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv(DATA_FILE)
        for col in ["Marca", "Modelo", "Fallo", "Solucion", "Fecha"]:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("")
        return df
    except Exception:
        return pd.DataFrame(columns=["Marca", "Modelo", "Fallo", "Solucion", "Fecha"])

def guardar_datos(df):
    df.to_csv(DATA_FILE, index=False)
    cargar_datos.clear()

# Importación avanzada
def importar_archivo(uploaded_file, dedupe=True):
    if not uploaded_file:
        return None, "No file"
    name = uploaded_file.name.lower()
    try:
        if name.endswith('.csv'):
            df_new = pd.read_csv(uploaded_file)
        elif name.endswith('.xls') or name.endswith('.xlsx'):
            df_new = pd.read_excel(uploaded_file)
        else:
            return None, "Formato no soportado"
        expected = ["Marca","Modelo","Fallo","Solucion","Fecha"]
        df_new.rename(columns={c: c.strip() for c in df_new.columns}, inplace=True)
        missing = [c for c in expected if c not in df_new.columns]
        if missing:
            return None, f"Faltan columnas: {', '.join(missing)}"
        df_new = df_new[expected].astype(str)
        df_all = cargar_datos()
        df_comb = pd.concat([df_all, df_new], ignore_index=True)
        if dedupe:
            df_comb = df_comb.drop_duplicates()
        guardar_datos(df_comb)
        return df_new, "OK"
    except Exception as e:
        return None, str(e)

# ---------------------------
# Modelo ML (clase sin cache en métodos)
# ---------------------------
class ModeloML:
    def __init__(self):
        self.clf = None
        self.le_marca = LabelEncoder()
        self.le_modelo = LabelEncoder()
        self.le_fallo = LabelEncoder()
        self.le_solucion = LabelEncoder()
        self.entrenado = False
        self.veces_entrenado = 0
        self.entrenar_modelo()

    def _safe_label_fit(self, le, values):
        le.fit(values.astype(str))
        return le

    def entrenar_modelo(self):
        df = cargar_datos()
        if df.empty or len(df) < 2:
            self.entrenado = False
            return
        X = df[["Marca","Modelo","Fallo"]].astype(str).fillna("Otros")
        y = df["Solucion"].astype(str).fillna("Otros")
        self.le_marca = self._safe_label_fit(LabelEncoder(), X["Marca"].unique())
        self.le_modelo = self._safe_label_fit(LabelEncoder(), X["Modelo"].unique())
        self.le_fallo = self._safe_label_fit(LabelEncoder(), X["Fallo"].unique())
        self.le_solucion = self._safe_label_fit(LabelEncoder(), y.unique())
        X_enc = X.copy()
        X_enc["Marca"] = self.le_marca.transform(X["Marca"])
        X_enc["Modelo"] = self.le_modelo.transform(X["Modelo"])
        X_enc["Fallo"] = self.le_fallo.transform(X["Fallo"])
        y_enc = self.le_solucion.transform(y)
        try:
            self.clf = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
            self.clf.fit(X_enc, y_enc)
            self.entrenado = True
            self.veces_entrenado += 1
        except Exception as e:
            st.warning(f"Error entrenando modelo: {e}")
            self.entrenado = False

    def _ensure_class_in_encoder(self, le, value):
        if value not in le.classes_:
            le.classes_ = np.append(le.classes_, value)

    def predecir_top3(self, marca, modelo, fallo):
        if not self.entrenado:
            return [], []
        marca = marca.strip() if marca else "Otros"
        modelo = modelo.strip() if modelo else "Otros"
        fallo = fallo.strip() if fallo else "Otros"
        try:
            self._ensure_class_in_encoder(self.le_marca, marca)
            self._ensure_class_in_encoder(self.le_modelo, modelo)
            self._ensure_class_in_encoder(self.le_fallo, fallo)
            X_test = pd.DataFrame([[marca, modelo, fallo]], columns=["Marca","Modelo","Fallo"])
            X_enc = X_test.copy()
            X_enc["Marca"] = self.le_marca.transform(X_test["Marca"])
            X_enc["Modelo"] = self.le_modelo.transform(X_test["Modelo"])
            X_enc["Fallo"] = self.le_fallo.transform(X_test["Fallo"])
            probs = self.clf.predict_proba(X_enc)[0]
            inds = np.argsort(probs)[-3:][::-1]
            top = []
            for i in inds:
                if i < len(self.le_solucion.classes_):
                    sol = self.le_solucion.inverse_transform([i])[0]
                else:
                    sol = "Desconocido"
                top.append((sol, float(probs[i])))
            return top, probs
        except Exception as e:
            st.warning(f"No se pudo predecir: {e}")
            return [], []

    def desaprender_registro(self, marca, modelo, fallo, solucion, fecha=None):
        df = cargar_datos()
        if fecha is not None:
            mask = ~((df["Marca"]==marca)&(df["Modelo"]==modelo)&(df["Fallo"]==fallo)&(df["Solucion"]==solucion)&(df["Fecha"]==fecha))
        else:
            mask = ~((df["Marca"]==marca)&(df["Modelo"]==modelo)&(df["Fallo"]==fallo)&(df["Solucion"]==solucion))
        df_new = df[mask]
        guardar_datos(df_new)
        self.entrenar_modelo()

# Instanciar el modelo de forma cacheada (evita UnhashableParamError)
@st.cache_resource
def get_model():
    return ModeloML()

ml_model = get_model()

# ---------------------------
# UI helpers: chips (botones pequeños)
# ---------------------------
def mostrar_chips(sugerencias, key_prefix, per_row=6, max_rows=4):
    """Muestra botones tipo 'chip'. Retorna el valor seleccionado (o None)."""
    if not sugerencias:
        return None
    max_items = per_row * max_rows
    sugerencias = list(dict.fromkeys(sugerencias))[:max_items]  # quitar duplicados, limitar
    cols = st.columns(per_row)
    for i, val in enumerate(sugerencias):
        c = cols[i % per_row]
        if c.button(val, key=f"{key_prefix}_chip_{i}"):
            return val
    return None

# ---------------------------
# App Layout (Diseño 'Tecnología')
# ---------------------------
st.set_page_config(page_title="Diagnostico con ML", layout="wide")
st.markdown("""
<div style='text-align:center; padding:18px;'>
  <h1 style='color:#0B74DE; margin:0;'>DIAGNOSTICO CON ML</h1>
  <p style='color:gray; margin:0;'>Servicio Técnico Inteligente — Diagnóstico y Registro</p>
</div>
<hr style='border:1px solid #0B74DE; margin-top:12px; margin-bottom:18px;'>
""", unsafe_allow_html=True)

menu = st.sidebar.selectbox("Navegación", ["Inicio","Nuevo Registro","Historial","Preguntar al Modelo","Importar Datos Avanzado"])

# ---------- INICIO ----------
if menu == "Inicio":
    st.subheader("Panel de Control")
    df = cargar_datos()
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros totales", len(df))
    c2.metric("Modelo entrenado (veces)", ml_model.veces_entrenado)
    last_date = df["Fecha"].max() if not df.empty else "-"
    c3.metric("Último registro", last_date)
    st.markdown("---")
    st.write("Últimos 6 registros:")
    if not df.empty:
        st.dataframe(df.tail(6).reset_index(drop=True))
    else:
        st.info("Aún no hay registros.")

# ---------- NUEVO REGISTRO ----------
elif menu == "Nuevo Registro":
    st.subheader("Nuevo Registro")
    df = cargar_datos()

    # sugerencias (chips) para marca
    marcas = sorted(df["Marca"].dropna().unique().tolist()) if not df.empty else []
    modelos = sorted(df["Modelo"].dropna().unique().tolist()) if not df.empty else []
    fallos = sorted(df["Fallo"].dropna().unique().tolist()) if not df.empty else []

    st.markdown("**Sugerencias (Marcas)**")
    sel_m = mostrar_chips(marcas, "chip_marca", per_row=6)
    marca_val = sel_m if sel_m else st.session_state.get("marca_tmp","")
    marca = st.text_input("Marca", value=marca_val)
    st.session_state["marca_tmp"] = marca

    # modelos filtrados por marca (usando datos del CSV)
    modelos_filtrados = modelos
    if marca:
        modelos_filtrados = sorted(df[df["Marca"].str.lower()==marca.lower()]["Modelo"].dropna().unique().tolist())
        if not modelos_filtrados:
            # si no hay modelos exactos para la marca, buscar por coincidencia parcial
            modelos_filtrados = [m for m in modelos if marca.lower() in m.lower()] or modelos
    st.markdown("**Sugerencias (Modelos)**")
    sel_mo = mostrar_chips(modelos_filtrados, "chip_modelo", per_row=6)
    modelo_val = sel_mo if sel_mo else st.session_state.get("modelo_tmp","")
    modelo = st.text_input("Modelo", value=modelo_val)
    st.session_state["modelo_tmp"] = modelo

    st.markdown("**Sugerencias (Fallos)**")
    sel_f = mostrar_chips(fallos, "chip_fallo", per_row=6)
    fallo_val = sel_f if sel_f else st.session_state.get("fallo_tmp","")
    fallo = st.text_input("Fallo", value=fallo_val)
    st.session_state["fallo_tmp"] = fallo

    solucion_val = st.session_state.get("solucion_tmp","")
    solucion = st.text_area("Solución aplicada", value=solucion_val)
    st.session_state["solucion_tmp"] = solucion

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Guardar Registro"):
            m, mo, f, s = marca.strip(), modelo.strip(), fallo.strip(), solucion.strip()
            if not (m and mo and f and s):
                st.error("Todos los campos son obligatorios.")
            else:
                fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df = pd.concat([df, pd.DataFrame([[m,mo,f,s,fecha]], columns=df.columns)], ignore_index=True)
                guardar_datos(df)
                ml_model.entrenar_modelo()
                st.success("Registro guardado y modelo reentrenado.")
                st.session_state['marca_tmp']=''
                st.session_state['modelo_tmp']=''
                st.session_state['fallo_tmp']=''
                st.session_state['solucion_tmp']=''
                st.rerun()
    with col2:
        if st.button("Limpiar campos"):
            st.session_state['marca_tmp']=''
            st.session_state['modelo_tmp']=''
            st.session_state['fallo_tmp']=''
            st.session_state['solucion_tmp']=''
            st.rerun()

# ---------- HISTORIAL ----------
elif menu == "Historial":
    st.subheader("Historial de Registros")
    df_hist = cargar_datos()
    if df_hist.empty:
        st.info("No hay registros todavía.")
    else:
        # filtros
        left, right = st.columns(2)
        q = left.text_input("Buscar (texto libre)")
        marcas_list = ["Todas"] + sorted(df_hist["Marca"].dropna().unique().tolist())
        marca_fil = right.selectbox("Filtrar por Marca", marcas_list)

        df_temp = df_hist.copy()
        if marca_fil != "Todas":
            df_temp = df_temp[df_temp["Marca"]==marca_fil]
        if q:
            df_temp = df_temp[df_temp.apply(lambda r: q.lower() in ' '.join(r.values.astype(str)).lower(), axis=1)]

        st.markdown(f"Registros encontrados: **{len(df_temp)}**")
        st.dataframe(df_temp.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        if len(df_temp) > 0:
            idx = st.number_input("Índice (fila) para Editar/Eliminar (0..n-1):", min_value=0, max_value=max(0,len(df_temp)-1), value=0, step=1)
            selected_row = df_temp.reset_index(drop=True).iloc[int(idx)]
            # Diseño limpio del registro seleccionado (tarjeta)
            st.markdown("### Registro seleccionado")
            st.markdown(
                f"""
                <div style='border:1px solid #e6e6e6; padding:12px; border-radius:8px;'>
                  <div style='display:flex; gap:20px;'>
                    <div><b>Marca:</b><br>{selected_row['Marca']}</div>
                    <div><b>Modelo:</b><br>{selected_row['Modelo']}</div>
                    <div><b>Fallo:</b><br>{selected_row['Fallo']}</div>
                    <div><b>Fecha:</b><br>{selected_row['Fecha']}</div>
                  </div>
                  <div style='margin-top:8px;'><b>Solución:</b><br>{selected_row['Solucion']}</div>
                </div>
                """, unsafe_allow_html=True
            )

            # Botones editar/eliminar con confirmación en página
            c_edit, c_del = st.columns(2)
            # EDITAR: controlar con flag editing
            if 'editing' not in st.session_state:
                st.session_state['editing'] = False
            if 'edit_original' not in st.session_state:
                st.session_state['edit_original'] = None

            with c_edit:
                if st.button("✏️ Editar registro"):
                    st.session_state['editing'] = True
                    st.session_state['edit_original'] = selected_row.to_dict()
                    st.rerun()

            if st.session_state.get('editing', False) and st.session_state.get('edit_original'):
                orig = st.session_state['edit_original']
                st.markdown("### Editar registro")
                with st.form("form_edit2"):
                    nm = st.text_input("Marca", value=orig["Marca"])
                    nmo = st.text_input("Modelo", value=orig["Modelo"])
                    nf = st.text_input("Fallo", value=orig["Fallo"])
                    ns = st.text_area("Solución", value=orig["Solucion"])
                    btn_save = st.form_submit_button("Guardar cambios")
                    btn_cancel = st.form_submit_button("Cancelar")
                    if btn_save:
                        df_all = cargar_datos()
                        mask = (df_all["Marca"]==orig["Marca"]) & (df_all["Modelo"]==orig["Modelo"]) & (df_all["Fallo"]==orig["Fallo"]) & (df_all["Solucion"]==orig["Solucion"]) & (df_all["Fecha"]==orig["Fecha"])
                        if not mask.any():
                            st.error("No se encontró el registro original (datos cambiados).")
                        else:
                            i = df_all[mask].index[0]
                            df_all.at[i,"Marca"] = nm.strip()
                            df_all.at[i,"Modelo"] = nmo.strip()
                            df_all.at[i,"Fallo"] = nf.strip()
                            df_all.at[i,"Solucion"] = ns.strip()
                            guardar_datos(df_all)
                            ml_model.entrenar_modelo()
                            st.success("Registro editado con éxito.")
                            st.session_state['editing'] = False
                            st.session_state['edit_original'] = None
                            st.rerun()
                    if btn_cancel:
                        st.session_state['editing'] = False
                        st.session_state['edit_original'] = None
                        st.rerun()

            # ELIMINAR: confirmar en página con botones
            with c_del:
                if st.button("🗑️ Eliminar registro"):
                    # preparar confirmacion
                    st.session_state['to_delete'] = selected_row.to_dict()
                    st.session_state['confirm_delete'] = False
                    st.rerun()

            if st.session_state.get('to_delete'):
                td = st.session_state['to_delete']
                st.warning(f"¿Confirma eliminar el registro: {td['Marca']} | {td['Modelo']} | {td['Fallo']} | {td['Fecha']} ?")
                dd_col1, dd_col2 = st.columns(2)
                with dd_col1:
                    if st.button("Confirmar eliminación"):
                        ml_model.desaprender_registro(td['Marca'], td['Modelo'], td['Fallo'], td['Solucion'], td['Fecha'])
                        st.success("Registro eliminado.")
                        st.session_state['to_delete'] = None
                        st.rerun()
                with dd_col2:
                    if st.button("Cancelar eliminación"):
                        st.session_state['to_delete'] = None
                        st.info("Eliminación cancelada.")
                        st.rerun()

        # Exportar
        st.markdown("---")
        e1, e2 = st.columns(2)
        with e1:
            if st.button("Exportar Excel"):
                to_export = df_temp.reset_index(drop=True)
                with io.BytesIO() as buffer:
                    to_export.to_excel(buffer, index=False)
                    buffer.seek(0)
                    st.download_button("Descargar Excel", buffer, file_name=f"historial_{datetime.date.today()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with e2:
            if st.button("Exportar PDF"):
                to_export = df_temp.reset_index(drop=True)
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Arial", size=10)
                for _, r in to_export.iterrows():
                    line = f"Marca: {r['Marca']} | Modelo: {r['Modelo']} | Fallo: {r['Fallo']} | Solución: {r['Solucion']} | Fecha: {r['Fecha']}"
                    pdf.multi_cell(0, 6, line)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf.output(tmp.name)
                    tmp.seek(0)
                    with open(tmp.name, "rb") as f:
                        st.download_button("Descargar PDF", f, file_name=f"historial_{datetime.date.today()}.pdf", mime="application/pdf")

# ---------- PREGUNTAR AL MODELO ----------
elif menu == "Preguntar al Modelo":
    st.subheader("Preguntar al Modelo (Top 3)")
    df = cargar_datos()
    marcas = sorted(df["Marca"].dropna().unique().tolist()) if not df.empty else []
    modelos = sorted(df["Modelo"].dropna().unique().tolist()) if not df.empty else []
    fallos = sorted(df["Fallo"].dropna().unique().tolist()) if not df.empty else []

    st.markdown("**Sugerencias (Marca)**")
    sel_pm = mostrar_chips(marcas, "pred_chip_marca", per_row=6)
    pm_val = sel_pm if sel_pm else st.session_state.get("pred_marca","")
    pm = st.text_input("Marca", value=pm_val)
    st.session_state["pred_marca"] = pm

    st.markdown("**Sugerencias (Modelo)**")
    modelos_fil = modelos
    if pm:
        modelos_fil = sorted(df[df["Marca"].str.lower()==pm.lower()]["Modelo"].dropna().unique().tolist())
        if not modelos_fil:
            modelos_fil = [m for m in modelos if pm.lower() in m.lower()] or modelos
    sel_pmo = mostrar_chips(modelos_fil, "pred_chip_modelo", per_row=6)
    pmo_val = sel_pmo if sel_pmo else st.session_state.get("pred_modelo","")
    pmo = st.text_input("Modelo", value=pmo_val)
    st.session_state["pred_modelo"] = pmo

    st.markdown("**Sugerencias (Fallo)**")
    sel_pf = mostrar_chips(fallos, "pred_chip_fallo", per_row=6)
    pf_val = sel_pf if sel_pf else st.session_state.get("pred_fallo","")
    pf = st.text_input("Fallo", value=pf_val)
    st.session_state["pred_fallo"] = pf

    if st.button("Predecir Top 3"):
        if not (pm and pmo and pf):
            st.error("Complete Marca, Modelo y Fallo antes de predecir.")
        else:
            top3, probs = ml_model.predecir_top3(pm, pmo, pf)
            if not top3:
                st.info("Modelo no entrenado o sin datos suficientes.")
            else:
                st.subheader("Top 3 soluciones posibles:")
                for sol, p in top3:
                    st.write(f"🔧 **{sol}** — {p*100:.2f}%")

                if st.button("Guardar mejor solución como registro"):
                    mejor = top3[0][0]
                    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df = cargar_datos()
                    df = pd.concat([df, pd.DataFrame([[pm,pmo,pf,mejor,fecha]], columns=df.columns)], ignore_index=True)
                    guardar_datos(df)
                    ml_model.entrenar_modelo()
                    st.success("Predicción guardada como registro.")
                    st.rerun()

# ---------- IMPORTAR DATOS AVANZADO ----------
elif menu == "Importar Datos Avanzado":
    st.subheader("Importación avanzada (CSV / Excel)")
    uploaded = st.file_uploader("Subir archivo (.csv, .xls, .xlsx)", type=['csv','xls','xlsx'])
    dedupe = st.checkbox("Eliminar duplicados al importar (recomendado)", value=True)
    if uploaded:
        df_new, msg = importar_archivo(uploaded, dedupe=dedupe)
        if df_new is None:
            st.error(f"Error: {msg}")
        else:
            st.success(f"Importado {len(df_new)} filas. {msg}")
            st.dataframe(df_new.head(50))

# Footer
st.markdown("---")
st.caption("App: Diagnostico con ML")
