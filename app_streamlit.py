import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import datetime
import os
import numpy as np
import io

# ---------------------------
# Config y archivo de datos
# ---------------------------
DATA_FILE = "registro_telefonos.csv"

# Inicializar el archivo de datos si no existe
if not os.path.exists(DATA_FILE):
    df_init = pd.DataFrame(columns=["Marca", "Modelo", "Fallo", "Solucion", "Fecha"])
    df_init.to_csv(DATA_FILE, index=False)

@st.cache_data
def cargar_datos():
    """Carga los datos del CSV. Usa caché para evitar recarga constante."""
    try:
        df = pd.read_csv(DATA_FILE)
        # Asegurarse de que las columnas clave sean de tipo string para el ML
        for col in ["Marca", "Modelo", "Fallo", "Solucion"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df
    except Exception:
        return pd.DataFrame(columns=["Marca", "Modelo", "Fallo", "Solucion", "Fecha"])

def guardar_datos(df):
    """Guarda los datos en el CSV y borra la caché para forzar la recarga."""
    df.to_csv(DATA_FILE, index=False)
    # Borrar la caché para que el modelo y la vista se actualicen
    cargar_datos.clear()

# ---------------------------
# Modelo ML (Marca+Modelo+Fallo -> Solución)
# ---------------------------
class ModeloML:
    def __init__(self):
        self.clf = RandomForestClassifier(random_state=42)
        self.le_marca = LabelEncoder()
        self.le_modelo = LabelEncoder()
        self.le_fallo = LabelEncoder()
        self.le_solucion = LabelEncoder()
        self.entrenado = False
        self.veces_entrenado = 0
        self.entrenar_modelo()

    def _safe_label_fit(self, le, values):
        # Ajusta LabelEncoder incluso si values tiene un único elemento
        le.fit(values.astype(str))
        return le

    @st.cache_resource(show_spinner="Reentrenando Modelo ML...")
    def entrenar_modelo(_self):
        df = cargar_datos()
        if df.empty or len(df) < 2: # Se necesitan al menos 2 registros para un clasificador
            _self.entrenado = False
            _self.veces_entrenado = 0
            return _self

        X = df[["Marca", "Modelo", "Fallo"]].astype(str).fillna("Otros")
        y = df["Solucion"].astype(str).fillna("Otros")

        # Aseguramos que "Otros" exista para manejar valores nuevos
        X["Marca"] = X["Marca"].apply(lambda x: x if x.strip() else "Otros")
        X["Modelo"] = X["Modelo"].apply(lambda x: x if x.strip() else "Otros")
        X["Fallo"] = X["Fallo"].apply(lambda x: x if x.strip() else "Otros")
        y = y.apply(lambda x: x if x.strip() else "Otros")

        # Fit encoders
        _self.le_marca = _self._safe_label_fit(LabelEncoder(), X["Marca"].unique())
        _self.le_modelo = _self._safe_label_fit(LabelEncoder(), X["Modelo"].unique())
        _self.le_fallo = _self._safe_label_fit(LabelEncoder(), X["Fallo"].unique())
        _self.le_solucion = _self._safe_label_fit(LabelEncoder(), y.unique())

        # Transformar
        X_enc = X.copy()
        X_enc["Marca"] = _self.le_marca.transform(X["Marca"])
        X_enc["Modelo"] = _self.le_modelo.transform(X["Modelo"])
        X_enc["Fallo"] = _self.le_fallo.transform(X["Fallo"])
        y_enc = _self.le_solucion.transform(y)

        # Entrenar
        try:
            _self.clf = RandomForestClassifier(random_state=42)
            _self.clf.fit(X_enc, y_enc)
            _self.entrenado = True
            _self.veces_entrenado += 1
            st.toast(f"Modelo reentrenado. Veces: {_self.veces_entrenado}", icon='🧠')
        except Exception as e:
            st.error(f"Error entrenando modelo: {e}")
            _self.entrenado = False
        return _self


    def _ensure_class_in_encoder(self, le, value):
        # Si el valor no está en le.classes_, lo añade para poder transformarlo.
        # En Streamlit, esto es clave ya que el objeto se mantiene en memoria
        if value not in le.classes_:
            le.classes_ = np.append(le.classes_, value)

    def predecir(self, marca, modelo, fallo):
        if not self.entrenado:
            return "No hay suficientes datos para predecir.", 0.0

        # Normalizar y manejar vacíos
        marca = str(marca).strip() if marca and str(marca).strip() else "Otros"
        modelo = str(modelo).strip() if modelo and str(modelo).strip() else "Otros"
        fallo = str(fallo).strip() if fallo and str(fallo).strip() else "Otros"

        # Asegurar que encoder tenga las clases
        try:
            self._ensure_class_in_encoder(self.le_marca, marca)
            self._ensure_class_in_encoder(self.le_modelo, modelo)
            self._ensure_class_in_encoder(self.le_fallo, fallo)

            X_test = pd.DataFrame([[marca, modelo, fallo]], columns=["Marca", "Modelo", "Fallo"])
            X_test["Marca"] = X_test["Marca"].astype(str)
            X_test["Modelo"] = X_test["Modelo"].astype(str)
            X_test["Fallo"] = X_test["Fallo"].astype(str)

            X_enc = X_test.copy()
            X_enc["Marca"] = self.le_marca.transform(X_test["Marca"])
            X_enc["Modelo"] = self.le_modelo.transform(X_test["Modelo"])
            X_enc["Fallo"] = self.le_fallo.transform(X_test["Fallo"])

            pred_enc = self.clf.predict(X_enc)[0]
            probs = self.clf.predict_proba(X_enc)[0]
            # Obtener la probabilidad de la clase predicha
            pred_idx = np.where(self.clf.classes_ == pred_enc)[0][0]
            prob = probs[pred_idx]

            sol = self.le_solucion.inverse_transform([pred_enc])[0]
            return sol, float(prob)
        except Exception as e:
            # st.exception(e) # Descomentar para debug
            return "No se pudo predecir", 0.0

    def desaprender_registro(self, marca, modelo, fallo, solucion):
        df = cargar_datos()
        mask = ~((df["Marca"]==marca) & (df["Modelo"]==modelo) & (df["Fallo"]==fallo) & (df["Solucion"]==solucion))
        df_new = df[mask]
        guardar_datos(df_new)
        # El reentrenamiento ocurre automáticamente al acceder a ml_model
        # ya que la instancia está en st.session_state

# ---------------------------
# Datos base de la app (se mantienen igual)
# ---------------------------
MARCAS_COMUNES = ["Samsung", "Apple", "Xiaomi", "Motorola", "Otros"]
MODELOS_POR_MARCA = {
    "Samsung": ["A10", "A20", "A30", "A50", "S20", "S21", "S22", "S23", "Z Flip", "Z Fold 2", "Otros"],
    "Apple": ["iPhone 8", "iPhone X", "iPhone 11", "iPhone 12", "iPhone 13", "iPhone 14", "iPhone 15", "Otros"],
    "Xiaomi": ["Mi 11", "Xiaomi 12", "Redmi Note 10", "Redmi Note 11", "Poco F3", "Poco X3", "Otros"],
    "Motorola": ["Moto G7", "Moto G8", "Moto G100", "Edge 30", "Edge 40", "Razr 40", "Otros"],
    "Otros": ["Otros"]
}
FALLOS_COMUNES = ["Pantalla rota", "No enciende", "Batería", "Conector de carga", "Otros"]

# ---------------------------
# Inicialización y UI de Streamlit
# ---------------------------
def init_state():
    if 'ml_model' not in st.session_state:
        # Inicializar y entrenar el modelo (con caché)
        st.session_state.ml_model = ModeloML().entrenar_modelo()
        # Asegurar que el DataFrame se cargue y se almacene
        st.session_state.df_hist = cargar_datos()

# --- Funciones de vista ---

def nuevo_registro_ui():
    st.title("➕ Nuevo Registro")

    with st.form("registro_form", clear_on_submit=True):
        st.subheader("Detalles del Teléfono")

        # Marca
        marca_choice = st.selectbox("Marca Común:", MARCAS_COMUNES, key="reg_marca_choice")
        marca_final = marca_choice
        if marca_choice == "Otros":
            marca_otro = st.text_input("Ingrese otra marca (obligatorio):", key="reg_marca_otro")
            marca_final = marca_otro

        # Modelo
        modelos_disponibles = MODELOS_POR_MARCA.get(marca_choice, ["Otros"])
        modelo_choice = st.selectbox("Modelo Común:", modelos_disponibles, key="reg_modelo_choice")
        modelo_final = modelo_choice
        if modelo_choice == "Otros":
            modelo_otro = st.text_input("Ingrese otro modelo (obligatorio):", key="reg_modelo_otro")
            modelo_final = modelo_otro

        # Fallo
        fallo_choice = st.selectbox("Fallo Común:", FALLOS_COMUNES, key="reg_fallo_choice")
        fallo_final = fallo_choice
        if fallo_choice == "Otros":
            fallo_otro = st.text_area("Describa el fallo (obligatorio):", key="reg_fallo_otro")
            fallo_final = fallo_otro

        # Solución
        solucion = st.text_area("Solución aplicada (obligatorio):", key="reg_solucion")

        submitted = st.form_submit_button("Guardar Registro")
        if submitted:
            # Manejar placeholders de entrada (si no se han completado)
            if marca_final.startswith("Ingrese otra marca...") or not marca_final:
                marca_final = marca_choice if marca_choice != "Otros" else ""
            if modelo_final.startswith("Ingrese otro modelo...") or not modelo_final:
                modelo_final = modelo_choice if modelo_choice != "Otros" else ""
            if fallo_final.startswith("Describa el fallo...") or not fallo_final:
                fallo_final = fallo_choice if fallo_choice != "Otros" else ""

            # Validar
            if not (marca_final and modelo_final and fallo_final and solucion):
                st.error("Todos los campos con valores finales son obligatorios.")
            else:
                fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                nuevo_registro = pd.DataFrame([[marca_final, modelo_final, fallo_final, solucion, fecha]],
                                               columns=["Marca", "Modelo", "Fallo", "Solucion", "Fecha"])

                df = cargar_datos()
                df = pd.concat([df, nuevo_registro], ignore_index=True)
                guardar_datos(df)

                # Reentrenar el modelo
                st.session_state.ml_model = st.session_state.ml_model.entrenar_modelo()

                st.success("✅ Registro guardado correctamente. Modelo reentrenado.")
                st.session_state.df_hist = cargar_datos() # Actualizar estado
                st.experimental_rerun() # Para limpiar el formulario y actualizar la UI


def ver_historial_ui():
    st.title("📚 Historial de Registros")
    df_hist = cargar_datos()

    if df_hist.empty:
        st.info("Aún no hay registros en el historial.")
        return

    st.subheader("Filtros")
    col1, col2 = st.columns(2)

    todas_marcas = ["Todas"] + sorted(df_hist["Marca"].unique().tolist())
    marca_filtro = col1.selectbox("Filtrar por Marca:", todas_marcas, key="hist_marca_filtro")

    df_filtrado = df_hist.copy()
    if marca_filtro != "Todas":
        df_filtrado = df_filtrado[df_filtrado["Marca"] == marca_filtro]

    # Mostrar modelos solo de la marca seleccionada
    modelos_en_csv = ["Todos"]
    if marca_filtro != "Todas":
        modelos_en_csv += sorted(df_filtrado["Modelo"].unique().tolist())

    modelo_filtro = col2.selectbox("Filtrar por Modelo:", modelos_en_csv, key="hist_modelo_filtro")

    if modelo_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Modelo"] == modelo_filtro]

    st.subheader("Registros Filtrados")
    st.dataframe(df_filtrado, height=350)

    # --- Opciones de Edición/Exportación ---
    st.subheader("Acciones")
    col_edit, col_del, col_exp_excel, col_exp_pdf = st.columns([1, 1, 1, 1])

    # 1. Eliminar
    with col_del:
        if st.button("🗑️ Eliminar Registro"):
            indices_eliminar = st.session_state.df_hist.index[df_hist.apply(tuple, axis=1).isin(df_filtrado.apply(tuple, axis=1))].tolist()
            if indices_eliminar:
                # Mostrar selección para eliminar con clave única
                opciones = [f"{row['Marca']} - {row['Modelo']} - {row['Fallo']} ({idx})"
                            for idx, row in df_filtrado.reset_index().iterrows()]
                st.warning("Seleccione el registro exacto a eliminar:")
                registro_a_eliminar = st.selectbox("Selección:", opciones, key="reg_del_select")
                idx_sel = opciones.index(registro_a_eliminar)

                if st.button("Confirmar Eliminación", key="confirm_del_btn"):
                    registro_original = df_filtrado.iloc[idx_sel]
                    st.session_state.ml_model.desaprender_registro(
                        registro_original["Marca"],
                        registro_original["Modelo"],
                        registro_original["Fallo"],
                        registro_original["Solucion"]
                    )
                    st.success("Registro eliminado y modelo reentrenado.")
                    st.session_state.df_hist = cargar_datos() # Forzar actualización
                    st.experimental_rerun() # Para recargar la tabla

            else:
                st.warning("Seleccione un registro para eliminar (puede usar los filtros).")

    # 2. Exportar Excel
    with col_exp_excel:
        if not df_filtrado.empty:
            excel_data = io.BytesIO()
            df_filtrado.to_excel(excel_data, index=False)
            st.download_button(
                label="⬇️ Exportar Excel",
                data=excel_data.getvalue(),
                file_name=f'historial_telefonos_{datetime.date.today()}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

    # 3. Exportar PDF
    with col_exp_pdf:
        if not df_filtrado.empty:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, 'Historial de Registros de Teléfonos', 0, 1, 'C')
            pdf.set_font("Arial", size=10)

            for _, row in df_filtrado.iterrows():
                line = f"Marca: {row['Marca']}, Modelo: {row['Modelo']}, Fallo: {row['Fallo']}, Solución: {row['Solucion']}, Fecha: {row['Fecha']}"
                pdf.multi_cell(0, 5, txt=line)
                pdf.ln(1) # Pequeño espacio

            pdf_output = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="⬇️ Exportar PDF",
                data=pdf_output,
                file_name=f'historial_telefonos_{datetime.date.today()}.pdf',
                mime='application/pdf'
            )


def preguntar_modelo_ui():
    st.title("🧠 Pregúntale al Modelo ML")

    # Obtener todas las marcas/modelos/fallos conocidos por el modelo + "Otros"
    df = cargar_datos()
    marcas_totales = sorted(df["Marca"].unique().tolist() + ["Otros"])
    fallos_totales = sorted(df["Fallo"].unique().tolist() + ["Otros"])

    with st.form("prediccion_form"):
        col1, col2 = st.columns(2)

        # Marca
        marca_input = col1.selectbox("Marca:", marcas_totales, key="pred_marca_select")
        marca_final = marca_input
        if marca_input == "Otros":
            marca_otro = col2.text_input("Especifique Marca:", key="pred_marca_otro")
            marca_final = marca_otro

        # Modelo
        modelos_conocidos = sorted(df[df["Marca"] == marca_input]["Modelo"].unique().tolist() + ["Otros"])
        modelo_input = col1.selectbox("Modelo:", modelos_conocidos, key="pred_modelo_select")
        modelo_final = modelo_input
        if modelo_input == "Otros":
            modelo_otro = col2.text_input("Especifique Modelo:", key="pred_modelo_otro")
            modelo_final = modelo_otro

        # Fallo
        fallo_input = col1.selectbox("Fallo:", fallos_totales, key="pred_fallo_select")
        fallo_final = fallo_input
        if fallo_input == "Otros":
            fallo_otro = col2.text_input("Especifique Fallo:", key="pred_fallo_otro")
            fallo_final = fallo_otro

        submitted = st.form_submit_button("Predecir Solución")

        if submitted:
            # Reemplazar valores de placeholder/vacío por "Otros"
            marca = marca_final.strip() if marca_final and marca_final.strip() else "Otros"
            modelo = modelo_final.strip() if modelo_final and modelo_final.strip() else "Otros"
            fallo = fallo_final.strip() if fallo_final and fallo_final.strip() else "Otros"

            if not (marca and modelo and fallo and marca != "Otros" and modelo != "Otros" and fallo != "Otros" and (marca_input != "Otros" or marca_final) and (modelo_input != "Otros" or modelo_final) and (fallo_input != "Otros" or fallo_final)):
                st.error("Por favor, complete Marca, Modelo y Fallo, o especifique los detalles si selecciona 'Otros'.")
                return

            sol, prob = st.session_state.ml_model.predecir(marca, modelo, fallo)

            st.markdown("---")
            st.subheader("Resultado de la Predicción")
            if st.session_state.ml_model.entrenado:
                st.info(f"**Predicción de Solución:** **{sol}**")
                st.metric(label="Confiabilidad", value=f"{prob*100:.2f}%")
                st.caption(f"Modelo entrenado **{st.session_state.ml_model.veces_entrenado}** veces.")
            else:
                st.warning("⚠️ El modelo no está entrenado (se necesitan al menos 2 registros).")
                st.write(sol) # Muestra el mensaje de "No hay suficientes datos..."


def main():
    # Inicialización del estado (incluyendo el modelo ML)
    init_state()

    # Configuración de la página (equivalente al root de Tkinter)
    st.set_page_config(
        page_title="Servicio Técnico - Diagnóstico ML",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.title("🛠️ Menú Principal")
    # Mapeo de páginas
    page_options = {
        "Nuevo Registro": nuevo_registro_ui,
        "Ver Historial": ver_historial_ui,
        "Pregúntale al Modelo": preguntar_modelo_ui,
    }

    selection = st.sidebar.radio("Elige una opción:", list(page_options.keys()))

    # Llama a la función de la página seleccionada
    page_options[selection]()

if __name__ == "__main__":
    main()