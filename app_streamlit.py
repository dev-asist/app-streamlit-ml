import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import datetime
import os
import numpy as np

# ---------------------------
# Config y archivo de datos
# ---------------------------
DATA_FILE = "registro_telefonos.csv"

if not os.path.exists(DATA_FILE):
    df_init = pd.DataFrame(columns=["Marca", "Modelo", "Fallo", "Solucion", "Fecha"])
    df_init.to_csv(DATA_FILE, index=False)

def cargar_datos():
    try:
        return pd.read_csv(DATA_FILE)
    except Exception:
        return pd.DataFrame(columns=["Marca", "Modelo", "Fallo", "Solucion", "Fecha"])

def guardar_datos(df):
    df.to_csv(DATA_FILE, index=False)

# ---------------------------
# Modelo ML
# ---------------------------
class ModeloML:
    def __init__(self):
        self.clf = RandomForestClassifier()
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
        if df.empty:
            self.entrenado = False
            return
        X = df[["Marca", "Modelo", "Fallo"]].astype(str).fillna("Otros")
        y = df["Solucion"].astype(str).fillna("Otros")

        X["Marca"] = X["Marca"].replace("", "Otros")
        X["Modelo"] = X["Modelo"].replace("", "Otros")
        X["Fallo"] = X["Fallo"].replace("", "Otros")
        y = y.replace("", "Otros")

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
            self.clf = RandomForestClassifier()
            self.clf.fit(X_enc, y_enc)
            self.entrenado = True
            self.veces_entrenado += 1
        except Exception as e:
            st.error(f"Error entrenando modelo: {e}")
            self.entrenado = False

    def _ensure_class_in_encoder(self, le, value):
        if value not in le.classes_:
            le.classes_ = np.append(le.classes_, value)

    def predecir(self, marca, modelo, fallo):
        if not self.entrenado:
            return "No hay datos para predecir.", 0.0

        marca = marca.strip() if marca else "Otros"
        modelo = modelo.strip() if modelo else "Otros"
        fallo = fallo.strip() if fallo else "Otros"

        try:
            self._ensure_class_in_encoder(self.le_marca, marca)
            self._ensure_class_in_encoder(self.le_modelo, modelo)
            self._ensure_class_in_encoder(self.le_fallo, fallo)

            X_test = pd.DataFrame([[marca, modelo, fallo]], columns=["Marca", "Modelo", "Fallo"])
            X_enc = X_test.copy()
            X_enc["Marca"] = self.le_marca.transform(X_test["Marca"])
            X_enc["Modelo"] = self.le_modelo.transform(X_test["Modelo"])
            X_enc["Fallo"] = self.le_fallo.transform(X_test["Fallo"])

            pred_enc = self.clf.predict(X_enc)[0]
            probs = self.clf.predict_proba(X_enc)[0]
            prob = max(probs) if len(probs) else 0.0
            sol = self.le_solucion.inverse_transform([pred_enc])[0]
            return sol, float(prob)
        except Exception as e:
            return f"No se pudo predecir: {e}", 0.0

    def desaprender_registro(self, marca, modelo, fallo, solucion):
        df = cargar_datos()
        mask = ~((df["Marca"]==marca) & (df["Modelo"]==modelo) & (df["Fallo"]==fallo) & (df["Solucion"]==solucion))
        df_new = df[mask]
        guardar_datos(df_new)
        self.entrenar_modelo()

ml_model = ModeloML()

# ---------------------------
# Datos de marcas y modelos
# ---------------------------
marcas_comunes = ["Samsung", "Apple", "Xiaomi", "Motorola", "Otros"]
modelos_por_marca = {
    "Samsung": ["A10","A20","A30","A50","S10","S20","Note 10","Z Flip","Otros"],
    "Apple": ["iPhone 6","iPhone 7","iPhone 8","iPhone X","iPhone 11","iPhone 12","Otros"],
    "Xiaomi": ["Mi 9","Mi 10","Mi 11","Redmi Note 8","Redmi Note 9","Otros"],
    "Motorola": ["Moto G7","Moto G8","Edge","Razr","Otros"],
    "Otros": ["Otros"]
}
fallos_comunes = ["Pantalla rota", "No enciende", "Batería", "Conector de carga", "Otros"]

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Servicio Técnico ML", layout="wide")

menu = ["Inicio", "Nuevo Registro", "Historial", "Preguntar al Modelo"]
choice = st.sidebar.radio("Ir a:", menu)

# ---------------------------
# Inicio
# ---------------------------
if choice == "Inicio":
    st.title("Servicio Técnico - Diagnóstico ML")
    st.write("Use el menú lateral para navegar entre las opciones.")

# ---------------------------
# Nuevo Registro
# ---------------------------
elif choice == "Nuevo Registro":
    st.header("Nuevo Registro")

    marca = st.selectbox("Marca", marcas_comunes)
    modelo = st.selectbox("Modelo", modelos_por_marca.get(marca, ["Otros"]))
    fallo = st.selectbox("Fallo", fallos_comunes)
    solucion = st.text_input("Solución")

    # Inputs “Otros”
    if marca=="Otros":
        marca = st.text_input("Ingrese otra marca")
    if modelo=="Otros":
        modelo = st.text_input("Ingrese otro modelo")
    if fallo=="Otros":
        fallo = st.text_input("Describa el fallo")

    if st.button("Guardar Registro"):
        if not marca or not modelo or not fallo or not solucion:
            st.error("Todos los campos son obligatorios")
        else:
            df = cargar_datos()
            fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.concat([df, pd.DataFrame([[marca, modelo, fallo, solucion, fecha]], columns=df.columns)], ignore_index=True)
            guardar_datos(df)
            ml_model.entrenar_modelo()
            st.success("Registro guardado correctamente")

# ---------------------------
# Historial
# ---------------------------
elif choice == "Historial":
    st.header("Historial de Registros")
    df_hist = cargar_datos()

    marcas_unicas = ["Todas"] + sorted(df_hist["Marca"].dropna().unique().tolist())
    marca_filtro = st.selectbox("Filtrar por Marca", marcas_unicas)

    if marca_filtro != "Todas":
        df_hist = df_hist[df_hist["Marca"]==marca_filtro]
        modelos_unicos = ["Todos"] + sorted(df_hist["Modelo"].unique().tolist())
    else:
        modelos_unicos = ["Todos"]

    modelo_filtro = st.selectbox("Filtrar por Modelo", modelos_unicos)
    if modelo_filtro != "Todos":
        df_hist = df_hist[df_hist["Modelo"]==modelo_filtro]

    st.dataframe(df_hist)

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_idx = st.number_input("Número de fila para Editar/Eliminar", min_value=0, max_value=len(df_hist)-1, step=1)
    with col2:
        if st.button("Eliminar Registro"):
            row = df_hist.iloc[sel_idx]
            ml_model.desaprender_registro(row["Marca"], row["Modelo"], row["Fallo"], row["Solucion"])
            st.success("Registro eliminado")
    with col3:
        if st.button("Exportar Excel"):
            df_hist.to_excel("exportado.xlsx", index=False)
            st.success("Datos exportados a exportado.xlsx")
        if st.button("Exportar PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=11)
            for _, row in df_hist.iterrows():
                line = f"Marca: {row['Marca']}, Modelo: {row['Modelo']}, Fallo: {row['Fallo']}, Solución: {row['Solucion']}, Fecha: {row['Fecha']}"
                pdf.multi_cell(0, 8, txt=line)
            pdf.output("exportado.pdf")
            st.success("Datos exportados a exportado.pdf")

# ---------------------------
# Preguntar al Modelo
# ---------------------------
elif choice == "Preguntar al Modelo":
    st.header("Pregúntale al Modelo")
    marca = st.selectbox("Marca", marcas_comunes)
    modelo = st.selectbox("Modelo", modelos_por_marca.get(marca, ["Otros"]))
    fallo = st.selectbox("Fallo", fallos_comunes)

    # Inputs “Otros”
    if marca=="Otros":
        marca = st.text_input("Ingrese otra marca")
    if modelo=="Otros":
        modelo = st.text_input("Ingrese otro modelo")
    if fallo=="Otros":
        fallo = st.text_input("Describa el fallo")

    if st.button("Predecir"):
        if not marca or not modelo or not fallo:
            st.error("Complete todos los campos")
        else:
            sol, prob = ml_model.predecir(marca, modelo, fallo)
            st.success(f"Predicción de Solución: {sol}\nConfiabilidad: {prob*100:.2f}%\nModelo entrenado {ml_model.veces_entrenado} veces")
