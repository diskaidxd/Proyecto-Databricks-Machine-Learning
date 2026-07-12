import os
import numpy as np
import streamlit as st
from databricks.sdk import WorkspaceClient


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Evaluación Crediticia",
    page_icon="🏦",
    layout="centered"
)

# El nombre del endpoint se obtiene desde app.yaml.
# Databricks Apps administra la autenticación automáticamente.
ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT", "credit_scoring")

# Cliente de Databricks.
# Dentro de Databricks Apps utiliza la identidad propia de la aplicación.
workspace = WorkspaceClient()


# ============================================================
# ESTILOS DE LA APLICACIÓN
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main-title {
            text-align: center;
            font-size: 2.3rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            text-align: center;
            color: #5f6b7a;
            margin-bottom: 1.8rem;
        }

        .info-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            background: rgba(120, 120, 120, 0.08);
            border: 1px solid rgba(120, 120, 120, 0.18);
            margin-bottom: 1.2rem;
        }

        div.stButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 10px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCIÓN PARA CONSULTAR EL MODELO
# ============================================================

def obtener_prediccion(
    income: float,
    loan_amount: float,
    term: int,
    credit_history: int
) -> int:
    """
    Prepara los datos exactamente como fueron utilizados
    durante el entrenamiento y consulta el Serving Endpoint.
    """

    # Las mismas transformaciones utilizadas en el notebook de entrenamiento.
    registro_modelo = {
        "log_income": float(np.log1p(income)),
        "log_loan_amount": float(np.log1p(loan_amount)),
        "term_binary": 1 if term == 60 else 0,
        "credit_history": int(credit_history)
    }

    respuesta = workspace.serving_endpoints.query(
        name=ENDPOINT_NAME,
        dataframe_records=[registro_modelo]
    )

    # Convertimos la respuesta del SDK a un diccionario.
    respuesta_dict = respuesta.as_dict()

    predicciones = respuesta_dict.get("predictions")

    if not predicciones:
        raise ValueError(
            f"El endpoint respondió, pero no devolvió predicciones: {respuesta_dict}"
        )

    return int(predicciones[0])


def crear_observaciones(
    income: float,
    loan_amount: float,
    term: int,
    credit_history: int
) -> list[str]:
    """
    Genera comentarios descriptivos para acompañar la predicción.
    No reemplaza una evaluación crediticia profesional.
    """

    observaciones = []
    relacion_prestamo_ingreso = loan_amount / income

    if relacion_prestamo_ingreso >= 0.70:
        observaciones.append(
            "El préstamo representa una parte alta del ingreso declarado."
        )
    elif relacion_prestamo_ingreso >= 0.40:
        observaciones.append(
            "El préstamo representa una parte moderada del ingreso declarado."
        )
    else:
        observaciones.append(
            "El préstamo representa una parte relativamente baja del ingreso declarado."
        )

    if credit_history == 0:
        observaciones.append(
            "El registro indica un historial crediticio desfavorable o no disponible."
        )
    else:
        observaciones.append(
            "El registro indica un historial crediticio favorable."
        )

    if term == 60:
        observaciones.append(
            "El plazo seleccionado es de 60 meses."
        )
    else:
        observaciones.append(
            "El plazo seleccionado es de 36 meses."
        )

    return observaciones


# ============================================================
# INTERFAZ
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Evaluación de Riesgo Crediticio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ingresa los datos del solicitante para obtener una predicción del modelo.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        <b>Problema del banco:</b> identificar anticipadamente solicitudes
        con mayor posibilidad de incumplimiento para apoyar la evaluación
        del riesgo crediticio.
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("credit_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Ingreso del solicitante",
            min_value=1.0,
            value=50000.0,
            step=1000.0,
            help="Ingreso registrado para el cliente."
        )

        term = st.selectbox(
            "Plazo del préstamo",
            options=[36, 60],
            format_func=lambda value: f"{value} meses"
        )

    with col2:
        loan_amount = st.number_input(
            "Monto del préstamo",
            min_value=1.0,
            value=15000.0,
            step=500.0,
            help="Monto que solicita el cliente."
        )

        credit_history_text = st.selectbox(
            "Historial crediticio",
            options=["Favorable", "Desfavorable o no disponible"]
        )

    submitted = st.form_submit_button(
        "Evaluar solicitud",
        use_container_width=True
    )


# ============================================================
# RESULTADO
# ============================================================

if submitted:
    credit_history = (
        1 if credit_history_text == "Favorable" else 0
    )

    with st.spinner("Consultando el modelo desplegado..."):
        try:
            prediction = obtener_prediccion(
                income=income,
                loan_amount=loan_amount,
                term=term,
                credit_history=credit_history
            )

            st.divider()
            st.subheader("Resultado del modelo")

            if prediction == 1:
                st.error("⚠️ Mayor riesgo estimado de incumplimiento")
                st.write(
                    "El modelo clasificó esta solicitud dentro del grupo "
                    "con mayor riesgo estimado."
                )
            else:
                st.success("✅ Menor riesgo estimado de incumplimiento")
                st.write(
                    "El modelo clasificó esta solicitud dentro del grupo "
                    "con menor riesgo estimado."
                )

            st.subheader("Comentarios sobre la solicitud")

            for comentario in crear_observaciones(
                income=income,
                loan_amount=loan_amount,
                term=term,
                credit_history=credit_history
            ):
                st.write(f"• {comentario}")

            st.caption(
                "Demostración educativa. La predicción es una señal de apoyo "
                "y no debe utilizarse por sí sola para aprobar o rechazar un crédito."
            )

        except Exception as error:
            st.error(
                "No se pudo obtener la predicción. Verifica que el endpoint "
                "esté en estado READY y que la aplicación tenga permiso CAN QUERY."
            )

            with st.expander("Ver detalle técnico"):
                st.code(str(error))
